"""Backfill S&P 500 companies into LexDrift.

Fetches SEC filings for a configurable set of tickers, ingests them into the
database, and optionally runs drift analysis on each filing.

Usage examples
--------------
    # Backfill all ~30 major S&P 500 tickers (default)
    python scripts/backfill.py --tickers sp500

    # Specific tickers, 3 filings each, with analysis
    python scripts/backfill.py --tickers AAPL,MSFT,GOOGL --limit 3 --analyze

    # 10-Q filings instead of 10-K
    python scripts/backfill.py --tickers sp500 --form-type 10-Q --limit 2
"""

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Ensure the src directory is on the import path when running as a script
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from lexdrift.config import settings
from lexdrift.db.models import Base, Company, Filing, Section
from lexdrift.edgar.filings import (
    build_document_url,
    download_filing,
    get_filing_metadata,
    parse_filing_list,
)
from lexdrift.edgar.parser import parse_filing
from lexdrift.edgar.tickers import lookup_ticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backfill")

# ---------------------------------------------------------------------------
# Hardcoded S&P 500 tickers across major sectors
# ---------------------------------------------------------------------------
SP500_TICKERS: list[str] = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    # Finance
    "JPM", "BAC", "GS", "WFC", "BRK-B",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "WMT", "PG", "KO", "MCD", "NKE",
    # Industrial
    "CAT", "BA", "HON", "GE",
    # Energy
    "XOM", "CVX",
    # Telecom
    "VZ", "T",
    # Real Estate
    "AMT",
    # Utilities
    "NEE",
]

# SEC asks for <= 10 requests/second; we stay well below that.
SEC_REQUEST_DELAY = 0.15  # seconds between HTTP requests


# ---------------------------------------------------------------------------
# Synchronous DB helpers (same pattern as workers)
# ---------------------------------------------------------------------------

def _make_sync_url(async_url: str) -> str:
    """Convert an async SQLAlchemy URL to its synchronous equivalent."""
    url = async_url.replace("+aiosqlite", "")
    url = url.replace("+asyncpg", "+psycopg2")
    return url


_sync_engine = create_engine(_make_sync_url(settings.database_url), echo=False, future=True)
SyncSessionFactory = sessionmaker(bind=_sync_engine, class_=Session, expire_on_commit=False)

Base.metadata.create_all(_sync_engine)


def _run_async(coro):
    """Run an async coroutine from synchronous code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------

def ingest_company(ticker: str, form_type: str, limit: int) -> list[int]:
    """Ingest filings for a single company. Returns list of new filing IDs."""
    # Resolve ticker -> CIK
    info = _run_async(lookup_ticker(ticker))
    if not info:
        logger.warning("Ticker '%s' not found in SEC data -- skipping", ticker)
        return []

    cik = info["cik"]
    company_name = info["name"]
    logger.info("Processing %s (%s) -- CIK %s", ticker, company_name, cik)

    # Fetch filing metadata
    time.sleep(SEC_REQUEST_DELAY)
    metadata = _run_async(get_filing_metadata(cik))
    filings_list = parse_filing_list(metadata, form_types={form_type})

    if not filings_list:
        logger.warning("  No %s filings found for %s", form_type, ticker)
        return []

    # Limit the number of filings to process
    filings_list = filings_list[:limit]
    logger.info("  Found %d %s filings (processing up to %d)", len(filings_list), form_type, limit)

    new_filing_ids: list[int] = []

    with SyncSessionFactory() as session:
        # Upsert company
        company = session.execute(
            select(Company).where(Company.cik == cik)
        ).scalar_one_or_none()

        if company is None:
            company = Company(cik=cik, ticker=ticker.upper(), name=company_name)
            session.add(company)
            session.flush()
            logger.info("  Created company record: %s (id=%s)", ticker, company.id)

        # Process each filing
        for filing_meta in filings_list:
            accession = filing_meta["accession_number"]

            # Skip if already ingested
            exists = session.execute(
                select(Filing.id).where(Filing.accession_number == accession)
            ).scalar_one_or_none()
            if exists is not None:
                logger.info("  Skipping %s (already ingested)", accession)
                continue

            # Respect SEC rate limit
            time.sleep(SEC_REQUEST_DELAY)

            # Download the filing HTML
            try:
                html_text = _run_async(
                    download_filing(cik, accession, filing_meta["primary_document"])
                )
            except Exception:
                logger.warning("  Failed to download filing %s -- skipping", accession)
                continue

            doc_url = build_document_url(cik, accession, filing_meta["primary_document"])

            # Parse into sections
            sections = parse_filing(html_text, form_type)

            # Create Filing record
            filing = Filing(
                company_id=company.id,
                accession_number=accession,
                form_type=filing_meta["form_type"],
                filing_date=filing_meta["filing_date"],
                report_date=filing_meta.get("report_date"),
                document_url=doc_url,
                raw_text=html_text[:500_000],
                status="ingested",
            )
            session.add(filing)
            session.flush()

            # Create Section records
            for section_type, section_text in sections.items():
                word_count = len(section_text.split())
                section = Section(
                    filing_id=filing.id,
                    section_type=section_type,
                    section_text=section_text,
                    word_count=word_count,
                )
                session.add(section)

            new_filing_ids.append(filing.id)
            logger.info(
                "  Ingested %s (%s, %d sections)",
                accession, filing_meta["filing_date"], len(sections),
            )

        session.commit()

    return new_filing_ids


def analyze_filings(filing_ids: list[int]) -> None:
    """Run drift analysis on a list of filing IDs."""
    from lexdrift.workers.analyze import _do_analyze

    for filing_id in filing_ids:
        try:
            result = _do_analyze(filing_id)
            logger.info(
                "  Analyzed filing %d: %d sections compared",
                filing_id, result.get("sections_analyzed", 0),
            )
        except Exception:
            logger.exception("  Analysis failed for filing %d", filing_id)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill SEC filings for S&P 500 companies into LexDrift.",
    )
    parser.add_argument(
        "--tickers",
        default="sp500",
        help=(
            'Comma-separated list of ticker symbols, or "sp500" for the '
            "built-in list of ~30 major tickers (default: sp500)"
        ),
    )
    parser.add_argument(
        "--form-type",
        default="10-K",
        help="SEC form type to ingest (default: 10-K)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of filings per company (default: 5)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run drift analysis after ingesting each company",
    )
    args = parser.parse_args()

    # Resolve ticker list
    if args.tickers.lower() == "sp500":
        tickers = SP500_TICKERS
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    if not tickers:
        logger.error("No tickers specified")
        sys.exit(1)

    logger.info(
        "Starting backfill: %d tickers, form=%s, limit=%d, analyze=%s",
        len(tickers), args.form_type, args.limit, args.analyze,
    )

    total_ingested = 0
    total_analyzed = 0

    for i, ticker in enumerate(tickers, 1):
        logger.info("=== [%d/%d] %s ===", i, len(tickers), ticker)

        try:
            new_ids = ingest_company(ticker, args.form_type, args.limit)
            total_ingested += len(new_ids)

            if args.analyze and new_ids:
                logger.info("  Running analysis on %d new filings...", len(new_ids))
                analyze_filings(new_ids)
                total_analyzed += len(new_ids)

        except Exception:
            logger.exception("Failed to process %s -- continuing with next ticker", ticker)

        # Polite pause between companies to respect SEC rate limits
        if i < len(tickers):
            time.sleep(1.0)

    logger.info(
        "Backfill complete: %d filings ingested, %d analyzed across %d tickers",
        total_ingested, total_analyzed, len(tickers),
    )


if __name__ == "__main__":
    main()
