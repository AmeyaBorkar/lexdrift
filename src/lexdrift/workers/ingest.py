"""Celery task: ingest SEC filings for a given ticker.

Fetches filing metadata from EDGAR, downloads the HTML documents, parses them
into sections, and stores Company / Filing / Section records in the database.

Uses SYNCHRONOUS SQLAlchemy because Celery workers run in regular threads.
The async EDGAR client is driven through ``asyncio.run()`` calls.
"""

import asyncio
import logging

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
from lexdrift.workers.celery_app import app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchronous DB setup for Celery workers
# ---------------------------------------------------------------------------

def _make_sync_url(async_url: str) -> str:
    """Convert an async SQLAlchemy URL to its synchronous equivalent."""
    url = async_url.replace("+aiosqlite", "")
    url = url.replace("+asyncpg", "+psycopg2")
    return url


_sync_engine = create_engine(_make_sync_url(settings.database_url), echo=False, future=True)
SyncSessionFactory = sessionmaker(bind=_sync_engine, class_=Session, expire_on_commit=False)

# Ensure tables exist (mirrors the FastAPI startup hook)
Base.metadata.create_all(_sync_engine)


def _run_async(coro):
    """Run an async coroutine from synchronous Celery code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Shouldn't happen in a Celery worker, but just in case
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@app.task(name="lexdrift.workers.ingest.ingest_filings", bind=True, max_retries=3)
def ingest_filings(self, ticker: str, form_type: str = "10-K"):
    """Ingest all filings of *form_type* for *ticker* from SEC EDGAR.

    Steps
    -----
    1. Resolve ticker -> CIK via the SEC company_tickers mapping.
    2. Fetch the company's filing metadata JSON from EDGAR.
    3. Filter to the requested form type.
    4. For each filing not already in the DB:
       a. Download the HTML document.
       b. Parse it into sections.
       c. Store Company, Filing, and Section records.

    Returns a summary dict with counts of filings processed.
    """
    try:
        return _do_ingest(ticker, form_type)
    except Exception as exc:
        logger.exception("Ingest failed for %s / %s", ticker, form_type)
        raise self.retry(exc=exc, countdown=60 * (self.request.retries + 1))


def _do_ingest(ticker: str, form_type: str) -> dict:
    # 1. Resolve ticker -> CIK
    info = _run_async(lookup_ticker(ticker))
    if not info:
        raise ValueError(f"Ticker '{ticker}' not found in SEC data")

    cik = info["cik"]
    company_name = info["name"]
    logger.info("Ingesting %s (%s) – CIK %s, form %s", ticker, company_name, cik, form_type)

    # 2. Fetch filing metadata
    metadata = _run_async(get_filing_metadata(cik))
    filings_list = parse_filing_list(metadata, form_types={form_type})

    if not filings_list:
        logger.warning("No %s filings found for %s", form_type, ticker)
        return {"ticker": ticker, "form_type": form_type, "ingested": 0, "skipped": 0}

    ingested = 0
    skipped = 0

    with SyncSessionFactory() as session:
        # 3. Upsert company
        company = session.execute(
            select(Company).where(Company.cik == cik)
        ).scalar_one_or_none()

        if company is None:
            company = Company(cik=cik, ticker=ticker.upper(), name=company_name)
            session.add(company)
            session.flush()
            logger.info("Created company record: %s (id=%s)", ticker, company.id)

        # 4. Process each filing
        for filing_meta in filings_list:
            accession = filing_meta["accession_number"]

            # Skip if already ingested
            exists = session.execute(
                select(Filing.id).where(Filing.accession_number == accession)
            ).scalar_one_or_none()
            if exists is not None:
                skipped += 1
                continue

            # Download the filing HTML
            try:
                html_text = _run_async(
                    download_filing(cik, accession, filing_meta["primary_document"])
                )
            except Exception:
                logger.warning("Failed to download filing %s, skipping", accession)
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
                raw_text=html_text[:500_000],  # cap storage
                status="ingested",
            )
            session.add(filing)
            session.flush()  # get filing.id

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

            ingested += 1
            logger.info(
                "Ingested filing %s (%s sections)", accession, len(sections)
            )

        session.commit()

    summary = {
        "ticker": ticker,
        "form_type": form_type,
        "ingested": ingested,
        "skipped": skipped,
        "total_found": len(filings_list),
    }
    logger.info("Ingest complete: %s", summary)
    return summary
