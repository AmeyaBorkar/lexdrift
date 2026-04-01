"""Celery beat periodic task: poll EDGAR for new filings.

Runs every 30 minutes (configured in celery_app.py beat_schedule).  For every
company on a watchlist it checks whether EDGAR has any filings that are not yet
in the local database and, if so, queues ingest + analyze tasks.

Uses SYNCHRONOUS SQLAlchemy because Celery workers run in regular threads.
The async EDGAR client is driven through ``asyncio.run()`` calls.
"""

import asyncio
import logging

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from lexdrift.config import settings
from lexdrift.db.models import (
    Base,
    Company,
    Filing,
    WatchlistCompany,
)
from lexdrift.edgar.filings import get_filing_metadata, parse_filing_list
from lexdrift.workers.celery_app import app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchronous DB setup (same pattern as other workers)
# ---------------------------------------------------------------------------

def _make_sync_url(async_url: str) -> str:
    url = async_url.replace("+aiosqlite", "")
    url = url.replace("+asyncpg", "+psycopg2")
    return url


_sync_engine = create_engine(_make_sync_url(settings.database_url), echo=False, future=True)
SyncSessionFactory = sessionmaker(bind=_sync_engine, class_=Session, expire_on_commit=False)

Base.metadata.create_all(_sync_engine)

# Form types to monitor
MONITORED_FORMS = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"}


def _run_async(coro):
    """Run an async coroutine from synchronous Celery code."""
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
# Task
# ---------------------------------------------------------------------------

@app.task(name="lexdrift.workers.monitor.poll_edgar")
def poll_edgar():
    """Check EDGAR for new filings for all watched companies.

    For each company found on any watchlist:
    1. Fetch the latest filing metadata from EDGAR.
    2. Compare against filings already stored in the DB.
    3. For each new filing, queue:
       a. ``ingest_filings`` to download and parse
       b. ``analyze_filing`` (chained after ingest) to compute drift

    Returns a summary of how many new filings were discovered.
    """
    from lexdrift.workers.ingest import ingest_filings
    from lexdrift.workers.analyze import analyze_filing

    total_new = 0
    companies_checked = 0

    with SyncSessionFactory() as session:
        # Gather all distinct companies across all watchlists
        stmt = (
            select(Company)
            .join(WatchlistCompany, WatchlistCompany.company_id == Company.id)
            .distinct()
        )
        watched_companies = session.execute(stmt).scalars().all()

        if not watched_companies:
            logger.info("No companies on any watchlist; nothing to poll")
            return {"companies_checked": 0, "new_filings_queued": 0}

        # Collect accession numbers already in DB per company for fast lookup
        existing_accessions: dict[int, set[str]] = {}
        for company in watched_companies:
            rows = session.execute(
                select(Filing.accession_number).where(Filing.company_id == company.id)
            ).scalars().all()
            existing_accessions[company.id] = set(rows)

    # Now hit EDGAR for each company (outside the DB session to avoid long locks)
    for company in watched_companies:
        companies_checked += 1
        try:
            metadata = _run_async(get_filing_metadata(company.cik))
            filings_list = parse_filing_list(metadata, form_types=MONITORED_FORMS)
        except Exception:
            logger.warning(
                "Failed to fetch EDGAR data for %s (CIK %s), skipping",
                company.ticker, company.cik,
            )
            continue

        known = existing_accessions.get(company.id, set())
        new_filings = [
            f for f in filings_list if f["accession_number"] not in known
        ]

        if not new_filings:
            continue

        logger.info(
            "Found %d new filings for %s (%s)",
            len(new_filings), company.ticker, company.cik,
        )
        total_new += len(new_filings)

        # Queue ingestion for each unique form type that has new filings, then
        # chain analysis.  Ingesting by ticker + form_type is idempotent (the
        # ingest worker skips accession numbers already in the DB).
        new_form_types = {f["form_type"] for f in new_filings}
        for form_type in new_form_types:
            # Chain: ingest first, then analyze each newly-created filing
            ingest_filings.apply_async(
                args=[company.ticker, form_type],
                link=analyze_filing.s(),  # will receive ingest result dict
            )
            logger.info(
                "Queued ingest+analyze for %s / %s", company.ticker, form_type
            )

    summary = {
        "companies_checked": companies_checked,
        "new_filings_queued": total_new,
    }
    logger.info("EDGAR poll complete: %s", summary)
    return summary
