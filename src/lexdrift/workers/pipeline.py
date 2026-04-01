"""Automated production pipeline -- runs daily to keep LexDrift data fresh.

Schedule (via Celery beat or cron):
  1. Check EDGAR RSS for new filings from all tracked companies
  2. Ingest any new filings found
  3. Analyze new filings (full NLP pipeline)
  4. Update TF-IDF corpus with new filing text
  5. Generate/update intelligence reports for affected companies
  6. Check if model retraining is needed (every 30 days or 100+ new filings)
  7. Send alert digest (summary of all new alerts since last run)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from lexdrift.config import settings
from lexdrift.db.models import (
    Alert,
    Base,
    Company,
    Filing,
    WatchlistCompany,
)
from lexdrift.workers.celery_app import app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Synchronous DB setup (same pattern as other workers)
# ---------------------------------------------------------------------------

_DATA_DIR = Path("data")
_TRAINING_STATE_FILE = _DATA_DIR / "training_state.json"


def _make_sync_url(async_url: str) -> str:
    url = async_url.replace("+aiosqlite", "")
    url = url.replace("+asyncpg", "+psycopg2")
    return url


_sync_engine = create_engine(
    _make_sync_url(settings.database_url), echo=False, future=True,
)
SyncSessionFactory = sessionmaker(
    bind=_sync_engine, class_=Session, expire_on_commit=False,
)
Base.metadata.create_all(_sync_engine)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_training_state() -> dict:
    """Load training state from JSON file."""
    if _TRAINING_STATE_FILE.exists():
        try:
            return json.loads(_TRAINING_STATE_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt training state file, resetting")
    return {"last_training_date": None, "filings_since_training": 0}


def _save_training_state(state: dict) -> None:
    """Persist training state to JSON file."""
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _TRAINING_STATE_FILE.write_text(json.dumps(state, indent=2))


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def check_for_new_filings(db_session: Session) -> list[dict]:
    """For each company in any watchlist, check EDGAR for filings not yet in DB.

    Returns list of {ticker, accession, form_type}.
    """
    import asyncio

    from lexdrift.edgar.filings import get_filing_metadata, parse_filing_list

    # Gather watched companies
    stmt = (
        select(Company)
        .join(WatchlistCompany, WatchlistCompany.company_id == Company.id)
        .distinct()
    )
    watched = db_session.execute(stmt).scalars().all()
    if not watched:
        logger.info("No watched companies found")
        return []

    # Collect existing accessions per company
    existing: dict[int, set[str]] = {}
    for company in watched:
        rows = db_session.execute(
            select(Filing.accession_number).where(Filing.company_id == company.id)
        ).scalars().all()
        existing[company.id] = set(rows)

    monitored_forms = {"10-K", "10-Q", "8-K", "10-K/A", "10-Q/A"}
    new_filings: list[dict] = []

    for company in watched:
        try:
            metadata = asyncio.run(get_filing_metadata(company.cik))
            filings_list = parse_filing_list(metadata, form_types=monitored_forms)
        except Exception:
            logger.warning(
                "Failed to fetch EDGAR data for %s (CIK %s), skipping",
                company.ticker, company.cik,
            )
            continue

        known = existing.get(company.id, set())
        for f in filings_list:
            if f["accession_number"] not in known:
                new_filings.append({
                    "ticker": company.ticker,
                    "accession": f["accession_number"],
                    "form_type": f["form_type"],
                })

    logger.info("Found %d new filings across %d companies", len(new_filings), len(watched))
    return new_filings


def should_retrain(db_session: Session) -> bool:
    """Check if model retraining is needed.

    Triggers:
      - 30+ days since last training
      - 100+ new filings since last training
    """
    state = _load_training_state()
    last_date_str = state.get("last_training_date")
    filings_since = state.get("filings_since_training", 0)

    if filings_since >= 100:
        logger.info("Retrain triggered: %d filings since last training", filings_since)
        return True

    if last_date_str:
        try:
            last_date = datetime.fromisoformat(last_date_str)
            if datetime.utcnow() - last_date > timedelta(days=30):
                logger.info("Retrain triggered: >30 days since last training")
                return True
        except ValueError:
            logger.warning("Invalid last_training_date in state: %s", last_date_str)
            return True
    else:
        # Never trained before
        total = db_session.execute(
            select(func.count(Filing.id)).where(Filing.status == "analyzed")
        ).scalar() or 0
        if total >= 10:
            logger.info("Retrain triggered: first training with %d analyzed filings", total)
            return True

    return False


def generate_alert_digest(db_session: Session) -> str:
    """Generate summary of all unread alerts, grouped by severity.

    Returns a formatted string digest.
    """
    stmt = (
        select(Alert)
        .where(Alert.read == False)  # noqa: E712
        .order_by(Alert.created_at.desc())
    )
    unread = db_session.execute(stmt).scalars().all()

    if not unread:
        return "No unread alerts."

    by_severity: dict[str, list[Alert]] = defaultdict(list)
    for alert in unread:
        by_severity[alert.severity].append(alert)

    lines: list[str] = [
        f"ALERT DIGEST -- {len(unread)} unread alert(s)",
        "=" * 60,
        "",
    ]

    severity_order = ["critical", "high", "medium", "low"]
    for severity in severity_order:
        alerts_in_group = by_severity.get(severity, [])
        if not alerts_in_group:
            continue

        lines.append(f"[{severity.upper()}] ({len(alerts_in_group)} alert(s))")
        lines.append("-" * 40)

        # Group by company
        by_company: dict[int, list[Alert]] = defaultdict(list)
        for a in alerts_in_group:
            by_company[a.company_id].append(a)

        for company_id, company_alerts in by_company.items():
            # Look up company name
            company = db_session.get(Company, company_id)
            name = company.ticker if company else f"Company #{company_id}"
            lines.append(f"  {name}: {len(company_alerts)} alert(s)")
            for a in company_alerts[:5]:
                msg = a.message[:120] if a.message else a.alert_type
                lines.append(f"    - [{a.alert_type}] {msg}")
            if len(company_alerts) > 5:
                lines.append(f"    ... and {len(company_alerts) - 5} more")

        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline task
# ---------------------------------------------------------------------------

@app.task(name="lexdrift.workers.pipeline.run_daily_pipeline")
def run_daily_pipeline():
    """Celery task: full daily production pipeline.

    1. Check EDGAR for new filings from all tracked companies
    2. Ingest any new filings found
    3. Analyze new filings (full NLP pipeline)
    4. Update TF-IDF corpus with new filing text
    5. Generate/update intelligence reports for affected companies
    6. Check if model retraining is needed
    7. Send alert digest
    """
    from lexdrift.workers.analyze import analyze_filing
    from lexdrift.workers.ingest import ingest_filings
    from lexdrift.nlp.intelligence import generate_intelligence

    logger.info("Starting daily pipeline run")

    with SyncSessionFactory() as session:
        # Step 1: Check for new filings
        new_filings = check_for_new_filings(session)
        logger.info("Step 1 complete: %d new filings found", len(new_filings))

        if not new_filings:
            logger.info("No new filings found, running digest only")
            digest = generate_alert_digest(session)
            logger.info("Alert digest:\n%s", digest)
            return {
                "new_filings": 0,
                "ingested": 0,
                "analyzed": 0,
                "retrain_needed": False,
                "digest": digest,
            }

        # Step 2: Ingest new filings
        ingested_ids: list[int] = []
        form_types_by_ticker: dict[str, set[str]] = defaultdict(set)
        for f in new_filings:
            form_types_by_ticker[f["ticker"]].add(f["form_type"])

        for ticker, form_types in form_types_by_ticker.items():
            for form_type in form_types:
                try:
                    result = ingest_filings(ticker, form_type)
                    logger.info("Ingested %s/%s: %s", ticker, form_type, result)
                except Exception:
                    logger.error(
                        "Ingest failed for %s/%s", ticker, form_type, exc_info=True,
                    )

        logger.info("Step 2 complete: ingestion finished")

        # Step 3: Analyze new filings (those in "ingested" status)
        new_ingested = session.execute(
            select(Filing).where(Filing.status == "ingested")
        ).scalars().all()

        analyzed_count = 0
        affected_company_ids: set[int] = set()
        for filing in new_ingested:
            try:
                analyze_filing(filing.id)
                analyzed_count += 1
                affected_company_ids.add(filing.company_id)
                ingested_ids.append(filing.id)
            except Exception:
                logger.error(
                    "Analysis failed for filing %d", filing.id, exc_info=True,
                )

        logger.info("Step 3 complete: %d filings analyzed", analyzed_count)

        # Step 4: Update TF-IDF corpus (refresh phrase models)
        try:
            from lexdrift.nlp.phrases import rebuild_tfidf_corpus
            rebuild_tfidf_corpus(session)
            logger.info("Step 4 complete: TF-IDF corpus updated")
        except (ImportError, AttributeError):
            logger.info("Step 4 skipped: TF-IDF corpus update not available")
        except Exception:
            logger.warning("Step 4 failed: TF-IDF corpus update error", exc_info=True)

        # Step 5: Generate intelligence reports for affected companies
        reports_generated = 0
        for company_id in affected_company_ids:
            company = session.get(Company, company_id)
            if company and company.ticker:
                try:
                    generate_intelligence(session, company.ticker)
                    reports_generated += 1
                except Exception:
                    logger.warning(
                        "Intelligence report failed for %s", company.ticker, exc_info=True,
                    )

        logger.info("Step 5 complete: %d intelligence reports generated", reports_generated)

        # Step 6: Check if model retraining is needed
        retrain_needed = should_retrain(session)
        if retrain_needed:
            logger.info("Step 6: Model retraining triggered")
            # Update training state
            state = _load_training_state()
            state["last_training_date"] = datetime.utcnow().isoformat()
            state["filings_since_training"] = 0
            _save_training_state(state)
        else:
            # Increment filing counter
            state = _load_training_state()
            state["filings_since_training"] = state.get("filings_since_training", 0) + len(ingested_ids)
            _save_training_state(state)
            logger.info("Step 6: No retraining needed")

        # Step 7: Alert digest
        digest = generate_alert_digest(session)
        logger.info("Step 7 complete. Alert digest:\n%s", digest)

    summary = {
        "new_filings": len(new_filings),
        "ingested": len(ingested_ids),
        "analyzed": analyzed_count,
        "reports_generated": reports_generated,
        "retrain_needed": retrain_needed,
        "digest": digest,
    }
    logger.info("Daily pipeline complete: %s", summary)
    return summary
