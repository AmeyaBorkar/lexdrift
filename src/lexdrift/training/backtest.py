"""Backtesting Framework -- validate drift scores against actual stock returns.

Tests the hypothesis: "Do our drift scores actually predict financial outcomes?"

For each company, orders drift scores by filing date, then checks whether
high-drift filings are followed by negative stock returns (>5% loss within 30
days of filing).

Usage:
    python -m lexdrift.training.backtest
    python -m lexdrift.training.backtest --ticker AAPL
    python -m lexdrift.training.backtest --calibrate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure src/ is importable when running as a script
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root / "src"))

from lexdrift.data.price_feed import (
    calibrate_risk_weights,
    compute_filing_outcomes,
    get_price_around_filing,
)
from lexdrift.db.models import Alert, Company, DriftScore, Filing, KeyPhrase

logger = logging.getLogger(__name__)


@dataclass
class SectionBacktestResult:
    """Per-section backtest statistics."""
    section_type: str
    total_scores: int = 0
    high_drift_count: int = 0
    high_drift_bad_outcome_count: int = 0
    low_drift_count: int = 0
    low_drift_clean_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    avg_drift_before_bad: float = 0.0
    avg_drift_before_good: float = 0.0


@dataclass
class BacktestResult:
    """Aggregate backtest results across all companies."""
    total_filings: int = 0
    high_drift_count: int = 0
    high_drift_bad_outcome_count: int = 0
    low_drift_count: int = 0
    low_drift_clean_count: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    avg_drift_before_bad_outcome: float = 0.0
    avg_drift_before_good_outcome: float = 0.0
    total_bad_outcomes: int = 0
    total_good_outcomes: int = 0
    by_section: dict[str, SectionBacktestResult] = field(default_factory=dict)


def _get_composite_drift(ds: DriftScore) -> float:
    """Compute a single drift score from cosine + jaccard distances."""
    cosine = ds.cosine_distance or 0.0
    jaccard = ds.jaccard_distance or 0.0
    # Weighted average — cosine is the primary metric
    return 0.7 * cosine + 0.3 * jaccard


def _has_bad_outcome(
    db_session: Session,
    company_id: int,
    current_filing: Filing,
    next_filing: Filing,
    current_sentiment_neg: float,
) -> bool:
    """Determine if a 'bad outcome' occurred after the current filing.

    A bad outcome is defined as a negative stock return >5% within 30 days
    of the filing date. Falls back to sentiment-based proxy if price data
    is unavailable.
    """
    # Look up the company ticker
    company = db_session.get(Company, company_id)
    ticker = company.ticker if company else None

    if ticker and current_filing.filing_date:
        fdate_str = (
            current_filing.filing_date.isoformat()
            if hasattr(current_filing.filing_date, "isoformat")
            else str(current_filing.filing_date)
        )
        price_data = get_price_around_filing(ticker, fdate_str, window_days=30)
        if price_data and price_data.get("return_pct") is not None:
            # Bad outcome = negative return exceeding 5%
            return price_data["return_pct"] < -0.05

    # Fallback: sentiment-based proxy when price data is unavailable
    next_drifts = db_session.execute(
        select(DriftScore).where(
            DriftScore.filing_id == next_filing.id,
            DriftScore.company_id == company_id,
        )
    ).scalars().all()

    for nd in next_drifts:
        if nd.sentiment_delta and isinstance(nd.sentiment_delta, dict):
            next_neg = nd.sentiment_delta.get("negative", 0.0)
            if isinstance(next_neg, (int, float)) and next_neg > 0.05:
                return True

    return False


def backtest_drift_vs_outcomes(
    db_session: Session,
    ticker: str | None = None,
) -> BacktestResult:
    """Run backtesting: does high drift predict bad outcomes?

    For each company, orders drift scores by filing date, then checks
    whether high-drift filings are followed by bad outcomes within the
    next filing period.

    Args:
        db_session: Synchronous SQLAlchemy session.
        ticker: If provided, backtest only this company. Otherwise all.

    Returns:
        BacktestResult with precision, recall, F1, and per-section stats.
    """
    result = BacktestResult()

    # Get companies to backtest
    stmt = select(Company)
    if ticker:
        stmt = stmt.where(Company.ticker == ticker.upper())
    companies = db_session.execute(stmt).scalars().all()

    if not companies:
        logger.warning("No companies found for backtest%s",
                        f" (ticker={ticker})" if ticker else "")
        return result

    # Collect all drift values to compute the 75th percentile threshold
    all_drift_values: list[float] = []
    # section_type -> list of drift values
    section_drift_values: dict[str, list[float]] = {}

    # First pass: gather all drift scores
    for company in companies:
        drift_scores = db_session.execute(
            select(DriftScore)
            .join(Filing, DriftScore.filing_id == Filing.id)
            .where(DriftScore.company_id == company.id)
            .order_by(Filing.filing_date)
        ).scalars().all()

        for ds in drift_scores:
            val = _get_composite_drift(ds)
            all_drift_values.append(val)
            section_drift_values.setdefault(ds.section_type, []).append(val)

    if not all_drift_values:
        logger.warning("No drift scores found for backtesting")
        return result

    # Compute thresholds
    global_threshold = float(np.percentile(all_drift_values, 75))
    section_thresholds: dict[str, float] = {}
    for sec, vals in section_drift_values.items():
        section_thresholds[sec] = float(np.percentile(vals, 75))

    logger.info("Global 75th percentile drift threshold: %.4f", global_threshold)

    # Tracking lists for overall stats
    drift_before_bad: list[float] = []
    drift_before_good: list[float] = []
    # Per-section tracking
    section_stats: dict[str, dict] = {}

    # Second pass: evaluate outcomes
    for company in companies:
        # Get all filings ordered by date
        filings = db_session.execute(
            select(Filing)
            .where(Filing.company_id == company.id)
            .order_by(Filing.filing_date)
        ).scalars().all()

        if len(filings) < 2:
            continue

        # Build filing_id -> next_filing mapping
        filing_map: dict[int, Filing] = {f.id: f for f in filings}
        next_filing_map: dict[int, Filing] = {}
        for i in range(len(filings) - 1):
            next_filing_map[filings[i].id] = filings[i + 1]

        # Get drift scores for this company
        drift_scores = db_session.execute(
            select(DriftScore)
            .join(Filing, DriftScore.filing_id == Filing.id)
            .where(DriftScore.company_id == company.id)
            .order_by(Filing.filing_date)
        ).scalars().all()

        result.total_filings += len(filings)

        for ds in drift_scores:
            current_filing = filing_map.get(ds.filing_id)
            next_filing = next_filing_map.get(ds.filing_id)
            if not current_filing or not next_filing:
                continue

            drift_val = _get_composite_drift(ds)
            is_high_drift = drift_val >= global_threshold

            # Extract current negative sentiment
            current_neg = 0.0
            if ds.sentiment_delta and isinstance(ds.sentiment_delta, dict):
                current_neg = ds.sentiment_delta.get("negative", 0.0)
                if not isinstance(current_neg, (int, float)):
                    current_neg = 0.0

            bad_outcome = _has_bad_outcome(
                db_session, company.id, current_filing, next_filing, current_neg
            )

            # Track overall stats
            if bad_outcome:
                result.total_bad_outcomes += 1
                drift_before_bad.append(drift_val)
            else:
                result.total_good_outcomes += 1
                drift_before_good.append(drift_val)

            if is_high_drift:
                result.high_drift_count += 1
                if bad_outcome:
                    result.high_drift_bad_outcome_count += 1
            else:
                result.low_drift_count += 1
                if not bad_outcome:
                    result.low_drift_clean_count += 1

            # Per-section tracking
            sec = ds.section_type
            if sec not in section_stats:
                section_stats[sec] = {
                    "total": 0, "high": 0, "high_bad": 0,
                    "low": 0, "low_clean": 0,
                    "drift_before_bad": [], "drift_before_good": [],
                }
            ss = section_stats[sec]
            ss["total"] += 1
            if bad_outcome:
                ss["drift_before_bad"].append(drift_val)
            else:
                ss["drift_before_good"].append(drift_val)
            sec_threshold = section_thresholds.get(sec, global_threshold)
            is_sec_high = drift_val >= sec_threshold
            if is_sec_high:
                ss["high"] += 1
                if bad_outcome:
                    ss["high_bad"] += 1
            else:
                ss["low"] += 1
                if not bad_outcome:
                    ss["low_clean"] += 1

    # Compute precision / recall / F1 (overall)
    # Precision: when we flagged high drift, how often was there a bad outcome?
    if result.high_drift_count > 0:
        result.precision = result.high_drift_bad_outcome_count / result.high_drift_count
    # Recall: of all bad outcomes, how many did we flag as high drift?
    if result.total_bad_outcomes > 0:
        result.recall = result.high_drift_bad_outcome_count / result.total_bad_outcomes
    # F1
    if result.precision + result.recall > 0:
        result.f1 = 2 * (result.precision * result.recall) / (result.precision + result.recall)

    # Average drift before bad/good outcomes
    if drift_before_bad:
        result.avg_drift_before_bad_outcome = float(np.mean(drift_before_bad))
    if drift_before_good:
        result.avg_drift_before_good_outcome = float(np.mean(drift_before_good))

    # Per-section results
    for sec, ss in section_stats.items():
        sr = SectionBacktestResult(section_type=sec)
        sr.total_scores = ss["total"]
        sr.high_drift_count = ss["high"]
        sr.high_drift_bad_outcome_count = ss["high_bad"]
        sr.low_drift_count = ss["low"]
        sr.low_drift_clean_count = ss["low_clean"]

        if sr.high_drift_count > 0:
            sr.precision = sr.high_drift_bad_outcome_count / sr.high_drift_count
        bad_in_sec = len(ss["drift_before_bad"])
        if bad_in_sec > 0:
            sr.recall = sr.high_drift_bad_outcome_count / bad_in_sec
        if sr.precision + sr.recall > 0:
            sr.f1 = 2 * (sr.precision * sr.recall) / (sr.precision + sr.recall)
        if ss["drift_before_bad"]:
            sr.avg_drift_before_bad = float(np.mean(ss["drift_before_bad"]))
        if ss["drift_before_good"]:
            sr.avg_drift_before_good = float(np.mean(ss["drift_before_good"]))

        result.by_section[sec] = sr

    return result


def generate_backtest_report(result: BacktestResult) -> str:
    """Generate a human-readable backtest report."""
    lines = [
        "=" * 70,
        "LEXDRIFT BACKTEST REPORT",
        "=" * 70,
        "",
        "Overall Statistics:",
        f"  Total filings evaluated:        {result.total_filings}",
        f"  Total bad outcomes detected:     {result.total_bad_outcomes}",
        f"  Total good outcomes detected:    {result.total_good_outcomes}",
        "",
        "High-Drift Predictions:",
        f"  High-drift filings:              {result.high_drift_count}",
        f"  High-drift with bad outcome:     {result.high_drift_bad_outcome_count}",
        f"  Precision:                       {result.precision:.4f}",
        "",
        "Recall (Coverage):",
        f"  Bad outcomes caught by high-drift: {result.high_drift_bad_outcome_count} / {result.total_bad_outcomes}",
        f"  Recall:                          {result.recall:.4f}",
        "",
        f"  F1 Score:                        {result.f1:.4f}",
        "",
        "Drift Magnitude:",
        f"  Avg drift before bad outcome:    {result.avg_drift_before_bad_outcome:.4f}",
        f"  Avg drift before good outcome:   {result.avg_drift_before_good_outcome:.4f}",
    ]

    if result.avg_drift_before_good_outcome > 0:
        ratio = result.avg_drift_before_bad_outcome / result.avg_drift_before_good_outcome
        lines.append(f"  Ratio (bad/good):                {ratio:.2f}x")

    lines.append("")
    lines.append("Low-Drift Accuracy:")
    lines.append(f"  Low-drift filings:               {result.low_drift_count}")
    lines.append(f"  Low-drift with clean outcome:    {result.low_drift_clean_count}")
    if result.low_drift_count > 0:
        clean_rate = result.low_drift_clean_count / result.low_drift_count
        lines.append(f"  Clean rate when drift is low:    {clean_rate:.4f}")

    if result.by_section:
        lines.append("")
        lines.append("-" * 70)
        lines.append("PER-SECTION BREAKDOWN:")
        lines.append("-" * 70)

        for sec_type, sr in sorted(result.by_section.items()):
            lines.append("")
            lines.append(f"  {sec_type}:")
            lines.append(f"    Scores:     {sr.total_scores}")
            lines.append(f"    Precision:  {sr.precision:.4f}  "
                          f"Recall: {sr.recall:.4f}  F1: {sr.f1:.4f}")
            lines.append(f"    Avg drift before bad:  {sr.avg_drift_before_bad:.4f}  "
                          f"good: {sr.avg_drift_before_good:.4f}")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)


def _get_sync_session():
    """Create a synchronous session for CLI usage."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session as SyncSession
    from sqlalchemy.orm import sessionmaker

    from lexdrift.config import settings
    from lexdrift.db.models import Base

    db_url = settings.database_url
    sync_url = db_url.replace("+aiosqlite", "").replace("+asyncpg", "+psycopg2")

    engine = create_engine(sync_url, echo=False, future=True)
    Base.metadata.create_all(engine)

    SessionFactory = sessionmaker(bind=engine, class_=SyncSession, expire_on_commit=False)
    return SessionFactory()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="LexDrift Backtesting Framework")
    parser.add_argument("--ticker", default=None, help="Backtest a single ticker (default: all)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run weight calibration and save to data/calibrated_weights.json")
    args = parser.parse_args()

    session = _get_sync_session()
    try:
        result = backtest_drift_vs_outcomes(session, ticker=args.ticker)
        report = generate_backtest_report(result)
        print(report)

        if args.calibrate:
            print("\nRunning weight calibration...")
            outcomes = compute_filing_outcomes(session)
            if outcomes:
                weights = calibrate_risk_weights(outcomes)
                out_path = Path("data/calibrated_weights.json")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(json.dumps(weights, indent=2))
                print(f"Calibrated weights saved to {out_path}:")
                for k, v in weights.items():
                    print(f"  {k}: {v:.6f}")
            else:
                print("No filing outcomes available for calibration.")
    finally:
        session.close()
