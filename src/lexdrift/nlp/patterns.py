"""Historical Pattern Matching -- "Have we seen this before? What happened?"

Defines canonical disclosure patterns drawn from academic research and
practitioner experience, then checks whether a company's current signal
profile matches any of them.  For each match, historical examples are
retrieved from the database so that analysts can reason by analogy.

Patterns
--------
1. **Lazy Prices signal** -- drift increasing for 3+ consecutive quarters
   (Cohen, Malloy & Nguyen 2020).
2. **Going concern cascade** -- negative sentiment rising + going concern
   language appearing.
3. **Obfuscation wave** -- obfuscation score rising while sentiment stays flat.
4. **Risk factor explosion** -- new-phrase count exceeds 2x historical baseline.
5. **Sentiment reversal** -- positive sentiment drops >50% in one quarter.
6. **Silent deletion** -- key phrases disappearing without explanation.
7. **Peer divergence** -- company drift diverging from sector average.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from lexdrift.db.models import (
    Alert,
    Company,
    DriftScore,
    Filing,
    KeyPhrase,
    SentenceChange,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PatternMatch:
    """A matched historical pattern with supporting evidence."""
    pattern_name: str  # machine-readable id, e.g. "lazy_prices"
    description: str
    companies_matched: list[str]  # tickers of historical matches
    outcomes: list[str]  # what happened to those companies
    match_score: float  # 0.0 - 1.0 how closely this company matches
    evidence: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _drift_series(db: Session, company_id: int) -> list[dict]:
    """Return chronological drift scores for a company (risk_factors preferred)."""
    stmt = (
        select(
            DriftScore.cosine_distance,
            DriftScore.sentiment_delta,
            DriftScore.section_type,
            Filing.filing_date,
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .where(DriftScore.company_id == company_id)
        .order_by(Filing.filing_date)
    )
    rows = db.execute(stmt).all()
    # Prefer risk_factors; fall back to any section
    rf = [r for r in rows if r.section_type == "risk_factors"]
    return [
        {
            "cosine_distance": r.cosine_distance,
            "sentiment_delta": r.sentiment_delta,
            "filing_date": r.filing_date,
        }
        for r in (rf if rf else rows)
        if r.cosine_distance is not None
    ]


def _phrase_history(db: Session, company_id: int) -> list[dict]:
    """Return key-phrase records ordered by filing date."""
    stmt = (
        select(
            KeyPhrase.phrase,
            KeyPhrase.status,
            KeyPhrase.section_type,
            Filing.filing_date,
        )
        .join(Filing, KeyPhrase.filing_id == Filing.id)
        .join(Company, Filing.company_id == Company.id)
        .where(Company.id == company_id)
        .order_by(Filing.filing_date)
    )
    return [
        {"phrase": r.phrase, "status": r.status, "section_type": r.section_type, "filing_date": r.filing_date}
        for r in db.execute(stmt).all()
    ]


def _find_historical_matches(
    db: Session,
    exclude_company_id: int,
    *,
    min_consecutive_increase: int = 3,
) -> list[dict]:
    """Find companies whose drift increased for N+ consecutive periods.

    Returns list of {ticker, name, outcome} dicts.
    """
    companies = db.execute(select(Company)).scalars().all()
    matches: list[dict] = []
    for co in companies:
        if co.id == exclude_company_id:
            continue
        series = _drift_series(db, co.id)
        if len(series) < min_consecutive_increase:
            continue
        drifts = [s["cosine_distance"] for s in series]
        # Check tail for consecutive increases
        tail = drifts[-(min_consecutive_increase):]
        if all(tail[i] < tail[i + 1] for i in range(len(tail) - 1)):
            # Determine outcome: did alerts increase?
            alert_count = db.execute(
                select(func.count(Alert.id)).where(Alert.company_id == co.id)
            ).scalar() or 0
            outcome = f"{alert_count} alert(s) generated" if alert_count else "No alerts generated"
            matches.append({
                "ticker": co.ticker or co.cik,
                "name": co.name or "Unknown",
                "outcome": outcome,
            })
    return matches[:10]  # cap for performance


# ---------------------------------------------------------------------------
# Pattern detection functions
# ---------------------------------------------------------------------------

def _check_lazy_prices(db: Session, company_id: int, series: list[dict]) -> PatternMatch | None:
    """Lazy Prices: drift increasing for 3+ consecutive quarters."""
    if len(series) < 3:
        return None
    drifts = [s["cosine_distance"] for s in series]
    # Find the longest consecutive increasing streak at the tail
    streak = 1
    for i in range(len(drifts) - 1, 0, -1):
        if drifts[i] > drifts[i - 1]:
            streak += 1
        else:
            break
    if streak < 3:
        return None

    historical = _find_historical_matches(db, company_id, min_consecutive_increase=3)
    score = min(streak / 6.0, 1.0)  # 6-period streak = perfect match

    return PatternMatch(
        pattern_name="lazy_prices",
        description=(
            f"Drift score has increased for {streak} consecutive periods. "
            "Cohen et al. (2020) showed that 10-K language changes predict "
            "negative future returns."
        ),
        companies_matched=[m["ticker"] for m in historical],
        outcomes=[m["outcome"] for m in historical],
        match_score=round(score, 3),
        evidence=[
            f"Period {i+1}: drift={drifts[-(streak-i)]:,.4f}"
            for i in range(streak)
        ],
    )


def _check_going_concern_cascade(
    db: Session, company_id: int, series: list[dict], phrases: list[dict],
) -> PatternMatch | None:
    """Going concern cascade: negative sentiment rising + going concern language."""
    concern_phrases = {"going concern", "substantial doubt", "ability to continue"}
    found_concern = [
        p for p in phrases
        if any(cp in p["phrase"].lower() for cp in concern_phrases)
        and p["status"] == "appeared"
    ]
    if not found_concern:
        return None

    # Check if negative sentiment is trending up
    neg_values = []
    for s in series:
        sd = s.get("sentiment_delta")
        if isinstance(sd, dict):
            neg_values.append(sd.get("negative", 0.0))
    sentiment_rising = False
    if len(neg_values) >= 2:
        sentiment_rising = neg_values[-1] > neg_values[-2]

    if not sentiment_rising and len(neg_values) > 0:
        # Still flag if concern language appeared, just lower score
        pass

    score = 0.7 if sentiment_rising else 0.4
    evidence = [f"Concern phrase appeared: '{p['phrase']}'" for p in found_concern[:5]]
    if sentiment_rising and neg_values:
        evidence.append(f"Negative sentiment: {neg_values[-2]:.4f} -> {neg_values[-1]:.4f}")

    return PatternMatch(
        pattern_name="going_concern_cascade",
        description=(
            "Going concern or substantial doubt language has appeared while "
            "negative sentiment is trending upward -- a strong distress signal."
        ),
        companies_matched=[],
        outcomes=["Historically associated with heightened default risk"],
        match_score=round(score, 3),
        evidence=evidence,
    )


def _check_obfuscation_wave(
    db: Session, company_id: int, series: list[dict],
) -> PatternMatch | None:
    """Obfuscation wave: sentiment stable while drift rising (hiding problems)."""
    if len(series) < 3:
        return None
    drifts = [s["cosine_distance"] for s in series[-4:]]
    neg_values = []
    for s in series[-4:]:
        sd = s.get("sentiment_delta")
        if isinstance(sd, dict):
            neg_values.append(abs(sd.get("negative", 0.0)))
        else:
            neg_values.append(0.0)

    if len(drifts) < 3 or len(neg_values) < 3:
        return None

    # Drift increasing but sentiment flat (stable obfuscation)
    drift_increasing = all(drifts[i] <= drifts[i + 1] for i in range(len(drifts) - 1))
    sentiment_flat = max(neg_values) - min(neg_values) < 0.02

    if not (drift_increasing and sentiment_flat):
        return None

    return PatternMatch(
        pattern_name="obfuscation_wave",
        description=(
            "Filing language is changing significantly (drift increasing) while "
            "sentiment remains stable -- the company may be altering disclosures "
            "without revealing new negative information (obfuscation)."
        ),
        companies_matched=[],
        outcomes=["Often precedes delayed revelation of adverse information"],
        match_score=0.6,
        evidence=[
            f"Drift trend: {' -> '.join(f'{d:.4f}' for d in drifts)}",
            f"Sentiment delta range: {max(neg_values) - min(neg_values):.4f} (flat)",
        ],
    )


def _check_risk_factor_explosion(
    db: Session, company_id: int, phrases: list[dict],
) -> PatternMatch | None:
    """Risk factor explosion: new-phrase count exceeds 2x historical baseline."""
    appeared = [p for p in phrases if p["status"] == "appeared"]
    if not appeared:
        return None

    # Group by filing date
    by_date: dict[str, int] = {}
    for p in appeared:
        key = str(p["filing_date"])
        by_date[key] = by_date.get(key, 0) + 1

    if len(by_date) < 2:
        return None

    counts = list(by_date.values())
    baseline = sum(counts[:-1]) / len(counts[:-1])
    latest = counts[-1]

    if baseline > 0 and latest >= baseline * 2:
        return PatternMatch(
            pattern_name="risk_factor_explosion",
            description=(
                f"Latest filing added {latest} new risk phrases, which is "
                f"{latest / baseline:.1f}x the historical average of {baseline:.1f}."
            ),
            companies_matched=[],
            outcomes=["Surge in new risk language often precedes earnings disappointments"],
            match_score=min(latest / (baseline * 4), 1.0),
            evidence=[
                f"Latest period: {latest} new phrases",
                f"Historical average: {baseline:.1f} new phrases per period",
            ],
        )
    return None


def _check_sentiment_reversal(
    db: Session, company_id: int, series: list[dict],
) -> PatternMatch | None:
    """Sentiment reversal: positive sentiment drops >50% in one quarter."""
    if len(series) < 2:
        return None

    pos_values = []
    for s in series:
        sd = s.get("sentiment_delta")
        if isinstance(sd, dict):
            pos_values.append(sd.get("positive", 0.0))

    if len(pos_values) < 2:
        return None

    prev, curr = pos_values[-2], pos_values[-1]
    if prev > 0.01 and curr < prev * 0.5:
        drop_pct = (1.0 - curr / prev) * 100
        return PatternMatch(
            pattern_name="sentiment_reversal",
            description=(
                f"Positive sentiment dropped {drop_pct:.0f}% in the latest period "
                f"({prev:.4f} -> {curr:.4f}) -- a sharp reversal."
            ),
            companies_matched=[],
            outcomes=["Sharp sentiment reversals correlate with negative forward guidance"],
            match_score=min(drop_pct / 100.0, 1.0),
            evidence=[
                f"Positive sentiment: {prev:.4f} -> {curr:.4f}",
                f"Drop: {drop_pct:.0f}%",
            ],
        )
    return None


def _check_silent_deletion(
    db: Session, company_id: int, phrases: list[dict],
) -> PatternMatch | None:
    """Silent deletion: key phrases disappearing from risk factors."""
    disappeared = [p for p in phrases if p["status"] == "disappeared"]
    if len(disappeared) < 2:
        return None

    # Group by filing date; check if the latest period has many removals
    by_date: dict[str, list[str]] = {}
    for p in disappeared:
        key = str(p["filing_date"])
        by_date.setdefault(key, []).append(p["phrase"])

    if not by_date:
        return None

    latest_key = max(by_date.keys())
    latest_removals = by_date[latest_key]
    if len(latest_removals) < 2:
        return None

    return PatternMatch(
        pattern_name="silent_deletion",
        description=(
            f"{len(latest_removals)} key phrases were removed from the latest filing "
            "without explanation -- removing previously disclosed risks is a red flag."
        ),
        companies_matched=[],
        outcomes=["Deletion of risk disclosures can indicate concealment of ongoing issues"],
        match_score=min(len(latest_removals) / 10.0, 1.0),
        evidence=[f"Removed: '{p}'" for p in latest_removals[:10]],
    )


def _check_peer_divergence(
    db: Session, company_id: int, company_sic: str | None, series: list[dict],
) -> PatternMatch | None:
    """Peer divergence: company drift diverging from sector average."""
    if not company_sic or len(series) < 2:
        return None

    # Find peer companies (same 2-digit SIC)
    sic_prefix = company_sic[:2]
    peer_stmt = (
        select(Company.id)
        .where(Company.sic_code.like(f"{sic_prefix}%"), Company.id != company_id)
    )
    peer_ids = [r for r in db.execute(peer_stmt).scalars().all()]
    if not peer_ids:
        return None

    # Get latest drift for peers
    peer_drifts: list[float] = []
    for pid in peer_ids[:50]:  # cap for performance
        peer_series = _drift_series(db, pid)
        if peer_series:
            peer_drifts.append(peer_series[-1]["cosine_distance"])

    if len(peer_drifts) < 3:
        return None

    company_latest = series[-1]["cosine_distance"]
    peer_mean = sum(peer_drifts) / len(peer_drifts)
    peer_std = math.sqrt(sum((d - peer_mean) ** 2 for d in peer_drifts) / len(peer_drifts))

    if peer_std < 0.001:
        return None

    z_score = (company_latest - peer_mean) / peer_std
    if z_score < 1.5:
        return None

    return PatternMatch(
        pattern_name="peer_divergence",
        description=(
            f"Company's latest drift ({company_latest:.4f}) is {z_score:.1f} "
            f"standard deviations above sector average ({peer_mean:.4f}). "
            "The company is changing disclosures while peers remain stable."
        ),
        companies_matched=[],
        outcomes=["Divergence from peers suggests company-specific risk emergence"],
        match_score=min(z_score / 4.0, 1.0),
        evidence=[
            f"Company drift: {company_latest:.4f}",
            f"Sector mean: {peer_mean:.4f}, std: {peer_std:.4f}",
            f"Z-score: {z_score:.2f} ({len(peer_drifts)} peers)",
        ],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_matching_patterns(db: Session, company_id: int) -> list[PatternMatch]:
    """Check all defined patterns against a company's history.

    Args:
        db: Synchronous SQLAlchemy session.
        company_id: Primary key of the company to evaluate.

    Returns:
        List of PatternMatch objects, sorted by match_score descending.
    """
    company = db.execute(
        select(Company).where(Company.id == company_id)
    ).scalar_one_or_none()
    if not company:
        logger.warning("Company id=%d not found", company_id)
        return []

    series = _drift_series(db, company_id)
    phrases = _phrase_history(db, company_id)

    checkers = [
        lambda: _check_lazy_prices(db, company_id, series),
        lambda: _check_going_concern_cascade(db, company_id, series, phrases),
        lambda: _check_obfuscation_wave(db, company_id, series),
        lambda: _check_risk_factor_explosion(db, company_id, phrases),
        lambda: _check_sentiment_reversal(db, company_id, series),
        lambda: _check_silent_deletion(db, company_id, phrases),
        lambda: _check_peer_divergence(db, company_id, company.sic_code, series),
    ]

    matches: list[PatternMatch] = []
    for checker in checkers:
        try:
            result = checker()
            if result is not None:
                matches.append(result)
        except Exception:
            logger.warning("Pattern check failed", exc_info=True)
            continue

    matches.sort(key=lambda m: m.match_score, reverse=True)
    return matches
