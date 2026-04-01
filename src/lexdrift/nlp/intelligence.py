"""Signal Synthesis Engine -- transforms raw NLP measurements into actionable
financial intelligence.

This is the "brain" of LexDrift.  It gathers every available signal for a
company (drift scores, sentiment deltas, phrase changes, obfuscation,
entropy, kinematics, anomaly z-scores, risk classifications, sentence-level
changes), scores overall risk, generates findings with specific evidence,
matches historical patterns, identifies comparable companies, and produces
a structured CompanyIntelligence assessment.

Usage (CLI):
    python -m lexdrift.nlp.intelligence TSLA
    python -m lexdrift.nlp.intelligence --ticker AAPL

Usage (programmatic):
    from lexdrift.nlp.intelligence import generate_intelligence
    report = generate_intelligence(db_session, "TSLA")
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine, desc, func, select
from sqlalchemy.orm import Session, sessionmaker

from lexdrift.config import settings
from lexdrift.db.models import (
    Alert,
    Company,
    DriftScore,
    Filing,
    KeyPhrase,
    SentenceChange,
    Section,
)
from lexdrift.nlp.anomaly import detect_anomaly, detect_trends
from lexdrift.nlp.patterns import PatternMatch, find_matching_patterns
from lexdrift.nlp.velocity import compute_semantic_kinematics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    """A single intelligence finding backed by evidence."""
    severity: str  # critical, high, medium, low
    category: str  # drift, sentiment, phrases, obfuscation, anomaly, trend
    title: str
    detail: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class Prediction:
    """A forward-looking prediction based on historical patterns."""
    probability: str  # high, moderate, low
    event: str
    basis: str
    confidence: float  # 0.0 to 1.0


@dataclass
class Comparable:
    """A company showing a similar signal profile."""
    ticker: str
    company_name: str
    similarity: float
    outcome: str


@dataclass
class SignalSummary:
    """Raw signal snapshot for transparency."""
    drift_velocity: float | None = None
    drift_acceleration: float | None = None
    drift_phase: str | None = None
    sentiment_trend: str | None = None  # improving, stable, deteriorating
    anomaly_level: str | None = None
    obfuscation_score: float | None = None
    entropy_novelty: float | None = None
    new_risk_phrases: list[str] = field(default_factory=list)
    removed_risk_phrases: list[str] = field(default_factory=list)
    critical_sentence_changes: int = 0


@dataclass
class CompanyIntelligence:
    """Synthesized intelligence assessment for a company."""
    ticker: str
    company_name: str
    assessment_date: str

    # Overall verdict
    risk_level: str  # minimal, low, moderate, elevated, high, critical
    risk_score: float  # 0.0 to 1.0
    headline: str

    # Key findings (ordered by importance)
    findings: list[Finding] = field(default_factory=list)

    # Predicted risks
    predictions: list[Prediction] = field(default_factory=list)

    # Recommended actions
    actions: list[str] = field(default_factory=list)

    # Comparable situations
    comparables: list[Comparable] = field(default_factory=list)

    # Raw signal summary
    signals: SignalSummary = field(default_factory=SignalSummary)

    # Pattern matches
    patterns: list[PatternMatch] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Risk level mapping
# ---------------------------------------------------------------------------

_RISK_LEVELS = [
    (0.15, "minimal"),
    (0.30, "low"),
    (0.45, "moderate"),
    (0.60, "elevated"),
    (0.75, "high"),
    (float("inf"), "critical"),
]

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _risk_level(score: float) -> str:
    for threshold, level in _RISK_LEVELS:
        if score < threshold:
            return level
    return "critical"


# ---------------------------------------------------------------------------
# Sync DB helper
# ---------------------------------------------------------------------------

def _make_sync_url(async_url: str) -> str:
    url = async_url.replace("+aiosqlite", "")
    url = url.replace("+asyncpg", "+psycopg2")
    return url


def _get_sync_session() -> Session:
    db_url = settings.database_url
    sync_url = _make_sync_url(db_url)
    eng = create_engine(sync_url, echo=False, future=True)
    factory = sessionmaker(bind=eng, class_=Session, expire_on_commit=False)
    return factory()


# ---------------------------------------------------------------------------
# Signal gathering
# ---------------------------------------------------------------------------

def _gather_drift_data(db: Session, company_id: int) -> dict:
    """Gather all drift scores for a company, grouped by section."""
    stmt = (
        select(
            DriftScore.section_type,
            DriftScore.cosine_distance,
            DriftScore.jaccard_distance,
            DriftScore.sentiment_delta,
            DriftScore.added_words,
            DriftScore.removed_words,
            Filing.filing_date,
        )
        .join(Filing, DriftScore.filing_id == Filing.id)
        .where(DriftScore.company_id == company_id)
        .order_by(Filing.filing_date)
    )
    rows = db.execute(stmt).all()

    by_section: dict[str, list[dict]] = defaultdict(list)
    all_drifts: list[float] = []
    all_sentiments: list[dict] = []

    for r in rows:
        entry = {
            "cosine_distance": r.cosine_distance,
            "jaccard_distance": r.jaccard_distance,
            "sentiment_delta": r.sentiment_delta,
            "added_words": r.added_words,
            "removed_words": r.removed_words,
            "filing_date": r.filing_date,
        }
        by_section[r.section_type].append(entry)
        if r.cosine_distance is not None:
            all_drifts.append(r.cosine_distance)
        if r.sentiment_delta and isinstance(r.sentiment_delta, dict):
            all_sentiments.append(r.sentiment_delta)

    return {
        "by_section": dict(by_section),
        "all_drifts": all_drifts,
        "all_sentiments": all_sentiments,
    }


def _gather_phrases(db: Session, company_id: int) -> dict:
    """Gather key-phrase data for the latest filing."""
    # Get latest filing
    latest_filing_stmt = (
        select(Filing.id)
        .where(Filing.company_id == company_id)
        .order_by(desc(Filing.filing_date))
        .limit(1)
    )
    latest_filing_id = db.execute(latest_filing_stmt).scalar_one_or_none()
    if not latest_filing_id:
        return {"appeared": [], "disappeared": [], "all_phrases": []}

    stmt = select(KeyPhrase).where(KeyPhrase.filing_id == latest_filing_id)
    phrases = db.execute(stmt).scalars().all()

    appeared = [p.phrase for p in phrases if p.status == "appeared"]
    disappeared = [p.phrase for p in phrases if p.status == "disappeared"]

    return {
        "appeared": appeared,
        "disappeared": disappeared,
        "all_phrases": phrases,
    }


def _gather_alerts(db: Session, company_id: int) -> list:
    """Gather recent alerts for a company."""
    stmt = (
        select(Alert)
        .where(Alert.company_id == company_id)
        .order_by(desc(Alert.created_at))
        .limit(20)
    )
    return db.execute(stmt).scalars().all()


def _gather_sentence_changes(db: Session, company_id: int) -> dict:
    """Gather sentence-level changes for the latest drift scores."""
    stmt = (
        select(SentenceChange)
        .join(DriftScore, SentenceChange.drift_score_id == DriftScore.id)
        .where(DriftScore.company_id == company_id)
        .order_by(desc(DriftScore.computed_at))
        .limit(200)
    )
    changes = db.execute(stmt).scalars().all()
    added = [c for c in changes if c.change_type == "added"]
    removed = [c for c in changes if c.change_type == "removed"]
    changed = [c for c in changes if c.change_type == "changed"]
    return {"added": added, "removed": removed, "changed": changed, "total": len(changes)}


# ---------------------------------------------------------------------------
# Risk scoring
# ---------------------------------------------------------------------------

_CRITICAL_PHRASES = {
    "going concern", "substantial doubt", "material weakness",
    "restatement", "fraud", "sec investigation", "delisted",
    "default", "bankruptcy", "insolvency",
}


def _compute_risk_score(
    drift_data: dict,
    phrases: dict,
    alerts: list,
    kinematics: dict | None,
    anomaly_result: dict | None,
    sentence_changes: dict,
) -> tuple[float, dict[str, float]]:
    """Compute weighted risk score. Returns (score, component_contributions)."""
    components: dict[str, float] = {}

    # 1. Anomaly z-score (weight 0.25)
    anomaly_raw = 0.0
    if anomaly_result:
        z = anomaly_result.get("company_z_score")
        if z is not None and z > 0:
            anomaly_raw = min(z / 4.0, 1.0)  # normalize: z=4 -> 1.0
        elif anomaly_result.get("anomaly_level") == "extreme":
            anomaly_raw = 1.0
        elif anomaly_result.get("anomaly_level") == "high":
            anomaly_raw = 0.7
        elif anomaly_result.get("anomaly_level") == "elevated":
            anomaly_raw = 0.4
    components["anomaly_z_score"] = anomaly_raw

    # 2. Drift acceleration (weight 0.20)
    accel_raw = 0.0
    if kinematics:
        accel = kinematics.get("latest_acceleration", 0.0)
        if accel > 0:
            accel_raw = min(accel / 0.1, 1.0)
        phase = kinematics.get("phase", "")
        if phase == "regime_change":
            accel_raw = max(accel_raw, 0.9)
        elif phase == "accelerating":
            accel_raw = max(accel_raw, 0.5)
    components["drift_acceleration"] = accel_raw

    # 3. Sentiment deterioration (weight 0.15)
    sentiment_raw = 0.0
    sentiments = drift_data.get("all_sentiments", [])
    if len(sentiments) >= 2:
        recent_neg = [s.get("negative", 0.0) for s in sentiments[-4:] if isinstance(s, dict)]
        if len(recent_neg) >= 2 and recent_neg[-1] > recent_neg[0]:
            delta = recent_neg[-1] - recent_neg[0]
            sentiment_raw = min(delta / 0.1, 1.0)
    components["sentiment_deterioration"] = sentiment_raw

    # 4. Critical phrase appearances (weight 0.15)
    phrase_raw = 0.0
    new_phrases = phrases.get("appeared", [])
    critical_found = [p for p in new_phrases if any(cp in p.lower() for cp in _CRITICAL_PHRASES)]
    if critical_found:
        phrase_raw = min(len(critical_found) / 3.0, 1.0)
    elif len(new_phrases) > 10:
        phrase_raw = 0.4
    elif len(new_phrases) > 5:
        phrase_raw = 0.2
    components["critical_phrases"] = phrase_raw

    # 5. Obfuscation score (weight 0.10)
    # We don't compute obfuscation in real-time (requires section text);
    # use drift-sentiment divergence as a proxy
    obfuscation_raw = 0.0
    drifts = drift_data.get("all_drifts", [])
    if drifts and sentiments:
        latest_drift = drifts[-1]
        latest_neg = sentiments[-1].get("negative", 0.0) if isinstance(sentiments[-1], dict) else 0.0
        # High drift + low sentiment change = possible obfuscation
        if latest_drift > 0.15 and latest_neg < 0.02:
            obfuscation_raw = min(latest_drift / 0.3, 1.0) * 0.6
    components["obfuscation"] = obfuscation_raw

    # 6. Entropy novelty (weight 0.10)
    # Proxy: use drift magnitude as stand-in for entropy
    entropy_raw = 0.0
    if drifts:
        latest_drift = drifts[-1]
        if latest_drift > 0.2:
            entropy_raw = min((latest_drift - 0.1) / 0.3, 1.0)
    components["entropy_novelty"] = entropy_raw

    # 7. Recent alert count (weight 0.05)
    alert_raw = 0.0
    if alerts:
        recent_alert_count = len([a for a in alerts[:10]])
        alert_raw = min(recent_alert_count / 5.0, 1.0)
    components["recent_alerts"] = alert_raw

    # Weighted sum
    weights = {
        "anomaly_z_score": 0.25,
        "drift_acceleration": 0.20,
        "sentiment_deterioration": 0.15,
        "critical_phrases": 0.15,
        "obfuscation": 0.10,
        "entropy_novelty": 0.10,
        "recent_alerts": 0.05,
    }

    total = sum(components[k] * weights[k] for k in weights)
    total = round(min(max(total, 0.0), 1.0), 4)

    logger.info("Risk score components: %s -> total=%.4f", components, total)
    return total, components


# ---------------------------------------------------------------------------
# Finding generation
# ---------------------------------------------------------------------------

def _generate_findings(
    drift_data: dict,
    phrases: dict,
    alerts: list,
    kinematics: dict | None,
    anomaly_result: dict | None,
    sentence_changes: dict,
    risk_components: dict[str, float],
    patterns: list[PatternMatch],
) -> list[Finding]:
    """Generate findings from all available signals."""
    findings: list[Finding] = []

    # Anomaly findings
    if anomaly_result:
        level = anomaly_result.get("anomaly_level", "normal")
        if level in ("extreme", "high"):
            z = anomaly_result.get("company_z_score", "N/A")
            findings.append(Finding(
                severity="critical" if level == "extreme" else "high",
                category="anomaly",
                title=f"Anomalous filing detected ({level})",
                detail=(
                    f"This filing's drift is {z} standard deviations above the company's "
                    f"historical mean ({anomaly_result.get('company_mean', 'N/A')})."
                ),
                evidence=[
                    f"Z-score: {z}",
                    f"Company mean drift: {anomaly_result.get('company_mean', 'N/A')}",
                    f"Company std dev: {anomaly_result.get('company_stddev', 'N/A')}",
                ],
            ))
        elif level == "elevated":
            findings.append(Finding(
                severity="medium",
                category="anomaly",
                title="Elevated drift relative to company history",
                detail=f"Z-score of {anomaly_result.get('company_z_score', 'N/A')} -- not anomalous but above average.",
                evidence=[f"Anomaly level: {level}"],
            ))

    # Kinematics findings
    if kinematics:
        phase = kinematics.get("phase", "stable")
        vel = kinematics.get("latest_velocity", 0)
        accel = kinematics.get("latest_acceleration", 0)
        if phase == "regime_change":
            findings.append(Finding(
                severity="critical",
                category="trend",
                title="Disclosure regime change detected",
                detail=(
                    f"The company's filing language is undergoing a structural shift. "
                    f"Velocity={vel:.4f}, Acceleration={accel:.4f}."
                ),
                evidence=[f"Phase: {phase}", f"Velocity: {vel:.4f}", f"Acceleration: {accel:.4f}"],
            ))
        elif phase == "accelerating":
            findings.append(Finding(
                severity="high",
                category="trend",
                title="Drift acceleration detected",
                detail=f"Filings are changing faster each period. Acceleration={accel:.4f}.",
                evidence=[f"Phase: {phase}", f"Velocity: {vel:.4f}", f"Acceleration: {accel:.4f}"],
            ))
        elif phase == "volatile":
            findings.append(Finding(
                severity="medium",
                category="trend",
                title="Volatile disclosure pattern",
                detail="Filing language is oscillating unpredictably between periods.",
                evidence=[f"Phase: {phase}", f"Velocity std: {kinematics.get('velocity_std', 'N/A')}"],
            ))

    # Phrase findings
    new_phrases = phrases.get("appeared", [])
    removed_phrases = phrases.get("disappeared", [])

    critical_new = [p for p in new_phrases if any(cp in p.lower() for cp in _CRITICAL_PHRASES)]
    if critical_new:
        findings.append(Finding(
            severity="critical",
            category="phrases",
            title="Critical risk language detected",
            detail=f"{len(critical_new)} critical phrase(s) appeared in the latest filing.",
            evidence=[f"New phrase: '{p}'" for p in critical_new[:10]],
        ))
    elif len(new_phrases) > 5:
        findings.append(Finding(
            severity="medium",
            category="phrases",
            title=f"{len(new_phrases)} new risk phrases added",
            detail="A significant number of new risk phrases appeared in this filing.",
            evidence=[f"New: '{p}'" for p in new_phrases[:10]],
        ))

    if len(removed_phrases) > 3:
        findings.append(Finding(
            severity="medium",
            category="phrases",
            title=f"{len(removed_phrases)} risk phrases removed",
            detail="Multiple risk phrases were removed -- may indicate concealment or resolution.",
            evidence=[f"Removed: '{p}'" for p in removed_phrases[:10]],
        ))

    # Sentiment findings
    sentiments = drift_data.get("all_sentiments", [])
    if len(sentiments) >= 2:
        recent = sentiments[-1]
        prev = sentiments[-2]
        if isinstance(recent, dict) and isinstance(prev, dict):
            neg_curr = recent.get("negative", 0.0)
            neg_prev = prev.get("negative", 0.0)
            if neg_curr > neg_prev + 0.05:
                findings.append(Finding(
                    severity="high",
                    category="sentiment",
                    title="Significant sentiment deterioration",
                    detail=f"Negative sentiment increased from {neg_prev:.4f} to {neg_curr:.4f}.",
                    evidence=[
                        f"Previous negative: {neg_prev:.4f}",
                        f"Current negative: {neg_curr:.4f}",
                        f"Delta: +{neg_curr - neg_prev:.4f}",
                    ],
                ))

    # Sentence change findings
    added_sentences = sentence_changes.get("added", [])
    removed_sentences = sentence_changes.get("removed", [])
    if len(added_sentences) > 20:
        sample = added_sentences[:5]
        findings.append(Finding(
            severity="medium",
            category="drift",
            title=f"{len(added_sentences)} new sentences added",
            detail="Significant new content was added to the filing.",
            evidence=[f"Added: \"{s.sentence_text[:120]}...\"" if len(s.sentence_text) > 120
                      else f"Added: \"{s.sentence_text}\"" for s in sample],
        ))
    if len(removed_sentences) > 20:
        sample = removed_sentences[:5]
        findings.append(Finding(
            severity="medium",
            category="drift",
            title=f"{len(removed_sentences)} sentences removed",
            detail="Significant content was removed from the filing.",
            evidence=[f"Removed: \"{s.sentence_text[:120]}...\"" if len(s.sentence_text) > 120
                      else f"Removed: \"{s.sentence_text}\"" for s in sample],
        ))

    # Pattern-based findings
    for pattern in patterns:
        sev = "high" if pattern.match_score > 0.6 else "medium"
        findings.append(Finding(
            severity=sev,
            category="trend",
            title=f"Pattern: {pattern.pattern_name.replace('_', ' ').title()}",
            detail=pattern.description,
            evidence=pattern.evidence[:5],
        ))

    # Alert-based findings
    if alerts:
        recent_critical = [a for a in alerts[:10] if a.severity in ("critical", "high")]
        if recent_critical:
            findings.append(Finding(
                severity="high",
                category="anomaly",
                title=f"{len(recent_critical)} high-severity alerts active",
                detail="Recent alerts have been triggered for this company.",
                evidence=[f"Alert: {a.message[:120]}" if a.message else f"Alert type: {a.alert_type}"
                          for a in recent_critical[:5]],
            ))

    # Sort by severity
    findings.sort(key=lambda f: _SEVERITY_ORDER.get(f.severity, 99))
    return findings


# ---------------------------------------------------------------------------
# Prediction generation
# ---------------------------------------------------------------------------

def _generate_predictions(
    risk_score: float,
    findings: list[Finding],
    patterns: list[PatternMatch],
    drift_data: dict,
) -> list[Prediction]:
    """Generate predictions based on patterns and signal strength."""
    predictions: list[Prediction] = []

    # Pattern-based predictions
    for p in patterns:
        if p.pattern_name == "lazy_prices" and p.match_score > 0.5:
            predictions.append(Prediction(
                probability="moderate" if p.match_score < 0.7 else "high",
                event="Negative return anomaly within 2 quarters",
                basis=(
                    f"Based on Lazy Prices research (Cohen et al. 2020). "
                    f"{len(p.companies_matched)} historical matches found."
                ),
                confidence=round(p.match_score * 0.7, 2),
            ))

        elif p.pattern_name == "going_concern_cascade":
            predictions.append(Prediction(
                probability="high",
                event="Material going-concern risk within 4 quarters",
                basis="Going concern language combined with rising negative sentiment.",
                confidence=round(p.match_score * 0.8, 2),
            ))

        elif p.pattern_name == "sentiment_reversal":
            predictions.append(Prediction(
                probability="moderate",
                event="Earnings miss or guidance reduction within 1-2 quarters",
                basis="Sharp sentiment reversals historically precede negative forward guidance.",
                confidence=round(p.match_score * 0.6, 2),
            ))

        elif p.pattern_name == "risk_factor_explosion":
            predictions.append(Prediction(
                probability="moderate",
                event="Increased regulatory or legal disclosure within 2 quarters",
                basis="Sudden expansion of risk factors often precedes formal disclosures.",
                confidence=round(p.match_score * 0.5, 2),
            ))

    # Risk-level-based predictions
    if risk_score >= 0.75 and not any(pr.probability == "high" for pr in predictions):
        predictions.append(Prediction(
            probability="high",
            event="Material adverse development within 4 quarters",
            basis=f"Composite risk score of {risk_score:.2f} places this company in the critical zone.",
            confidence=round(risk_score * 0.6, 2),
        ))
    elif risk_score >= 0.45:
        has_accel = any(f.category == "trend" and "accel" in f.title.lower() for f in findings)
        if has_accel:
            predictions.append(Prediction(
                probability="moderate",
                event="Continued drift escalation in next filing",
                basis="Accelerating drift pattern suggests continued divergence.",
                confidence=round(risk_score * 0.5, 2),
            ))

    return predictions


# ---------------------------------------------------------------------------
# Comparable finding
# ---------------------------------------------------------------------------

def _find_comparables(
    db: Session, company_id: int, company_sic: str | None,
) -> list[Comparable]:
    """Find companies with similar current drift profiles."""
    # Get this company's latest drift
    stmt = (
        select(DriftScore.cosine_distance)
        .join(Filing, DriftScore.filing_id == Filing.id)
        .where(DriftScore.company_id == company_id)
        .order_by(desc(Filing.filing_date))
        .limit(1)
    )
    own_drift = db.execute(stmt).scalar_one_or_none()
    if own_drift is None:
        return []

    # Find peers -- prefer same SIC, but also include similar drift
    peer_filter = Company.id != company_id
    if company_sic:
        sic_prefix = company_sic[:2]
        peer_filter = (Company.id != company_id) & (Company.sic_code.like(f"{sic_prefix}%"))

    peer_stmt = select(Company).where(peer_filter).limit(50)
    peers = db.execute(peer_stmt).scalars().all()

    comparables: list[Comparable] = []
    for peer in peers:
        peer_drift_stmt = (
            select(DriftScore.cosine_distance)
            .join(Filing, DriftScore.filing_id == Filing.id)
            .where(DriftScore.company_id == peer.id)
            .order_by(desc(Filing.filing_date))
            .limit(1)
        )
        peer_drift = db.execute(peer_drift_stmt).scalar_one_or_none()
        if peer_drift is None:
            continue

        # Similarity = 1 - normalized distance between drift scores
        distance = abs(own_drift - peer_drift)
        similarity = max(1.0 - distance / max(own_drift, peer_drift, 0.01), 0.0)

        if similarity > 0.5:
            # Check outcome: count alerts
            alert_count = db.execute(
                select(func.count(Alert.id)).where(Alert.company_id == peer.id)
            ).scalar() or 0
            outcome = f"{alert_count} alert(s)" if alert_count else "No alerts"

            comparables.append(Comparable(
                ticker=peer.ticker or peer.cik,
                company_name=peer.name or "Unknown",
                similarity=round(similarity, 3),
                outcome=outcome,
            ))

    comparables.sort(key=lambda c: c.similarity, reverse=True)
    return comparables[:5]


# ---------------------------------------------------------------------------
# Headline and action generation
# ---------------------------------------------------------------------------

def _generate_headline(
    ticker: str, risk_level: str, findings: list[Finding],
) -> str:
    """Generate a one-sentence headline summarizing the most important finding."""
    if not findings:
        return f"{ticker}: {risk_level.title()} risk -- no significant findings"

    top = findings[0]
    return f"{ticker}: {risk_level.title()} risk -- {top.title.lower()}"


def _generate_actions(
    risk_level: str, findings: list[Finding], patterns: list[PatternMatch],
) -> list[str]:
    """Generate recommended actions based on risk level and findings."""
    actions: list[str] = []

    if risk_level == "critical":
        # Check for specific critical findings
        critical_findings = [f for f in findings if f.severity == "critical"]
        for f in critical_findings[:3]:
            actions.append(f"Immediate review required -- {f.title.lower()}")
        if not critical_findings:
            actions.append("Immediate review required -- critical risk score")

    elif risk_level == "high":
        actions.append("Monitor closely -- elevated risk across multiple signals")
        for f in findings[:2]:
            if f.severity in ("critical", "high"):
                actions.append(f"Investigate: {f.title}")

    elif risk_level in ("elevated", "moderate"):
        topics = set()
        for f in findings[:3]:
            if f.category == "phrases" and f.evidence:
                topics.update(e.split("'")[1] for e in f.evidence if "'" in e)
        if topics:
            topic_str = ", ".join(list(topics)[:3])
            actions.append(f"Watch for escalation -- new risk language around: {topic_str}")
        else:
            actions.append("Watch for escalation in the next filing period")

    elif risk_level in ("low", "minimal"):
        actions.append("No action needed -- filing within historical norms")

    # Pattern-specific actions
    for p in patterns[:2]:
        if p.pattern_name == "lazy_prices" and p.match_score > 0.5:
            actions.append("Review for Lazy Prices signal -- consider position adjustment")
        elif p.pattern_name == "silent_deletion":
            actions.append("Review removed risk phrases for potential concealment")

    return actions


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_intelligence(db: Session, ticker: str) -> CompanyIntelligence:
    """Generate a comprehensive intelligence assessment for a company.

    Args:
        db: Synchronous SQLAlchemy session.
        ticker: Company ticker symbol (e.g. "TSLA").

    Returns:
        CompanyIntelligence with risk score, findings, predictions, and actions.
    """
    ticker = ticker.upper()

    # Resolve company
    company = db.execute(
        select(Company).where(Company.ticker == ticker)
    ).scalar_one_or_none()

    if not company:
        # Try CIK lookup
        company = db.execute(
            select(Company).where(Company.cik == ticker)
        ).scalar_one_or_none()

    if not company:
        return CompanyIntelligence(
            ticker=ticker,
            company_name="Unknown",
            assessment_date=datetime.utcnow().strftime("%Y-%m-%d"),
            risk_level="minimal",
            risk_score=0.0,
            headline=f"{ticker}: Company not found in database",
            actions=["Ingest filings for this company first"],
        )

    # Gather all signals
    drift_data = _gather_drift_data(db, company.id)
    phrases = _gather_phrases(db, company.id)
    alerts = _gather_alerts(db, company.id)
    sentence_changes = _gather_sentence_changes(db, company.id)

    # Kinematics
    kinematics_result: dict | None = None
    by_section = drift_data["by_section"]
    # Prefer risk_factors section
    kin_section = by_section.get("risk_factors") or (list(by_section.values())[0] if by_section else None)
    if kin_section and len(kin_section) >= 2:
        try:
            history = [
                {"filing_date": e["filing_date"].isoformat() if hasattr(e["filing_date"], "isoformat") else str(e["filing_date"]),
                 "cosine_distance": e["cosine_distance"]}
                for e in kin_section
                if e["cosine_distance"] is not None and e["filing_date"] is not None
            ]
            if len(history) >= 2:
                kin = compute_semantic_kinematics(history)
                kinematics_result = {
                    "latest_velocity": kin.latest_velocity,
                    "latest_acceleration": kin.latest_acceleration,
                    "latest_momentum": kin.latest_momentum,
                    "phase": kin.phase,
                    "velocity_std": kin.velocity_std,
                }
        except Exception:
            logger.warning("Kinematics computation failed for %s", ticker, exc_info=True)

    # Anomaly detection
    anomaly_result: dict | None = None
    all_drifts = drift_data["all_drifts"]
    if all_drifts and len(all_drifts) >= 2:
        try:
            current = all_drifts[-1]
            history = all_drifts[:-1]
            anom = detect_anomaly(current, history)
            anomaly_result = {
                "company_z_score": anom.company_z_score,
                "is_anomalous": anom.is_anomalous,
                "anomaly_level": anom.anomaly_level,
                "company_mean": anom.company_mean,
                "company_stddev": anom.company_stddev,
            }
        except Exception:
            logger.warning("Anomaly detection failed for %s", ticker, exc_info=True)

    # Pattern matching
    patterns = find_matching_patterns(db, company.id)

    # Compute risk score
    risk_score, risk_components = _compute_risk_score(
        drift_data, phrases, alerts, kinematics_result, anomaly_result, sentence_changes,
    )
    risk_level = _risk_level(risk_score)

    # Generate findings
    findings = _generate_findings(
        drift_data, phrases, alerts, kinematics_result, anomaly_result,
        sentence_changes, risk_components, patterns,
    )

    # Generate predictions
    predictions = _generate_predictions(risk_score, findings, patterns, drift_data)

    # Find comparables
    comparables = _find_comparables(db, company.id, company.sic_code)

    # Headline and actions
    headline = _generate_headline(ticker, risk_level, findings)
    actions = _generate_actions(risk_level, findings, patterns)

    # Trend detection
    sentiments = drift_data["all_sentiments"]
    trends = detect_trends(all_drifts, sentiments if sentiments else None)
    sentiment_trend = "stable"
    if trends.get("has_trend"):
        for sig in trends.get("signals", []):
            if sig.get("type") == "sentiment_deterioration":
                sentiment_trend = "deteriorating"
                break
            elif sig.get("type") == "drift_acceleration":
                sentiment_trend = "deteriorating"

    # Build signal summary
    signals = SignalSummary(
        drift_velocity=kinematics_result["latest_velocity"] if kinematics_result else None,
        drift_acceleration=kinematics_result["latest_acceleration"] if kinematics_result else None,
        drift_phase=kinematics_result["phase"] if kinematics_result else None,
        sentiment_trend=sentiment_trend,
        anomaly_level=anomaly_result["anomaly_level"] if anomaly_result else None,
        obfuscation_score=risk_components.get("obfuscation"),
        entropy_novelty=risk_components.get("entropy_novelty"),
        new_risk_phrases=phrases.get("appeared", []),
        removed_risk_phrases=phrases.get("disappeared", []),
        critical_sentence_changes=sentence_changes.get("total", 0),
    )

    return CompanyIntelligence(
        ticker=ticker,
        company_name=company.name or "Unknown",
        assessment_date=datetime.utcnow().strftime("%Y-%m-%d"),
        risk_level=risk_level,
        risk_score=risk_score,
        headline=headline,
        findings=findings,
        predictions=predictions,
        actions=actions,
        comparables=comparables,
        signals=signals,
        patterns=patterns,
    )


# ---------------------------------------------------------------------------
# CLI pretty printer
# ---------------------------------------------------------------------------

def _print_report(intel: CompanyIntelligence) -> None:
    """Print a human-readable intelligence report to stdout."""
    sep = "=" * 72
    thin = "-" * 72

    print(f"\n{sep}")
    print(f"  LEXDRIFT INTELLIGENCE REPORT")
    print(f"  {intel.company_name} ({intel.ticker})")
    print(f"  Assessment Date: {intel.assessment_date}")
    print(sep)

    # Headline
    print(f"\n  >> {intel.headline}")
    print(f"  Risk Level: {intel.risk_level.upper()}  |  Risk Score: {intel.risk_score:.2f}")

    # Signals summary
    print(f"\n{thin}")
    print("  SIGNAL SUMMARY")
    print(thin)
    s = intel.signals
    if s.drift_velocity is not None:
        print(f"  Drift Velocity:      {s.drift_velocity:.6f}")
    if s.drift_acceleration is not None:
        print(f"  Drift Acceleration:  {s.drift_acceleration:.6f}")
    if s.drift_phase:
        print(f"  Drift Phase:         {s.drift_phase}")
    if s.sentiment_trend:
        print(f"  Sentiment Trend:     {s.sentiment_trend}")
    if s.anomaly_level:
        print(f"  Anomaly Level:       {s.anomaly_level}")
    if s.obfuscation_score is not None:
        print(f"  Obfuscation Proxy:   {s.obfuscation_score:.4f}")
    if s.entropy_novelty is not None:
        print(f"  Entropy Novelty:     {s.entropy_novelty:.4f}")
    print(f"  Sentence Changes:    {s.critical_sentence_changes}")
    if s.new_risk_phrases:
        print(f"  New Risk Phrases:    {', '.join(s.new_risk_phrases[:10])}")
    if s.removed_risk_phrases:
        print(f"  Removed Phrases:     {', '.join(s.removed_risk_phrases[:10])}")

    # Findings
    if intel.findings:
        print(f"\n{thin}")
        print("  KEY FINDINGS")
        print(thin)
        for i, f in enumerate(intel.findings, 1):
            marker = {"critical": "***", "high": "** ", "medium": "*  ", "low": "   "}.get(f.severity, "   ")
            print(f"  {marker} [{f.severity.upper()}] {f.title}")
            print(f"      {f.detail}")
            for e in f.evidence[:3]:
                print(f"        - {e}")
            print()

    # Predictions
    if intel.predictions:
        print(f"{thin}")
        print("  PREDICTIONS")
        print(thin)
        for p in intel.predictions:
            print(f"  [{p.probability.upper()} probability] {p.event}")
            print(f"    Basis: {p.basis}")
            print(f"    Confidence: {p.confidence:.0%}")
            print()

    # Comparables
    if intel.comparables:
        print(f"{thin}")
        print("  COMPARABLE COMPANIES")
        print(thin)
        for c in intel.comparables:
            print(f"  {c.ticker} ({c.company_name}) -- similarity: {c.similarity:.1%}, outcome: {c.outcome}")

    # Actions
    if intel.actions:
        print(f"\n{thin}")
        print("  RECOMMENDED ACTIONS")
        print(thin)
        for a in intel.actions:
            print(f"  > {a}")

    # Patterns
    if intel.patterns:
        print(f"\n{thin}")
        print("  MATCHED PATTERNS")
        print(thin)
        for p in intel.patterns:
            print(f"  [{p.match_score:.0%}] {p.pattern_name.replace('_', ' ').title()}")
            print(f"    {p.description}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="LexDrift Intelligence Report Generator",
    )
    parser.add_argument(
        "ticker",
        nargs="?",
        default=None,
        help="Ticker symbol to analyze (e.g. TSLA)",
    )
    parser.add_argument("--ticker", dest="ticker_flag", default=None, help="Ticker symbol (alternative flag)")
    args = parser.parse_args()

    ticker = args.ticker or args.ticker_flag
    if not ticker:
        parser.error("Provide a ticker symbol, e.g.: python -m lexdrift.nlp.intelligence TSLA")

    session = _get_sync_session()
    try:
        report = generate_intelligence(session, ticker)
        _print_report(report)
    finally:
        session.close()
