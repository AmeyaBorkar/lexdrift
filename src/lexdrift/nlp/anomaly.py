"""Company-specific anomaly detection for drift scores.

A static threshold (e.g., 0.15) is nearly useless across 8,000+ companies
because different companies have wildly different baseline change rates.
A tech startup rewrites half its risk factors every quarter (normal).
A utility company barely changes a word (so any change is alarming).

This module computes:
1. Company-specific z-scores: how unusual is this drift relative to THIS company's history?
2. Sector-relative z-scores: how unusual is this drift relative to industry peers?
3. Combined anomaly score that accounts for both.
"""

import math
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """Anomaly assessment for a single drift score."""
    company_z_score: float | None  # stddev from company's own mean
    sector_z_score: float | None   # stddev from sector mean
    is_anomalous: bool             # exceeds threshold
    anomaly_level: str             # normal, elevated, high, extreme
    company_mean: float | None
    company_stddev: float | None
    sector_mean: float | None
    sector_stddev: float | None


def compute_z_score(value: float, mean: float, stddev: float) -> float | None:
    """Compute z-score. Returns None if stddev is too small."""
    if stddev < 0.001:
        return None
    return (value - mean) / stddev


def detect_anomaly(
    current_drift: float,
    company_history: list[float],
    sector_history: list[float] | None = None,
    company_z_threshold: float = 2.0,
    sector_z_threshold: float = 2.5,
) -> AnomalyResult:
    """Detect whether a drift score is anomalous for this company.

    Args:
        current_drift: The drift score to evaluate.
        company_history: All prior drift scores for this company (same section type).
        sector_history: Drift scores from peer companies (same sector + section type).
        company_z_threshold: Z-score threshold for company-level anomaly.
        sector_z_threshold: Z-score threshold for sector-level anomaly.
    """
    # Company-specific baseline
    company_z = None
    company_mean = None
    company_stddev = None
    if len(company_history) >= 3:
        company_mean = sum(company_history) / len(company_history)
        company_stddev = math.sqrt(
            sum((x - company_mean) ** 2 for x in company_history) / len(company_history)
        )
        company_z = compute_z_score(current_drift, company_mean, company_stddev)

    # Sector-relative baseline
    sector_z = None
    sector_mean = None
    sector_stddev = None
    if sector_history and len(sector_history) >= 5:
        sector_mean = sum(sector_history) / len(sector_history)
        sector_stddev = math.sqrt(
            sum((x - sector_mean) ** 2 for x in sector_history) / len(sector_history)
        )
        sector_z = compute_z_score(current_drift, sector_mean, sector_stddev)

    # Determine anomaly level
    is_anomalous = False
    anomaly_level = "normal"

    if company_z is not None:
        if company_z > company_z_threshold * 1.5:
            anomaly_level = "extreme"
            is_anomalous = True
        elif company_z > company_z_threshold:
            anomaly_level = "high"
            is_anomalous = True
        elif company_z > company_z_threshold * 0.7:
            anomaly_level = "elevated"

    # Sector comparison provides additional signal
    if sector_z is not None and sector_z > sector_z_threshold:
        # Company's drift is unusual even for its sector
        if anomaly_level == "normal":
            anomaly_level = "elevated"
        elif anomaly_level == "elevated":
            anomaly_level = "high"
            is_anomalous = True

    # If we don't have enough history, fall back to absolute check
    if company_z is None and sector_z is None:
        if current_drift > 0.3:
            anomaly_level = "high"
            is_anomalous = True
        elif current_drift > 0.2:
            anomaly_level = "elevated"

    return AnomalyResult(
        company_z_score=round(company_z, 4) if company_z is not None else None,
        sector_z_score=round(sector_z, 4) if sector_z is not None else None,
        is_anomalous=is_anomalous,
        anomaly_level=anomaly_level,
        company_mean=round(company_mean, 4) if company_mean is not None else None,
        company_stddev=round(company_stddev, 4) if company_stddev is not None else None,
        sector_mean=round(sector_mean, 4) if sector_mean is not None else None,
        sector_stddev=round(sector_stddev, 4) if sector_stddev is not None else None,
    )


def detect_trends(
    drift_history: list[float],
    sentiment_history: list[dict[str, float]] | None = None,
    min_periods: int = 4,
) -> dict:
    """Detect multi-period trends in drift and sentiment.

    Looks for:
    - Drift acceleration: are filings changing more and more each quarter?
    - Sentiment deterioration: is negativity/uncertainty trending upward?
    - Volatility shift: is the company's change pattern becoming erratic?
    """
    if len(drift_history) < min_periods:
        return {"has_trend": False, "reason": "insufficient_history"}

    recent = drift_history[-min_periods:]
    signals = []

    # Drift acceleration: is each period's drift higher than the last?
    increasing = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
    if increasing:
        signals.append({
            "type": "drift_acceleration",
            "description": f"Drift score has increased for {min_periods} consecutive periods",
            "severity": "high",
        })

    # Drift spike: most recent is much higher than the preceding average
    if len(recent) >= 3:
        preceding_avg = sum(recent[:-1]) / len(recent[:-1])
        if preceding_avg > 0 and recent[-1] > preceding_avg * 2:
            signals.append({
                "type": "drift_spike",
                "description": f"Latest drift ({recent[-1]:.4f}) is {recent[-1]/preceding_avg:.1f}x the recent average ({preceding_avg:.4f})",
                "severity": "high",
            })

    # Elevated baseline: recent average is notably higher than early history
    if len(drift_history) >= 8:
        early = drift_history[:4]
        late = drift_history[-4:]
        early_avg = sum(early) / len(early)
        late_avg = sum(late) / len(late)
        if early_avg > 0 and late_avg > early_avg * 1.5:
            signals.append({
                "type": "elevated_baseline",
                "description": f"Recent average drift ({late_avg:.4f}) is {late_avg/early_avg:.1f}x the historical average ({early_avg:.4f})",
                "severity": "medium",
            })

    # Sentiment deterioration
    if sentiment_history and len(sentiment_history) >= min_periods:
        recent_neg = [s.get("negative", 0) for s in sentiment_history[-min_periods:]]
        neg_increasing = all(recent_neg[i] <= recent_neg[i + 1] for i in range(len(recent_neg) - 1))
        if neg_increasing and recent_neg[-1] > recent_neg[0]:
            signals.append({
                "type": "sentiment_deterioration",
                "description": f"Negative sentiment has increased for {min_periods} consecutive periods",
                "severity": "high",
            })

        recent_unc = [s.get("uncertainty", 0) for s in sentiment_history[-min_periods:]]
        unc_increasing = all(recent_unc[i] <= recent_unc[i + 1] for i in range(len(recent_unc) - 1))
        if unc_increasing and recent_unc[-1] > recent_unc[0]:
            signals.append({
                "type": "uncertainty_increasing",
                "description": f"Uncertainty language has increased for {min_periods} consecutive periods",
                "severity": "medium",
            })

    return {
        "has_trend": len(signals) > 0,
        "signals": signals,
        "periods_analyzed": len(drift_history),
    }
