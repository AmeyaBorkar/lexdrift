"""Natural-language report generator — template-based analyst narratives.

Produces Bloomberg-style analyst notes from structured data without any
LLM dependency. Templates use enough variation to read naturally while
remaining deterministic and reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from lexdrift.nlp.cross_filing import MarketIntelligence


# ---------------------------------------------------------------------------
# CompanyIntelligence dataclass
# ---------------------------------------------------------------------------

@dataclass
class CompanyIntelligence:
    """Structured intelligence for a single company."""
    ticker: str
    company_name: str
    risk_level: str  # critical, high, medium, low
    risk_score: float  # 0.0-1.0
    findings: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    signals: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vocabulary pools for natural variation
# ---------------------------------------------------------------------------

_TRANSITION_WORDS = [
    "Notably", "Of particular note", "Significantly", "Worth highlighting",
    "Importantly", "Of concern",
]

_TREND_OPENERS = [
    "Consistent with historical patterns",
    "Based on historical filing data",
    "Historical analysis suggests",
    "Filing history indicates",
]

_RISK_DESCRIPTORS = {
    "critical": ["warrants immediate attention", "represents a material concern",
                 "should be flagged for urgent review"],
    "high": ["merits close monitoring", "suggests elevated risk",
             "indicates significant disclosure changes"],
    "medium": ["bears watching", "reflects moderate revision activity",
               "suggests evolving risk language"],
    "low": ["appears within normal parameters", "reflects routine updates",
            "shows typical filing variation"],
}


def _pick(items: list[str], seed: str = "") -> str:
    """Deterministic-ish pick from a list seeded by a string."""
    idx = hash(seed) % len(items)
    return items[idx]


# ---------------------------------------------------------------------------
# generate_company_narrative
# ---------------------------------------------------------------------------

def generate_company_narrative(intelligence: CompanyIntelligence) -> str:
    """Generate a multi-paragraph analyst narrative for a company.

    Returns a report formatted as a plain-text analyst note covering:
    executive summary, risk assessment, key findings, trend analysis,
    and recommended actions.
    """
    ticker = intelligence.ticker
    name = intelligence.company_name
    sections: list[str] = []

    # ----- Executive Summary -----
    risk_desc = _pick(
        _RISK_DESCRIPTORS.get(intelligence.risk_level, _RISK_DESCRIPTORS["low"]),
        seed=ticker,
    )
    summary = (
        f"EXECUTIVE SUMMARY\n\n"
        f"{name} ({ticker}) — The latest filing analysis yields a "
        f"{intelligence.risk_level.upper()} risk designation "
        f"(score: {intelligence.risk_score:.2f}). "
        f"This {risk_desc}."
    )
    sections.append(summary)

    # ----- Risk Assessment -----
    risk_lines = [f"RISK ASSESSMENT\n\nOverall risk level: {intelligence.risk_level.upper()}"]

    if intelligence.risk_level in ("critical", "high"):
        risk_lines.append(
            f"{ticker}'s disclosure activity is elevated relative to its filing "
            f"history. The combination of risk indicators places it in the top "
            f"tier of monitored companies requiring analyst attention."
        )
    elif intelligence.risk_level == "medium":
        risk_lines.append(
            f"{ticker}'s filing shows moderate changes that, while not immediately "
            f"alarming, deviate from the company's baseline disclosure patterns."
        )
    else:
        risk_lines.append(
            f"{ticker}'s latest filing shows minimal divergence from prior periods. "
            f"No material changes in risk language were identified."
        )

    sections.append("\n".join(risk_lines))

    # ----- Key Findings -----
    if intelligence.findings:
        findings_lines = ["KEY FINDINGS\n"]
        for i, finding in enumerate(intelligence.findings, 1):
            transition = _pick(_TRANSITION_WORDS, seed=f"{ticker}-{i}")
            desc = finding.get("description", "")
            value = finding.get("value")
            if value is not None:
                findings_lines.append(
                    f"{transition}, {desc.lower() if desc else 'a notable change was detected'}"
                    f" (measured at {value})."
                )
            else:
                findings_lines.append(
                    f"{transition}, {desc.lower() if desc else 'a notable change was detected'}."
                )
        sections.append("\n".join(findings_lines))
    else:
        sections.append(
            "KEY FINDINGS\n\nNo significant individual findings flagged in this "
            "filing period."
        )

    # ----- Trend Analysis -----
    trend_lines = ["TREND ANALYSIS\n"]
    if intelligence.predictions:
        opener = _pick(_TREND_OPENERS, seed=ticker)
        trend_lines.append(f"{opener}:")
        for pred in intelligence.predictions:
            desc = pred.get("description", "")
            confidence = pred.get("confidence")
            if confidence is not None:
                trend_lines.append(
                    f"  - {desc} (confidence: {confidence:.0%})"
                )
            else:
                trend_lines.append(f"  - {desc}")
    else:
        trend_lines.append(
            "Insufficient historical data to project forward trends at this time."
        )
    sections.append("\n".join(trend_lines))

    # ----- Signals -----
    if intelligence.signals:
        signal_lines = ["CROSS-FILING SIGNALS\n"]
        for sig in intelligence.signals:
            sig_type = sig.get("signal_type", "")
            title = sig.get("title", "")
            desc = sig.get("description", "")
            if sig_type == "risk_propagation":
                signal_lines.append(
                    f"Risk propagation detected — {title}. {desc}"
                )
            elif sig_type == "divergence":
                signal_lines.append(
                    f"Divergence alert — {title}. {desc}"
                )
            else:
                signal_lines.append(f"{title}. {desc}")
        sections.append("\n".join(signal_lines))

    # ----- Recommended Actions -----
    if intelligence.actions:
        action_lines = ["RECOMMENDED ACTIONS\n"]
        for j, action in enumerate(intelligence.actions, 1):
            action_lines.append(f"  {j}. {action}")
        sections.append("\n".join(action_lines))
    else:
        default_actions = {
            "critical": [
                f"Escalate {ticker} for immediate senior analyst review.",
                "Cross-reference with recent 8-K filings for undisclosed events.",
                "Review upcoming earnings call for management commentary on flagged changes.",
            ],
            "high": [
                f"Schedule detailed review of {ticker}'s changed risk factors.",
                "Compare flagged language against peer filings in the same sector.",
            ],
            "medium": [
                f"Add {ticker} to watchlist for the next filing cycle.",
                "Monitor for continuation of observed trends.",
            ],
            "low": [
                "No immediate action required.",
                f"Continue routine monitoring of {ticker}.",
            ],
        }
        action_lines = ["RECOMMENDED ACTIONS\n"]
        for j, action in enumerate(
            default_actions.get(intelligence.risk_level, default_actions["low"]), 1
        ):
            action_lines.append(f"  {j}. {action}")
        sections.append("\n".join(action_lines))

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# generate_market_narrative
# ---------------------------------------------------------------------------

def generate_market_narrative(market_intel: MarketIntelligence) -> str:
    """Generate a market-level analyst narrative from MarketIntelligence."""

    sections: list[str] = []

    # ----- Market Overview -----
    drift_level = market_intel.overall_market_drift_level
    if drift_level > 0.15:
        tone = "elevated"
        commentary = (
            "Disclosure revision activity across the filing universe is running "
            "well above historical norms, suggesting broad-based changes in "
            "corporate risk communication."
        )
    elif drift_level > 0.08:
        tone = "moderate"
        commentary = (
            "Market-wide disclosure drift is within a normal-to-slightly-elevated "
            "range. Selected sectors show higher activity."
        )
    else:
        tone = "subdued"
        commentary = (
            "Overall disclosure drift remains low across the monitored filing "
            "universe. Most companies are making only routine revisions."
        )

    overview = (
        f"MARKET INTELLIGENCE REPORT — {market_intel.date}\n\n"
        f"MARKET OVERVIEW\n\n"
        f"Average market drift level: {drift_level:.4f} ({tone}). "
        f"{commentary}"
    )
    sections.append(overview)

    # ----- Sector Trends -----
    if market_intel.sector_trends:
        trend_lines = [
            f"SECTOR TRENDS ({len(market_intel.sector_trends)} detected)\n"
        ]
        for trend in market_intel.sector_trends[:5]:  # Top 5
            companies = ", ".join(trend.companies_involved[:5])
            if len(trend.companies_involved) > 5:
                companies += f" and {len(trend.companies_involved) - 5} others"
            trend_lines.append(
                f"  - {trend.title}: {trend.description} "
                f"Companies: {companies}. "
                f"Significance: {trend.significance:.2f}."
            )
        sections.append("\n".join(trend_lines))
    else:
        sections.append(
            "SECTOR TRENDS\n\nNo coordinated sector-level drift patterns "
            "detected in the current period."
        )

    # ----- Risk Propagation -----
    if market_intel.risk_propagations:
        prop_lines = [
            f"RISK LANGUAGE PROPAGATION ({len(market_intel.risk_propagations)} signals)\n"
        ]
        for prop in market_intel.risk_propagations[:5]:
            prop_lines.append(
                f"  - {prop.title}: {prop.description} "
                f"Lag: {prop.propagation_lag} quarter(s). "
                f"Significance: {prop.significance:.2f}."
            )
        sections.append("\n".join(prop_lines))
    else:
        sections.append(
            "RISK LANGUAGE PROPAGATION\n\nNo significant cross-company "
            "risk language propagation detected."
        )

    # ----- Divergent Companies -----
    if market_intel.divergent_companies:
        div_lines = [
            f"DIVERGENT FILERS ({len(market_intel.divergent_companies)} flagged)\n"
        ]
        for div in market_intel.divergent_companies:
            div_lines.append(f"  - {div.title}: {div.description}")
        sections.append("\n".join(div_lines))
    else:
        sections.append(
            "DIVERGENT FILERS\n\nAll monitored companies are within "
            "normal drift ranges relative to their peer groups."
        )

    return "\n\n".join(sections)
