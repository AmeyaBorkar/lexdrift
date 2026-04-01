"""Sentence-level financial risk scoring.

Instead of treating all flagged changes equally, this module scores each
sentence for financial risk severity. A sentence about "material weakness
in internal controls" should trigger very differently than a sentence about
"updated our office lease agreement."

Three scoring layers:
1. Loughran-McDonald sentiment at the sentence level
2. High-severity keyword proximity (domain-specific danger signals)
3. Risk classification: critical / high / medium / low / boilerplate
"""

import re
from dataclasses import dataclass

from lexdrift.nlp.sentiment import score_sentiment

# Tiered keyword groups — ordered by severity.
# Presence of these terms near each other escalates the risk classification.
CRITICAL_TERMS = {
    "going concern", "restatement", "material weakness",
    "fraud", "sec investigation", "delisted", "default",
    "bankruptcy", "insolvency",
}

HIGH_RISK_TERMS = {
    "goodwill impairment", "asset impairment", "write-down", "write-off",
    "covenant breach", "covenant violation", "class action",
    "regulatory action", "securities litigation", "shareholder lawsuit",
    "discontinued operation", "qualified opinion", "adverse opinion",
    "significant deficiency", "internal control",
}

MEDIUM_RISK_TERMS = {
    "restructuring", "workforce reduction", "layoff", "severance",
    "supply chain disruption", "liquidity risk", "credit risk",
    "cybersecurity incident", "data breach", "force majeure",
    "impairment", "provision", "contingent liability",
    "decline in revenue", "decline in sales", "operating loss",
    "net loss", "negative cash flow", "going forward",
}

# Terms that strongly indicate boilerplate / non-risk
BOILERPLATE_SIGNALS = {
    "accounting standard", "fasb", "asu", "adoption of",
    "effective for fiscal years", "forward-looking statements",
    "safe harbor", "private securities litigation",
    "pursuant to", "in accordance with", "as required by",
}


@dataclass
class RiskScore:
    """Risk assessment for a single sentence or text fragment."""
    level: str  # critical, high, medium, low, boilerplate
    score: float  # 0.0 to 1.0
    triggers: list[str]  # which terms/signals triggered this classification
    sentiment: dict[str, float]  # Loughran-McDonald scores for this sentence


def _find_terms(text_lower: str, term_set: set[str]) -> list[str]:
    """Find all terms from a set that appear in the text."""
    return [term for term in term_set if term in text_lower]


def score_sentence_risk(sentence: str) -> RiskScore:
    """Score a single sentence for financial risk severity.

    Returns a RiskScore with level, numeric score, trigger terms, and sentiment.
    """
    text_lower = sentence.lower()
    triggers: list[str] = []

    # Check term tiers
    critical_hits = _find_terms(text_lower, CRITICAL_TERMS)
    high_hits = _find_terms(text_lower, HIGH_RISK_TERMS)
    medium_hits = _find_terms(text_lower, MEDIUM_RISK_TERMS)
    boilerplate_hits = _find_terms(text_lower, BOILERPLATE_SIGNALS)

    # Sentence-level sentiment
    sentiment = score_sentiment(sentence)

    # Classification logic
    if boilerplate_hits and not critical_hits and not high_hits:
        return RiskScore(
            level="boilerplate", score=0.05,
            triggers=boilerplate_hits, sentiment=sentiment,
        )

    if critical_hits:
        triggers = critical_hits + high_hits
        return RiskScore(
            level="critical", score=0.95,
            triggers=triggers, sentiment=sentiment,
        )

    if high_hits:
        triggers = high_hits
        # Boost if sentiment is also strongly negative
        base = 0.7
        if sentiment.get("negative", 0) > 0.05:
            base = 0.8
        return RiskScore(
            level="high", score=base,
            triggers=triggers, sentiment=sentiment,
        )

    if medium_hits:
        triggers = medium_hits
        base = 0.45
        if sentiment.get("negative", 0) > 0.05:
            base = 0.55
        return RiskScore(
            level="medium", score=base,
            triggers=triggers, sentiment=sentiment,
        )

    # No keyword hits — classify purely on sentiment density
    neg = sentiment.get("negative", 0)
    unc = sentiment.get("uncertainty", 0)
    lit = sentiment.get("litigious", 0)
    combined_risk = neg + unc * 0.5 + lit * 0.5

    if combined_risk > 0.08:
        return RiskScore(
            level="medium", score=0.35,
            triggers=["elevated_negative_sentiment"], sentiment=sentiment,
        )

    return RiskScore(
        level="low", score=0.1,
        triggers=[], sentiment=sentiment,
    )


def score_changes(sentence_changes: dict) -> dict:
    """Score all flagged sentence changes for financial risk.

    Takes the output of sentences.compare_sentences() and enriches
    each entry with a risk score.

    Returns the same structure with risk_score added to each entry.
    """
    scored_added = []
    for entry in sentence_changes.get("added", []):
        risk = score_sentence_risk(entry["text"])
        scored_added.append({
            **entry,
            "risk": {"level": risk.level, "score": risk.score, "triggers": risk.triggers},
        })

    scored_removed = []
    for entry in sentence_changes.get("removed", []):
        risk = score_sentence_risk(entry["text"])
        scored_removed.append({
            **entry,
            "risk": {"level": risk.level, "score": risk.score, "triggers": risk.triggers},
        })

    scored_changed = []
    for entry in sentence_changes.get("changed", []):
        risk_prev = score_sentence_risk(entry["prev_text"])
        risk_curr = score_sentence_risk(entry["curr_text"])
        scored_changed.append({
            **entry,
            "risk_prev": {"level": risk_prev.level, "score": risk_prev.score, "triggers": risk_prev.triggers},
            "risk_curr": {"level": risk_curr.level, "score": risk_curr.score, "triggers": risk_curr.triggers},
        })

    scored_replacements = []
    for entry in sentence_changes.get("likely_replacements", []):
        risk_prev = score_sentence_risk(entry["prev_text"])
        risk_curr = score_sentence_risk(entry["curr_text"])
        scored_replacements.append({
            **entry,
            "risk_prev": {"level": risk_prev.level, "score": risk_prev.score, "triggers": risk_prev.triggers},
            "risk_curr": {"level": risk_curr.level, "score": risk_curr.score, "triggers": risk_curr.triggers},
        })

    # Sort each list by risk score descending — most dangerous changes first
    scored_added.sort(key=lambda x: x["risk"]["score"], reverse=True)
    scored_removed.sort(key=lambda x: x["risk"]["score"], reverse=True)

    # Compute aggregate risk summary
    all_risk_scores = (
        [e["risk"]["score"] for e in scored_added]
        + [e["risk"]["score"] for e in scored_removed]
        + [e.get("risk_curr", {}).get("score", 0) for e in scored_changed]
    )
    max_risk = max(all_risk_scores) if all_risk_scores else 0.0
    critical_count = sum(1 for s in all_risk_scores if s >= 0.9)
    high_count = sum(1 for s in all_risk_scores if 0.6 <= s < 0.9)

    return {
        "added": scored_added,
        "removed": scored_removed,
        "changed": scored_changed,
        "likely_replacements": scored_replacements,
        "unchanged_count": sentence_changes.get("unchanged_count", 0),
        "stats": sentence_changes.get("stats", {}),
        "risk_summary": {
            "max_risk_score": round(max_risk, 4),
            "max_risk_level": (
                "critical" if max_risk >= 0.9 else
                "high" if max_risk >= 0.6 else
                "medium" if max_risk >= 0.3 else "low"
            ),
            "critical_changes": critical_count,
            "high_risk_changes": high_count,
        },
    }
