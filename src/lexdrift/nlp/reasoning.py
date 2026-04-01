"""LLM-powered reasoning layer for LexDrift.

Replaces template-based narratives with actual reasoning. Uses Groq's
free API (Llama 3.3 70B) to synthesize all signals into genuine
analyst-quality intelligence with causal reasoning, not f-string templates.

The LLM receives structured signal data and is prompted to reason like
a senior financial analyst — connecting signals, identifying causation,
assessing materiality, and making specific predictions grounded in the data.

Falls back to template-based narrative (narrative.py) if the API key
is not configured or the call fails.
"""

import json
import logging
from dataclasses import asdict

from lexdrift.config import settings

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a senior SEC filing analyst at a top-tier investment bank. You analyze changes in public company filings to identify material risks, strategic shifts, and early warning signals.

You will receive structured data about a company's recent SEC filing changes. Your job is to synthesize this into a clear, actionable intelligence brief.

Rules:
- Write like a Bloomberg analyst note — precise, evidence-based, no hedging language
- Every claim must reference specific data from the signals provided
- Identify CAUSAL relationships: "The company added supply chain risk language BECAUSE..."
- Assess MATERIALITY: is this a routine update or a genuine warning sign?
- Make SPECIFIC predictions grounded in the data pattern, not generic warnings
- Compare to historical patterns when provided
- Be direct. If the data shows trouble, say so. If it's routine, say that.
- Do NOT use phrases like "AI analysis", "our system detected", "the algorithm found"
- DO use phrases like "the filing reveals", "notably", "of particular concern", "consistent with"
- Keep it under 500 words
- Structure: Executive Summary (2 sentences) → Key Findings → Risk Assessment → Outlook"""

_USER_PROMPT_TEMPLATE = """Analyze the following SEC filing intelligence data for {ticker} ({company_name}):

## Signal Summary
- Risk Score: {risk_score:.2f}/1.00 (Level: {risk_level})
- Drift Phase: {drift_phase}
- Drift Velocity: {drift_velocity}
- Drift Acceleration: {drift_acceleration}
- Sentiment Trend: {sentiment_trend}
- Anomaly Level: {anomaly_level}
- Obfuscation Score: {obfuscation_score}
- Entropy Novelty: {entropy_novelty}

## Key Phrase Changes
New phrases in risk factors: {new_phrases}
Removed phrases: {removed_phrases}

## Critical Sentence Changes
{critical_changes}

## Findings from Signal Analysis
{findings}

## Historical Pattern Matches
{patterns}

## Comparable Companies
{comparables}

Generate a comprehensive intelligence brief for this company. Be specific — reference the actual numbers, phrases, and patterns above."""


def _format_findings(findings: list) -> str:
    if not findings:
        return "No significant findings."
    lines = []
    for f in findings[:8]:
        if isinstance(f, dict):
            lines.append(f"- [{f.get('severity', '?')}] {f.get('title', '')}: {f.get('detail', '')}")
        else:
            lines.append(f"- [{f.severity}] {f.title}: {f.detail}")
    return "\n".join(lines)


def _format_patterns(patterns: list) -> str:
    if not patterns:
        return "No historical patterns matched."
    lines = []
    for p in patterns[:5]:
        if isinstance(p, dict):
            lines.append(f"- {p.get('pattern_name', '')}: {p.get('description', '')} (match: {p.get('match_score', 0):.0%})")
        else:
            lines.append(f"- {p.pattern_name}: {p.description} (match: {p.match_score:.0%})")
    return "\n".join(lines)


def _format_comparables(comparables: list) -> str:
    if not comparables:
        return "No comparable companies identified."
    lines = []
    for c in comparables[:5]:
        if isinstance(c, dict):
            lines.append(f"- {c.get('ticker', '?')} ({c.get('company_name', '?')}): {c.get('outcome', '?')}")
        else:
            lines.append(f"- {c.ticker} ({c.company_name}): {c.outcome}")
    return "\n".join(lines)


def _format_critical_changes(sentence_changes: int, new_phrases: list) -> str:
    parts = []
    if sentence_changes > 0:
        parts.append(f"{sentence_changes} critical-risk sentences added/changed")
    if not parts:
        return "No critical sentence-level changes detected."
    return "; ".join(parts)


def reason_about_company(intelligence_data: dict) -> str:
    """Use LLM to generate a reasoned intelligence narrative.

    Args:
        intelligence_data: Dict from CompanyIntelligence (via dataclasses.asdict)

    Returns:
        LLM-generated analyst narrative, or template fallback on failure.
    """
    if not settings.groq_api_key or not settings.llm_enabled:
        logger.debug("LLM disabled or no API key — falling back to template narrative")
        return _template_fallback(intelligence_data)

    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)

        signals = intelligence_data.get("signals", {})
        findings = intelligence_data.get("findings", [])
        patterns = intelligence_data.get("patterns", [])
        comparables = intelligence_data.get("comparables", [])

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            ticker=intelligence_data.get("ticker", "?"),
            company_name=intelligence_data.get("company_name", "?"),
            risk_score=intelligence_data.get("risk_score", 0),
            risk_level=intelligence_data.get("risk_level", "unknown"),
            drift_phase=signals.get("drift_phase", "unknown"),
            drift_velocity=signals.get("drift_velocity", "N/A"),
            drift_acceleration=signals.get("drift_acceleration", "N/A"),
            sentiment_trend=signals.get("sentiment_trend", "unknown"),
            anomaly_level=signals.get("anomaly_level", "unknown"),
            obfuscation_score=signals.get("obfuscation_score", "N/A"),
            entropy_novelty=signals.get("entropy_novelty", "N/A"),
            new_phrases=", ".join(signals.get("new_risk_phrases", [])[:10]) or "None",
            removed_phrases=", ".join(signals.get("removed_risk_phrases", [])[:10]) or "None",
            critical_changes=_format_critical_changes(
                signals.get("critical_sentence_changes", 0),
                signals.get("new_risk_phrases", []),
            ),
            findings=_format_findings(findings),
            patterns=_format_patterns(patterns),
            comparables=_format_comparables(comparables),
        )

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        narrative = response.choices[0].message.content.strip()
        logger.info("LLM narrative generated for %s (%d chars)", intelligence_data.get("ticker"), len(narrative))
        return narrative

    except Exception:
        logger.warning("LLM reasoning failed — falling back to template narrative", exc_info=True)
        return _template_fallback(intelligence_data)


def reason_about_market(market_data: dict) -> str:
    """Use LLM to generate a market-level intelligence narrative."""
    if not settings.groq_api_key or not settings.llm_enabled:
        return _market_template_fallback(market_data)

    try:
        from groq import Groq

        client = Groq(api_key=settings.groq_api_key)

        prompt = f"""Analyze the following market-wide SEC filing intelligence:

## Overall Market Drift
- Average drift level: {market_data.get('overall_drift_level', 'unknown')}
- Date: {market_data.get('date', 'unknown')}

## Sector Trends
{json.dumps(market_data.get('sector_trends', []), indent=2, default=str)[:2000]}

## Risk Propagation Signals
{json.dumps(market_data.get('risk_propagations', []), indent=2, default=str)[:2000]}

## Divergent Companies
{json.dumps(market_data.get('divergent_companies', []), indent=2, default=str)[:1000]}

Generate a market intelligence brief. Focus on systemic risks, sector-wide trends, and companies that stand out. Be specific."""

        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )

        return response.choices[0].message.content.strip()

    except Exception:
        logger.warning("LLM market reasoning failed — falling back to template", exc_info=True)
        return _market_template_fallback(market_data)


# ---------------------------------------------------------------------------
# Template fallbacks (when LLM is unavailable)
# ---------------------------------------------------------------------------

def _template_fallback(data: dict) -> str:
    """Generate a basic narrative without LLM — structured but not reasoned."""
    ticker = data.get("ticker", "?")
    risk_level = data.get("risk_level", "unknown")
    risk_score = data.get("risk_score", 0)
    signals = data.get("signals", {})
    findings = data.get("findings", [])

    parts = [f"## {ticker} — Risk Level: {risk_level.upper()} ({risk_score:.2f}/1.00)\n"]

    if findings:
        parts.append("### Key Findings\n")
        for f in findings[:5]:
            sev = f.get("severity", "?") if isinstance(f, dict) else f.severity
            title = f.get("title", "") if isinstance(f, dict) else f.title
            detail = f.get("detail", "") if isinstance(f, dict) else f.detail
            parts.append(f"- **[{sev}]** {title}: {detail}")

    phase = signals.get("drift_phase", "unknown")
    vel = signals.get("drift_velocity")
    parts.append(f"\n### Trend\nDrift phase: {phase}. Velocity: {vel}.")

    new_p = signals.get("new_risk_phrases", [])
    if new_p:
        parts.append(f"\n### New Risk Language\n{', '.join(new_p[:10])}")

    actions = data.get("actions", [])
    if actions:
        parts.append("\n### Recommended Actions")
        for a in actions:
            parts.append(f"- {a}")

    return "\n".join(parts)


def _market_template_fallback(data: dict) -> str:
    """Basic market narrative without LLM."""
    level = data.get("overall_drift_level", "unknown")
    trends = data.get("sector_trends", [])
    divergent = data.get("divergent_companies", [])

    parts = [f"## Market Filing Intelligence\n\nOverall drift level: {level}.\n"]

    if trends:
        parts.append(f"### Sector Trends\n{len(trends)} sector-wide trends detected.")

    if divergent:
        tickers = [d.get("ticker", "?") if isinstance(d, dict) else str(d) for d in divergent[:5]]
        parts.append(f"\n### Divergent Filers\n{', '.join(tickers)}")

    return "\n".join(parts)
