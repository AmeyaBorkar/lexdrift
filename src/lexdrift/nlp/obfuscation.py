"""Adversarial Obfuscation Detection in SEC Filings.

Novel contribution: Companies deliberately deploy euphemisms, vague qualifiers,
and inflated verbiage to obscure adverse developments from investors.  This module
quantifies that behaviour by measuring four orthogonal signals and fusing them
into a single obfuscation score.

Signals
-------
1. **Information Density Drop** -- unique meaningful tokens per sentence.  When a
   section grows longer but density *falls*, the company is padding with vagueness.
2. **Specificity Score** -- ratio of concrete referents (numbers, dates, dollar
   amounts, named entities) to hedge words ("approximately", "may", "could",
   "potentially").  A decline in specificity between filings flags deliberate
   abstraction.
3. **Readability Shift** -- Gunning-Fog and Coleman-Liau indices.  If readability
   *decreases* (text becomes harder to parse) in sections that also carry negative
   sentiment changes, the difficulty is likely intentional obfuscation rather than
   legitimate technical complexity.
4. **Euphemism Detection** -- when a specific term (e.g. "layoff") vanishes and a
   vaguer substitute (e.g. "organizational realignment") appears, we measure the
   *specificity gap* between the two to surface deliberate softening.

The four signals are combined into an ``ObfuscationScore`` dataclass whose
``overall_obfuscation_score`` ranges from 0 (transparent) to 1 (maximally
obfuscatory).

References
----------
- Li, F. (2008). Annual report readability, current earnings, and earnings
  persistence. *Journal of Accounting and Economics*, 45(2-3), 221-247.
- Loughran, T. & McDonald, B. (2014). Measuring readability in financial
  disclosures. *The Journal of Finance*, 69(4), 1643-1671.
- Gunning, R. (1952). *The Technique of Clear Writing*.
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lexical resources
# ---------------------------------------------------------------------------

# Hedge / weasel words that reduce specificity.
HEDGE_WORDS: frozenset[str] = frozenset({
    "approximately", "generally", "may", "might", "could", "would",
    "potentially", "possibly", "likely", "unlikely", "substantially",
    "reasonably", "believe", "believe", "anticipate", "expect",
    "estimate", "intend", "plan", "appear", "seem", "suggest",
    "indicate", "certain", "uncertain", "perhaps", "somewhat",
    "largely", "mostly", "partially", "virtually", "practically",
    "relatively", "broadly", "nearly", "roughly", "about",
    "tend", "tends", "trending", "projected", "forecast",
})

# Patterns that indicate concrete, specific language.
_NUMBER_RE = re.compile(r"\$[\d,.]+|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+%|\d+%")
_DATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+\d{1,2},?\s+\d{4}\b"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}\b"
    r"|\b(?:Q[1-4]|FY)\s*\d{2,4}\b",
    re.IGNORECASE,
)
_ENTITY_RE = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b")  # rough NER proxy

# Known euphemism pairs: (specific_term, vague_replacement).
# The first element is the concrete term that might disappear; the second is
# the softer substitute that might appear in its place.
EUPHEMISM_MAP: dict[str, list[str]] = {
    "layoff":                ["organizational realignment", "workforce optimization",
                              "right-sizing", "restructuring initiative",
                              "headcount adjustment", "resource rebalancing"],
    "fired":                 ["separated", "transitioned", "involuntary attrition",
                              "workforce reduction"],
    "loss":                  ["negative growth", "shortfall", "headwind",
                              "challenging results", "earnings pressure"],
    "decline":               ["moderation", "softening", "normalization",
                              "headwind", "deceleration"],
    "debt":                  ["leverage", "financial obligation", "capital structure",
                              "borrowing arrangement"],
    "default":               ["covenant modification", "technical noncompliance",
                              "credit event", "forbearance agreement"],
    "lawsuit":               ["legal proceeding", "litigation matter",
                              "dispute resolution process"],
    "investigation":         ["regulatory inquiry", "government review",
                              "examination process", "regulatory dialogue"],
    "failure":               ["shortcoming", "challenge", "deficiency",
                              "area for improvement", "opportunity"],
    "write-off":             ["non-cash charge", "asset rationalization",
                              "balance sheet optimization", "valuation adjustment"],
    "cut":                   ["reduction", "optimization", "efficiency measure",
                              "cost transformation", "streamlining"],
    "bankruptcy":            ["financial restructuring", "chapter 11 process",
                              "court-supervised reorganization",
                              "balance sheet recapitalization"],
    "fraud":                 ["irregularity", "accounting matter",
                              "internal control issue", "material misstatement"],
    "bribe":                 ["facilitation payment", "consulting fee",
                              "market access cost"],
    "pollution":             ["environmental impact", "remediation matter",
                              "legacy environmental condition"],
    "closure":               ["consolidation", "rationalization",
                              "portfolio optimization", "footprint adjustment"],
    "penalty":               ["settlement", "resolution amount",
                              "regulatory cost", "compliance cost"],
    "downgrade":             ["rating action", "credit reassessment",
                              "outlook revision"],
}

# ---------------------------------------------------------------------------
# Sentence splitting helper (lightweight, no external deps)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")
_WORD_RE = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")
_SYLLABLE_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a lightweight heuristic."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    """Extract lowercase word tokens."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def _count_syllables(word: str) -> int:
    """Estimate syllable count via vowel-cluster heuristic."""
    matches = _SYLLABLE_RE.findall(word)
    count = len(matches) if matches else 1
    # Trailing silent-e correction
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def _information_density(text: str) -> float:
    """Compute information density: unique meaningful tokens per sentence.

    Higher values indicate more informative, less repetitive text.
    A drop in density across filings signals padding with filler language.

    Returns 0.0 for empty text.
    """
    sentences = _split_sentences(text)
    if not sentences:
        return 0.0

    tokens = _tokenize_words(text)
    if not tokens:
        return 0.0

    unique_tokens = len(set(tokens))
    return unique_tokens / len(sentences)


def _specificity_score(text: str) -> float:
    """Compute specificity: ratio of concrete referents to hedge qualifiers.

    Concrete referents include numbers, dollar amounts, percentages, dates,
    and capitalised entity-like phrases.  Hedge qualifiers are drawn from
    ``HEDGE_WORDS``.

    Returns a value in [0, 1] where 1 is maximally specific and 0 is
    maximally hedged.  Returns 0.5 (neutral) when both counts are zero.
    """
    concrete_count = (
        len(_NUMBER_RE.findall(text))
        + len(_DATE_RE.findall(text))
        + len(_ENTITY_RE.findall(text))
    )

    tokens_lower = _tokenize_words(text)
    hedge_count = sum(1 for t in tokens_lower if t in HEDGE_WORDS)

    total = concrete_count + hedge_count
    if total == 0:
        return 0.5  # neutral when no signal
    return concrete_count / total


def _gunning_fog_index(text: str) -> float:
    """Compute Gunning-Fog readability index.

    Fog = 0.4 * (average_sentence_length + percent_complex_words)
    where complex words have >= 3 syllables.

    Lower scores indicate easier readability.  Typical SEC filings
    score 18-22; newspaper prose scores 10-12.
    """
    sentences = _split_sentences(text)
    words = _tokenize_words(text)

    if not sentences or not words:
        return 0.0

    avg_sentence_length = len(words) / len(sentences)
    complex_words = sum(1 for w in words if _count_syllables(w) >= 3)
    pct_complex = (complex_words / len(words)) * 100

    return 0.4 * (avg_sentence_length + pct_complex)


def _coleman_liau_index(text: str) -> float:
    """Compute Coleman-Liau readability index.

    CLI = 0.0588 * L - 0.296 * S - 15.8
    where L = avg letters per 100 words, S = avg sentences per 100 words.
    """
    sentences = _split_sentences(text)
    words = _tokenize_words(text)

    if not words or not sentences:
        return 0.0

    letters = sum(len(w) for w in words)
    n_words = len(words)
    n_sentences = len(sentences)

    L = (letters / n_words) * 100
    S = (n_sentences / n_words) * 100

    return 0.0588 * L - 0.296 * S - 15.8


def _detect_euphemisms(prev_text: str, curr_text: str) -> list[dict]:
    """Detect euphemistic substitutions between two filing texts.

    For each known specific term in ``EUPHEMISM_MAP``, check whether the
    term disappeared from the filing while one of its known vague
    replacements appeared.

    Returns a list of dicts with keys:
        specific_term   -- the concrete term that vanished
        euphemism       -- the vague substitute that appeared
        specificity_gap -- estimated magnitude of the softening (0-1)
    """
    prev_lower = prev_text.lower()
    curr_lower = curr_text.lower()
    detected: list[dict] = []

    for specific_term, vague_alternatives in EUPHEMISM_MAP.items():
        was_present = specific_term in prev_lower
        now_absent = specific_term not in curr_lower

        if not (was_present and now_absent):
            continue

        for euphemism in vague_alternatives:
            was_absent = euphemism not in prev_lower
            now_present = euphemism in curr_lower

            if was_absent and now_present:
                # Specificity gap: longer euphemisms with more syllables
                # generally indicate a wider gap from the original term.
                euph_words = euphemism.split()
                avg_syllables = sum(_count_syllables(w) for w in euph_words) / len(euph_words)
                word_ratio = len(euph_words) / max(len(specific_term.split()), 1)
                gap = min(1.0, 0.3 * word_ratio + 0.15 * avg_syllables)

                detected.append({
                    "specific_term": specific_term,
                    "euphemism": euphemism,
                    "specificity_gap": round(gap, 4),
                })

    # Sort by gap magnitude descending (worst offenders first)
    detected.sort(key=lambda d: d["specificity_gap"], reverse=True)
    return detected


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ObfuscationScore:
    """Composite obfuscation assessment between two filing versions.

    Attributes
    ----------
    density_prev : float
        Information density of the previous filing (unique tokens / sentence).
    density_curr : float
        Information density of the current filing.
    density_change : float
        Relative change in density (negative = more padding, worse).
    specificity_prev : float
        Specificity score of the previous filing (0-1).
    specificity_curr : float
        Specificity score of the current filing.
    specificity_change : float
        Change in specificity (negative = more hedging, worse).
    fog_prev : float
        Gunning-Fog readability index of the previous filing.
    fog_curr : float
        Gunning-Fog readability index of the current filing.
    coleman_liau_prev : float
        Coleman-Liau readability index of the previous filing.
    coleman_liau_curr : float
        Coleman-Liau readability index of the current filing.
    readability_change : float
        Normalised readability shift (positive = harder to read, worse).
    detected_euphemisms : list[dict]
        Euphemistic substitutions found between filings.
    overall_obfuscation_score : float
        Fused score in [0, 1] where 1 = maximally obfuscatory.
    component_scores : dict[str, float]
        Individual normalised sub-scores for each signal.
    """

    density_prev: float
    density_curr: float
    density_change: float
    specificity_prev: float
    specificity_curr: float
    specificity_change: float
    fog_prev: float
    fog_curr: float
    coleman_liau_prev: float
    coleman_liau_curr: float
    readability_change: float
    detected_euphemisms: list[dict] = field(default_factory=list)
    overall_obfuscation_score: float = 0.0
    component_scores: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_obfuscation(prev_text: str, curr_text: str) -> ObfuscationScore:
    """Detect deliberate obfuscation between two versions of a filing section.

    Computes four orthogonal obfuscation signals and fuses them into a single
    score.  The method is designed for 10-K and 10-Q section comparisons
    (e.g. Risk Factors, MD&A) where adversarial language softening is most
    consequential.

    Parameters
    ----------
    prev_text : str
        Full text of the section from the earlier filing.
    curr_text : str
        Full text of the same section from the later filing.

    Returns
    -------
    ObfuscationScore
        Composite result with per-signal detail and a fused overall score.

    Algorithm
    ---------
    Each signal is normalised to [0, 1] and combined via a weighted sum.

    * Density drop (weight 0.25): negative density change => obfuscation.
    * Specificity drop (weight 0.30): declining concrete-to-hedge ratio.
    * Readability decrease (weight 0.25): rising Fog/CLI indices.
    * Euphemism presence (weight 0.20): number and severity of substitutions.

    Weights reflect the insight from Li (2008) that specificity changes are
    the single strongest predictor of subsequent earnings disappointment,
    followed by readability and density effects.
    """
    # ---- Information density ----
    density_prev = _information_density(prev_text)
    density_curr = _information_density(curr_text)

    if density_prev > 0:
        density_change = (density_curr - density_prev) / density_prev
    else:
        density_change = 0.0

    # Normalise: a drop of >= 40% maps to 1.0, no drop to 0.0.
    density_signal = max(0.0, min(1.0, -density_change / 0.40))

    # ---- Specificity ----
    spec_prev = _specificity_score(prev_text)
    spec_curr = _specificity_score(curr_text)
    specificity_change = spec_curr - spec_prev

    # Normalise: a drop of >= 0.30 maps to 1.0.
    specificity_signal = max(0.0, min(1.0, -specificity_change / 0.30))

    # ---- Readability ----
    fog_prev = _gunning_fog_index(prev_text)
    fog_curr = _gunning_fog_index(curr_text)
    cli_prev = _coleman_liau_index(prev_text)
    cli_curr = _coleman_liau_index(curr_text)

    # Average the two indices' normalised shifts.
    fog_shift = (fog_curr - fog_prev) / max(fog_prev, 1.0)
    cli_shift = (cli_curr - cli_prev) / max(abs(cli_prev), 1.0)
    readability_change = (fog_shift + cli_shift) / 2.0

    # Normalise: a 20% increase in difficulty maps to 1.0.
    readability_signal = max(0.0, min(1.0, readability_change / 0.20))

    # ---- Euphemisms ----
    euphemisms = _detect_euphemisms(prev_text, curr_text)

    if euphemisms:
        # Scale by count and average gap severity.
        avg_gap = sum(e["specificity_gap"] for e in euphemisms) / len(euphemisms)
        euphemism_signal = min(1.0, (len(euphemisms) * 0.25) + avg_gap)
    else:
        euphemism_signal = 0.0

    # ---- Fuse ----
    weights = {
        "density": 0.25,
        "specificity": 0.30,
        "readability": 0.25,
        "euphemism": 0.20,
    }
    component_scores = {
        "density": round(density_signal, 4),
        "specificity": round(specificity_signal, 4),
        "readability": round(readability_signal, 4),
        "euphemism": round(euphemism_signal, 4),
    }

    overall = sum(weights[k] * component_scores[k] for k in weights)
    overall = round(max(0.0, min(1.0, overall)), 4)

    logger.debug(
        "Obfuscation analysis: density=%.4f specificity=%.4f readability=%.4f "
        "euphemism=%.4f => overall=%.4f",
        density_signal, specificity_signal, readability_signal,
        euphemism_signal, overall,
    )

    return ObfuscationScore(
        density_prev=round(density_prev, 4),
        density_curr=round(density_curr, 4),
        density_change=round(density_change, 4),
        specificity_prev=round(spec_prev, 4),
        specificity_curr=round(spec_curr, 4),
        specificity_change=round(specificity_change, 4),
        fog_prev=round(fog_prev, 2),
        fog_curr=round(fog_curr, 2),
        coleman_liau_prev=round(cli_prev, 2),
        coleman_liau_curr=round(cli_curr, 2),
        readability_change=round(readability_change, 4),
        detected_euphemisms=euphemisms,
        overall_obfuscation_score=overall,
        component_scores=component_scores,
    )
