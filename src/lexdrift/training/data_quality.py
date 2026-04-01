"""Elite Training Data Generator — High-quality pairs from text-level signals.

The standard training pipeline (finetune.generate_training_pairs) uses the base
model's own similarity scores as labels, creating a circular dependency:

    noisy model -> noisy labels -> train on noisy labels -> still noisy model

This module breaks the cycle by generating training pairs using ONLY text-level
signals that require ZERO model involvement:

    Tier 1: Exact-match positives (verbatim sentences across consecutive filings)
    Tier 2: High-overlap positives (Jaccard similarity on token sets > 0.8)
    Tier 3: Cross-section hard negatives (risk_factors vs mdna, same company)
    Tier 4: Cross-company boilerplate positives (same sentence in 3+ companies)
    Tier 5: Outcome-anchored hard negatives (critical-term adds vs removes)

Every label is derived from string operations, set arithmetic, or database
lookups — never from embeddings or cosine similarity.

Usage:
    from lexdrift.training.data_quality import generate_elite_pairs

    pairs = generate_elite_pairs(db_session, max_pairs=50000)
"""

from __future__ import annotations

import logging
import random
from collections import Counter, defaultdict

from sqlalchemy import select
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_SENTENCE_LENGTH: int = 30
"""Minimum character length to include a sentence (filters headers/artifacts)."""

JACCARD_HIGH_OVERLAP_THRESHOLD: float = 0.8
"""Token-set Jaccard similarity above which a pair is considered a near-match."""

BOILERPLATE_COMPANY_THRESHOLD: int = 3
"""Minimum distinct companies sharing a sentence for it to count as boilerplate."""

CROSS_SECTION_PAIRS = [
    ("risk_factors", "mdna"),
    ("risk_factors", "business"),
    ("mdna", "business"),
]
"""Section-type pairs used for Tier 3 cross-section hard negatives."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase and strip whitespace for comparison."""
    return text.strip().lower()


def _jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity (intersection / union) between two token sets."""
    if not tokens_a and not tokens_b:
        return 1.0
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    return intersection / union if union > 0 else 0.0


def _split_and_filter(text: str, min_length: int = MIN_SENTENCE_LENGTH) -> list[str]:
    """Split text into sentences and filter out short ones."""
    from lexdrift.nlp.tokenizer import sentence_split

    return [s for s in sentence_split(text) if len(s) >= min_length]


def _tokenize_sentence(sentence: str) -> set[str]:
    """Tokenize a sentence into a set of lowercase word tokens."""
    from lexdrift.nlp.tokenizer import tokenize

    return set(tokenize(sentence))


# ---------------------------------------------------------------------------
# Tier 1: Exact Match Positives
# ---------------------------------------------------------------------------

def _generate_tier1_exact_matches(
    section_pairs: list[tuple[str, str, int, str]],
    max_per_pair: int = 500,
) -> list[dict]:
    """Find sentences that appear verbatim in both filings of a consecutive pair.

    Args:
        section_pairs: List of (prev_text, curr_text, company_id, section_type).
        max_per_pair: Cap on exact-match pairs per filing pair.

    Returns:
        List of dicts with keys: text_a, text_b, label, tier, metadata.
    """
    pairs = []

    for prev_text, curr_text, company_id, section_type in section_pairs:
        prev_sentences = _split_and_filter(prev_text)
        curr_sentences = _split_and_filter(curr_text)

        # Build normalized lookup for current filing
        curr_norm_to_original: dict[str, str] = {}
        for s in curr_sentences:
            norm = _normalize(s)
            if norm not in curr_norm_to_original:
                curr_norm_to_original[norm] = s

        pair_matches = []
        for prev_s in prev_sentences:
            norm = _normalize(prev_s)
            if norm in curr_norm_to_original:
                pair_matches.append({
                    "text_a": prev_s,
                    "text_b": curr_norm_to_original[norm],
                    "label": 1.0,
                    "tier": 1,
                    "metadata": {
                        "company_id": company_id,
                        "section_type": section_type,
                        "signal": "exact_match",
                    },
                })

        # Cap per filing pair to avoid overweighting prolific filers
        if len(pair_matches) > max_per_pair:
            pair_matches = random.sample(pair_matches, max_per_pair)

        pairs.extend(pair_matches)

    logger.info("Tier 1 (exact match): %d pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Tier 2: High-Overlap Positives (Jaccard on token sets)
# ---------------------------------------------------------------------------

def _generate_tier2_high_overlap(
    section_pairs: list[tuple[str, str, int, str]],
    max_per_pair: int = 500,
) -> list[dict]:
    """Find sentence pairs with high word overlap (Jaccard > 0.8) that are NOT
    exact matches. These are sentences with minor edits across filings.

    The label is the Jaccard similarity score itself (0.85-0.95 range).

    Args:
        section_pairs: List of (prev_text, curr_text, company_id, section_type).
        max_per_pair: Cap on pairs per filing pair.

    Returns:
        List of dicts with keys: text_a, text_b, label, tier, metadata.
    """
    pairs = []

    for prev_text, curr_text, company_id, section_type in section_pairs:
        prev_sentences = _split_and_filter(prev_text)
        curr_sentences = _split_and_filter(curr_text)

        if not prev_sentences or not curr_sentences:
            continue

        # Precompute token sets for current sentences
        curr_token_sets = [(s, _tokenize_sentence(s)) for s in curr_sentences]

        # Build set of exact-match norms to exclude
        curr_norms = {_normalize(s) for s in curr_sentences}

        pair_candidates = []
        for prev_s in prev_sentences:
            prev_norm = _normalize(prev_s)
            if prev_norm in curr_norms:
                continue  # Skip exact matches (those are Tier 1)

            prev_tokens = _tokenize_sentence(prev_s)
            if not prev_tokens:
                continue

            # Find the best-matching current sentence by Jaccard
            best_jaccard = 0.0
            best_curr_s = None
            for curr_s, curr_tokens in curr_token_sets:
                if _normalize(curr_s) == prev_norm:
                    continue
                jac = _jaccard_similarity(prev_tokens, curr_tokens)
                if jac > best_jaccard:
                    best_jaccard = jac
                    best_curr_s = curr_s

            if best_jaccard >= JACCARD_HIGH_OVERLAP_THRESHOLD and best_curr_s is not None:
                pair_candidates.append({
                    "text_a": prev_s,
                    "text_b": best_curr_s,
                    "label": round(best_jaccard, 4),
                    "tier": 2,
                    "metadata": {
                        "company_id": company_id,
                        "section_type": section_type,
                        "signal": "high_jaccard_overlap",
                        "jaccard": round(best_jaccard, 4),
                    },
                })

        if len(pair_candidates) > max_per_pair:
            pair_candidates = random.sample(pair_candidates, max_per_pair)

        pairs.extend(pair_candidates)

    logger.info("Tier 2 (high overlap): %d pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Tier 3: Cross-Section Hard Negatives
# ---------------------------------------------------------------------------

def _generate_tier3_cross_section(
    sections_by_company: dict[int, dict[str, list[str]]],
    max_per_company: int = 200,
) -> list[dict]:
    """Pair sentences from different section types of the SAME company.

    For example, a sentence from risk_factors paired with a sentence from mdna.
    These share company context but discuss fundamentally different topics,
    making them strong negatives.

    Args:
        sections_by_company: {company_id: {section_type: [sentences]}}.
        max_per_company: Cap on cross-section pairs per company.

    Returns:
        List of dicts with keys: text_a, text_b, label, tier, metadata.
    """
    pairs = []

    for company_id, sections in sections_by_company.items():
        company_pairs = []

        for section_a, section_b in CROSS_SECTION_PAIRS:
            sents_a = sections.get(section_a, [])
            sents_b = sections.get(section_b, [])

            if not sents_a or not sents_b:
                continue

            # Sample pairs: pair random sentences from different sections
            n_pairs = min(len(sents_a), len(sents_b), max_per_company // len(CROSS_SECTION_PAIRS))
            sampled_a = random.sample(sents_a, min(n_pairs, len(sents_a)))
            sampled_b = random.sample(sents_b, min(n_pairs, len(sents_b)))

            for sa, sb in zip(sampled_a, sampled_b):
                company_pairs.append({
                    "text_a": sa,
                    "text_b": sb,
                    "label": 0.0,
                    "tier": 3,
                    "metadata": {
                        "company_id": company_id,
                        "section_a": section_a,
                        "section_b": section_b,
                        "signal": "cross_section_negative",
                    },
                })

        if len(company_pairs) > max_per_company:
            company_pairs = random.sample(company_pairs, max_per_company)

        pairs.extend(company_pairs)

    logger.info("Tier 3 (cross-section negatives): %d pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Tier 4: Cross-Company Boilerplate Positives
# ---------------------------------------------------------------------------

def _generate_tier4_boilerplate(
    all_sentences_by_company: dict[int, list[str]],
    min_companies: int = BOILERPLATE_COMPANY_THRESHOLD,
    max_pairs: int = 5000,
) -> list[dict]:
    """Find sentences that appear in filings of 3+ different companies.

    These are regulatory boilerplate — same meaning in different contexts.
    Pairing them teaches the model that regulatory language should cluster
    together regardless of company.

    Args:
        all_sentences_by_company: {company_id: [all sentences from all sections]}.
        min_companies: Minimum distinct companies for a sentence to count.
        max_pairs: Cap on total boilerplate pairs.

    Returns:
        List of dicts with keys: text_a, text_b, label, tier, metadata.
    """
    # Map normalized sentence -> {company_id: original_text}
    sentence_origins: dict[str, dict[int, str]] = defaultdict(dict)

    for company_id, sentences in all_sentences_by_company.items():
        for s in sentences:
            norm = _normalize(s)
            if company_id not in sentence_origins[norm]:
                sentence_origins[norm][company_id] = s

    # Filter to sentences appearing in min_companies+ companies
    boilerplate_groups = {
        norm: origins
        for norm, origins in sentence_origins.items()
        if len(origins) >= min_companies
    }

    logger.info(
        "Tier 4: found %d boilerplate sentences (appearing in %d+ companies)",
        len(boilerplate_groups), min_companies,
    )

    pairs = []
    for norm, origins in boilerplate_groups.items():
        company_ids = list(origins.keys())
        # Create pairs between different companies' versions
        for i in range(len(company_ids)):
            for j in range(i + 1, len(company_ids)):
                pairs.append({
                    "text_a": origins[company_ids[i]],
                    "text_b": origins[company_ids[j]],
                    "label": 0.9,
                    "tier": 4,
                    "metadata": {
                        "company_a": company_ids[i],
                        "company_b": company_ids[j],
                        "signal": "cross_company_boilerplate",
                        "n_companies": len(origins),
                    },
                })

    if len(pairs) > max_pairs:
        pairs = random.sample(pairs, max_pairs)

    logger.info("Tier 4 (boilerplate positives): %d pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Tier 5: Outcome-Anchored Hard Negatives
# ---------------------------------------------------------------------------

def _generate_tier5_outcome_anchored(
    section_pairs: list[tuple[str, str, int, str]],
    max_per_pair: int = 200,
) -> list[dict]:
    """Pair 'added' sentences containing CRITICAL_TERMS with 'removed' sentences
    from the same section.

    Intuition: if a company ADDED 'going concern' language and REMOVED
    'strong financial position' language in the same section, those sentences
    are maximally different in financial meaning.

    This is the text-level proxy for outcome-anchored negatives. The full
    version would incorporate stock price data (designed but not implemented).

    Args:
        section_pairs: List of (prev_text, curr_text, company_id, section_type).
        max_per_pair: Cap on pairs per filing pair.

    Returns:
        List of dicts with keys: text_a, text_b, label, tier, metadata.
    """
    from lexdrift.nlp.risk import CRITICAL_TERMS

    pairs = []

    for prev_text, curr_text, company_id, section_type in section_pairs:
        prev_sentences = _split_and_filter(prev_text)
        curr_sentences = _split_and_filter(curr_text)

        if not prev_sentences or not curr_sentences:
            continue

        # Find sentences in current that are NOT in previous (added)
        prev_norms = {_normalize(s) for s in prev_sentences}
        curr_norms = {_normalize(s) for s in curr_sentences}

        added = [s for s in curr_sentences if _normalize(s) not in prev_norms]
        removed = [s for s in prev_sentences if _normalize(s) not in curr_norms]

        if not added or not removed:
            continue

        # Filter added sentences to those containing critical terms
        critical_added = []
        for s in added:
            s_lower = s.lower()
            triggers = [term for term in CRITICAL_TERMS if term in s_lower]
            if triggers:
                critical_added.append((s, triggers))

        if not critical_added:
            continue

        # Pair each critical-term add with a random removed sentence
        pair_candidates = []
        for added_s, triggers in critical_added:
            removed_s = random.choice(removed)
            pair_candidates.append({
                "text_a": added_s,
                "text_b": removed_s,
                "label": 0.0,
                "tier": 5,
                "metadata": {
                    "company_id": company_id,
                    "section_type": section_type,
                    "signal": "outcome_anchored_negative",
                    "critical_triggers": triggers,
                },
            })

        if len(pair_candidates) > max_per_pair:
            pair_candidates = random.sample(pair_candidates, max_per_pair)

        pairs.extend(pair_candidates)

    logger.info("Tier 5 (outcome-anchored negatives): %d pairs", len(pairs))
    return pairs


# ---------------------------------------------------------------------------
# Outcome-Anchored Interface (future: stock price integration)
# ---------------------------------------------------------------------------

def generate_outcome_anchored_pairs(
    db_session: Session,
    price_data: dict[int, list[dict]] | None = None,
    max_pairs: int = 5000,
) -> list[dict]:
    """[INTERFACE ONLY] Generate training pairs anchored to post-filing outcomes.

    When price_data is provided, pairs sentences from filings where the company
    experienced significant post-filing price moves with sentences from stable
    periods. This teaches the model which disclosure changes are financially
    material.

    Args:
        db_session: Synchronous SQLAlchemy session.
        price_data: Optional mapping of company_id -> list of
            {"filing_date": date, "return_5d": float, "return_30d": float}.
            When None, falls back to Tier 5 text-level signals.
        max_pairs: Maximum number of pairs to generate.

    Returns:
        List of dicts with keys: text_a, text_b, label, tier, metadata.

    Note:
        Stock price integration is designed but not yet implemented.
        This function currently raises NotImplementedError when price_data
        is provided.
    """
    if price_data is not None:
        raise NotImplementedError(
            "Stock price-anchored training pairs are designed but not yet "
            "implemented. Pass price_data=None to use Tier 5 text-level "
            "signals instead."
        )

    # Fallback to Tier 5 text-level signals
    logger.info("Outcome-anchored interface called without price_data; using Tier 5 text signals")
    return []


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_elite_pairs(
    db_session: Session,
    max_pairs: int = 50000,
) -> list:
    """Generate high-quality training pairs using text-level signals only.

    No model-based labels. No circular dependency. 5 tiers of confidence.

    Queries the Section and Filing tables directly to get raw section text,
    then splits into sentences locally. Does NOT depend on SentenceChange
    table, so this can run before any /analyze calls.

    Args:
        db_session: A synchronous SQLAlchemy session.
        max_pairs: Maximum total pairs to generate across all tiers.

    Returns:
        A list of ``sentence_transformers.InputExample`` objects ready for
        training, identical format to ``generate_training_pairs``.
    """
    from sentence_transformers import InputExample

    from lexdrift.db.models import Company, Filing, Section

    logger.info("Generating elite training pairs from raw section text (no model labels)")

    # ------------------------------------------------------------------
    # Step 1: Load all sections grouped by (company_id, section_type)
    # ------------------------------------------------------------------
    stmt = (
        select(
            Section.section_text,
            Section.section_type,
            Filing.company_id,
            Filing.id.label("filing_id"),
            Filing.filing_date,
        )
        .join(Filing, Section.filing_id == Filing.id)
        .where(Section.section_text.isnot(None))
        .where(Section.word_count >= 100)  # Exclude bad parses (short stubs / cross-refs)
        .order_by(Filing.company_id, Section.section_type, Filing.filing_date)
    )

    rows = db_session.execute(stmt).all()
    logger.info("Loaded %d section records from the database", len(rows))

    if not rows:
        logger.warning("No sections found in database; cannot generate training pairs")
        return []

    # Group by (company_id, section_type) in filing-date order
    grouped: dict[tuple[int, str], list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        key = (row.company_id, row.section_type)
        grouped[key].append((row.section_text, row.filing_id))

    # ------------------------------------------------------------------
    # Step 2: Build consecutive section pairs for Tiers 1, 2, 5
    # ------------------------------------------------------------------
    section_pairs: list[tuple[str, str, int, str]] = []
    for (company_id, section_type), filings in grouped.items():
        for i in range(1, len(filings)):
            prev_text, _prev_id = filings[i - 1]
            curr_text, _curr_id = filings[i]
            if prev_text and curr_text:
                section_pairs.append((prev_text, curr_text, company_id, section_type))

    logger.info(
        "Built %d consecutive section pairs from %d (company, section) groups",
        len(section_pairs), len(grouped),
    )

    # ------------------------------------------------------------------
    # Step 3: Build per-company section sentence pools for Tiers 3, 4
    # ------------------------------------------------------------------
    # For Tier 3: {company_id: {section_type: [sentences]}}
    sections_by_company: dict[int, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    # For Tier 4: {company_id: [all sentences]}
    all_sentences_by_company: dict[int, list[str]] = defaultdict(list)

    for (company_id, section_type), filings in grouped.items():
        for text, _fid in filings:
            if not text:
                continue
            sentences = _split_and_filter(text)
            sections_by_company[company_id][section_type].extend(sentences)
            all_sentences_by_company[company_id].extend(sentences)

    # Deduplicate per-section sentence lists (keep order)
    for company_id in sections_by_company:
        for section_type in sections_by_company[company_id]:
            seen = set()
            deduped = []
            for s in sections_by_company[company_id][section_type]:
                norm = _normalize(s)
                if norm not in seen:
                    seen.add(norm)
                    deduped.append(s)
            sections_by_company[company_id][section_type] = deduped

    # ------------------------------------------------------------------
    # Step 4: Generate pairs from each tier
    # ------------------------------------------------------------------
    # Allocate budget proportionally: T1=30%, T2=20%, T3=25%, T4=15%, T5=10%
    budget = {
        1: int(max_pairs * 0.30),
        2: int(max_pairs * 0.20),
        3: int(max_pairs * 0.25),
        4: int(max_pairs * 0.15),
        5: int(max_pairs * 0.10),
    }

    tier1 = _generate_tier1_exact_matches(section_pairs)
    tier2 = _generate_tier2_high_overlap(section_pairs)
    tier3 = _generate_tier3_cross_section(dict(sections_by_company))
    tier4 = _generate_tier4_boilerplate(dict(all_sentences_by_company))
    tier5 = _generate_tier5_outcome_anchored(section_pairs)

    # Cap each tier to its budget
    if len(tier1) > budget[1]:
        tier1 = random.sample(tier1, budget[1])
    if len(tier2) > budget[2]:
        tier2 = random.sample(tier2, budget[2])
    if len(tier3) > budget[3]:
        tier3 = random.sample(tier3, budget[3])
    if len(tier4) > budget[4]:
        tier4 = random.sample(tier4, budget[4])
    if len(tier5) > budget[5]:
        tier5 = random.sample(tier5, budget[5])

    all_pairs = tier1 + tier2 + tier3 + tier4 + tier5

    # Global cap
    if len(all_pairs) > max_pairs:
        all_pairs = random.sample(all_pairs, max_pairs)

    # Shuffle
    random.shuffle(all_pairs)

    # ------------------------------------------------------------------
    # Step 5: Convert to InputExample objects
    # ------------------------------------------------------------------
    examples = [
        InputExample(
            texts=[p["text_a"], p["text_b"]],
            label=float(p["label"]),
        )
        for p in all_pairs
    ]

    logger.info(
        "Generated %d elite training pairs: T1=%d, T2=%d, T3=%d, T4=%d, T5=%d",
        len(examples), len(tier1), len(tier2), len(tier3), len(tier4), len(tier5),
    )

    return examples


# ---------------------------------------------------------------------------
# Data Quality Report
# ---------------------------------------------------------------------------

def data_quality_report(pairs: list) -> dict:
    """Generate statistics about training data quality.

    Accepts either the raw dict pairs (before InputExample conversion) or
    InputExample objects.

    Returns a dict with:
        - total_pairs: int
        - tier_counts: dict[int, int]
        - label_distribution: dict with mean, std, min, max, histogram
        - avg_sentence_length: float (characters)
        - vocabulary_coverage: dict with total_unique_tokens, avg_tokens_per_sentence
        - positive_negative_ratio: float
    """
    from lexdrift.nlp.tokenizer import tokenize

    if not pairs:
        return {"total_pairs": 0, "error": "No pairs provided"}

    # Detect input type and extract texts + labels
    labels = []
    all_texts = []
    tier_counts: Counter = Counter()

    for p in pairs:
        if hasattr(p, "texts") and hasattr(p, "label"):
            # InputExample
            labels.append(p.label)
            all_texts.extend(p.texts)
        elif isinstance(p, dict):
            labels.append(p.get("label", 0.0))
            all_texts.append(p.get("text_a", ""))
            all_texts.append(p.get("text_b", ""))
            tier_counts[p.get("tier", 0)] += 1

    # Label distribution
    labels_sorted = sorted(labels)
    n = len(labels)
    mean_label = sum(labels) / n if n > 0 else 0.0
    variance = sum((l - mean_label) ** 2 for l in labels) / n if n > 0 else 0.0
    std_label = variance ** 0.5

    # Histogram: bucket into 10 bins
    histogram = Counter()
    for l in labels:
        bucket = min(int(l * 10), 9)  # 0-9
        histogram[f"{bucket * 0.1:.1f}-{(bucket + 1) * 0.1:.1f}"] = (
            histogram.get(f"{bucket * 0.1:.1f}-{(bucket + 1) * 0.1:.1f}", 0) + 1
        )

    # Sentence lengths
    sentence_lengths = [len(t) for t in all_texts if t]
    avg_sentence_length = (
        sum(sentence_lengths) / len(sentence_lengths) if sentence_lengths else 0.0
    )

    # Vocabulary coverage
    all_tokens = []
    for t in all_texts:
        if t:
            all_tokens.extend(tokenize(t))
    unique_tokens = set(all_tokens)
    n_sentences = len([t for t in all_texts if t])

    # Positive/negative ratio
    n_positive = sum(1 for l in labels if l >= 0.5)
    n_negative = sum(1 for l in labels if l < 0.5)

    return {
        "total_pairs": n,
        "tier_counts": dict(sorted(tier_counts.items())),
        "label_distribution": {
            "mean": round(mean_label, 4),
            "std": round(std_label, 4),
            "min": round(min(labels) if labels else 0.0, 4),
            "max": round(max(labels) if labels else 0.0, 4),
            "histogram": dict(sorted(histogram.items())),
        },
        "avg_sentence_length_chars": round(avg_sentence_length, 1),
        "vocabulary_coverage": {
            "total_unique_tokens": len(unique_tokens),
            "total_tokens": len(all_tokens),
            "avg_tokens_per_sentence": round(
                len(all_tokens) / n_sentences if n_sentences else 0.0, 1
            ),
        },
        "positive_negative_ratio": round(
            n_positive / n_negative if n_negative > 0 else float("inf"), 3
        ),
        "positive_count": n_positive,
        "negative_count": n_negative,
    }
