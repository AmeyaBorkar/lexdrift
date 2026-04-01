import json
import logging
from collections import Counter
from pathlib import Path

from lexdrift.config import settings
from lexdrift.nlp.tokenizer import tokenize

logger = logging.getLogger(__name__)

# Loaded at runtime from file + DB
_priority_phrases: set[str] = set()
_loaded = False


def _extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Extract n-grams from a token list."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def load_priority_phrases(path: Path | None = None) -> set[str]:
    """Load high-priority phrases from the default config file.

    These are phrases that ALWAYS trigger alerts on appearance/disappearance,
    independent of the automatic n-gram discovery.
    """
    global _priority_phrases, _loaded
    if _loaded:
        return _priority_phrases

    path = path or Path(settings.priority_phrases_path)
    if path.exists():
        data = json.loads(path.read_text())
        _priority_phrases = {p.lower() for p in data.get("phrases", [])}
        logger.info(f"Loaded {len(_priority_phrases)} priority phrases from {path}")
    else:
        logger.warning(f"Priority phrases file not found at {path}")

    _loaded = True
    return _priority_phrases


def discover_ngram_changes(
    prev_text: str,
    curr_text: str,
    min_freq: int | None = None,
    top_k: int | None = None,
) -> dict[str, list[dict]]:
    """Automatically discover meaningful n-gram changes between two texts.

    Instead of checking a hardcoded list, this diffs ALL bigrams and trigrams
    between two filings and surfaces the most significant additions/removals.

    Args:
        prev_text: Previous filing section text.
        curr_text: Current filing section text.
        min_freq: Minimum frequency in a document for an n-gram to be considered.
        top_k: Max number of appeared/disappeared n-grams to return.

    Returns:
        {appeared: [{phrase, freq}], disappeared: [{phrase, freq}]}
    """
    min_freq = min_freq if min_freq is not None else settings.ngram_min_freq
    top_k = top_k if top_k is not None else settings.ngram_top_k

    prev_tokens = tokenize(prev_text)
    curr_tokens = tokenize(curr_text)

    # Build frequency maps for bigrams and trigrams
    prev_ngrams: Counter[str] = Counter()
    curr_ngrams: Counter[str] = Counter()

    for n in (2, 3):
        prev_ngrams.update(_extract_ngrams(prev_tokens, n))
        curr_ngrams.update(_extract_ngrams(curr_tokens, n))

    # Filter to meaningful frequency
    prev_significant = {ng for ng, count in prev_ngrams.items() if count >= min_freq}
    curr_significant = {ng for ng, count in curr_ngrams.items() if count >= min_freq}

    # Find appeared and disappeared
    raw_appeared = curr_significant - prev_significant
    raw_disappeared = prev_significant - curr_significant

    # Rank by frequency (most frequent first — these are the most deliberate additions)
    appeared = sorted(
        [{"phrase": ng, "freq": curr_ngrams[ng]} for ng in raw_appeared],
        key=lambda x: x["freq"],
        reverse=True,
    )[:top_k]

    disappeared = sorted(
        [{"phrase": ng, "freq": prev_ngrams[ng]} for ng in raw_disappeared],
        key=lambda x: x["freq"],
        reverse=True,
    )[:top_k]

    return {"appeared": appeared, "disappeared": disappeared}


def check_priority_phrases(
    prev_text: str,
    curr_text: str,
    extra_phrases: set[str] | None = None,
) -> dict[str, list[str]]:
    """Check high-priority phrases (from config + user-supplied) for changes.

    These are the curated "always important" phrases. Runs alongside
    automatic n-gram discovery.

    Args:
        prev_text: Previous filing section text.
        curr_text: Current filing section text.
        extra_phrases: Additional user-defined phrases to track.
    """
    priority = load_priority_phrases()
    all_phrases = priority | (extra_phrases or set())

    prev_lower = prev_text.lower()
    curr_lower = curr_text.lower()

    prev_found = {p for p in all_phrases if p in prev_lower}
    curr_found = {p for p in all_phrases if p in curr_lower}

    return {
        "appeared": sorted(curr_found - prev_found),
        "disappeared": sorted(prev_found - curr_found),
        "persisted": sorted(prev_found & curr_found),
    }


def compare_phrases(
    prev_text: str,
    curr_text: str,
    extra_phrases: set[str] | None = None,
) -> dict:
    """Full phrase comparison: priority phrase check + automatic n-gram discovery.

    Returns:
        {
            priority: {appeared, disappeared, persisted},
            discovered: {appeared: [{phrase, freq}], disappeared: [{phrase, freq}]}
        }
    """
    priority = check_priority_phrases(prev_text, curr_text, extra_phrases)
    discovered = discover_ngram_changes(prev_text, curr_text)

    return {
        "priority": priority,
        "discovered": discovered,
    }
