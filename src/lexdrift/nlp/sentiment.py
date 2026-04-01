"""Sentiment scoring using the Loughran-McDonald financial dictionary.

Context-aware: checks for negation within a 3-token window before each
sentiment word, flipping the category when negated.  This handles cases
like "no material weakness" correctly (negated negative -> positive signal).
"""

import logging
from pathlib import Path

import pandas as pd

from lexdrift.nlp.tokenizer import tokenize, sentence_split

logger = logging.getLogger(__name__)

LEXICON_PATH = Path("data/Loughran-McDonald_MasterDictionary_1993-2024.csv")

# Sentiment categories from the Loughran-McDonald dictionary
CATEGORIES = ["negative", "positive", "uncertainty", "litigious", "constraining"]

# In-memory lookup: word -> set of categories it belongs to
_lexicon: dict[str, set[str]] = {}
_loaded = False

# Negation words — if one of these appears within 3 tokens before a
# sentiment word, the sentiment category is FLIPPED.
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "without", "lack", "absence",
    "neither", "nor", "fail", "failed", "unable",
    "hardly", "barely", "scarcely", "none", "cannot",
})

# Category flip mapping: negated negative -> positive, negated positive -> negative
_FLIP_MAP: dict[str, str] = {
    "negative": "positive",
    "positive": "negative",
    # uncertainty, litigious, constraining don't have a natural opposite;
    # negation simply cancels them (they are not counted).
}


def load_lexicon(path: Path | None = None) -> None:
    """Load the Loughran-McDonald master dictionary CSV.

    Expected columns: Word, Negative, Positive, Uncertainty, Litigious, Constraining
    Non-zero values in a column mean the word belongs to that category.
    """
    global _lexicon, _loaded
    if _loaded:
        return

    path = path or LEXICON_PATH
    if not path.exists():
        logger.warning(
            f"Loughran-McDonald dictionary not found at {path}. "
            "Download from https://sraf.nd.edu/loughranmcdonald-master-dictionary/ "
            "and place the CSV at data/loughran_mcdonald.csv"
        )
        _loaded = True
        return

    logger.info(f"Loading Loughran-McDonald dictionary from {path}")
    df = pd.read_csv(path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    for _, row in df.iterrows():
        word = str(row["word"]).upper()
        cats = set()
        for cat in CATEGORIES:
            col = cat
            if col in df.columns and pd.notna(row[col]) and row[col] != 0:
                cats.add(cat)
        if cats:
            _lexicon[word] = cats

    _loaded = True
    logger.info(f"Loaded {len(_lexicon)} sentiment words")


def _is_negated(tokens: list[str], index: int, window: int = 3) -> bool:
    """Check if the token at *index* is preceded by a negation word within *window* tokens."""
    start = max(0, index - window)
    for i in range(start, index):
        if tokens[i].lower() in _NEGATION_WORDS:
            return True
    return False


def _score_tokens_contextual(tokens: list[str]) -> dict[str, int]:
    """Score a list of tokens with negation awareness.

    For each sentiment word, check if it is negated. If negated:
    - negative/positive are FLIPPED (negated negative counts as positive)
    - uncertainty/litigious/constraining are CANCELLED (not counted)
    """
    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for i, token in enumerate(tokens):
        cats = _lexicon.get(token.upper(), set())
        if not cats:
            continue

        negated = _is_negated(tokens, i)

        for cat in cats:
            if negated:
                flipped = _FLIP_MAP.get(cat)
                if flipped:
                    counts[flipped] += 1
                # else: cancelled (uncertainty etc.) — don't count
            else:
                counts[cat] += 1

    return counts


def score_sentiment_contextual(text: str) -> dict[str, float]:
    """Score text against Loughran-McDonald categories with negation awareness.

    Splits into sentences, scores each with context, and returns
    aggregated normalized scores (count / total words).
    """
    load_lexicon()

    sentences = sentence_split(text)
    if not sentences:
        # Fall back to treating entire text as one block
        sentences = [text] if text.strip() else []

    total_tokens = 0
    total_counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for sent in sentences:
        tokens = tokenize(sent)
        total_tokens += len(tokens)
        sent_counts = _score_tokens_contextual(tokens)
        for cat in CATEGORIES:
            total_counts[cat] += sent_counts[cat]

    if total_tokens == 0:
        return {cat: 0.0 for cat in CATEGORIES}

    return {cat: total_counts[cat] / total_tokens for cat in CATEGORIES}


def score_sentiment(text: str) -> dict[str, float]:
    """Score text against Loughran-McDonald categories.

    Returns normalized scores (count / total words) for each category.
    Uses context-aware scoring with negation handling by default.
    """
    return score_sentiment_contextual(text)
