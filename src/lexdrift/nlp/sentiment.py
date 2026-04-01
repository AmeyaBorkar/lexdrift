"""Sentiment scoring using the Loughran-McDonald financial dictionary.

Context-aware: checks for negation within a 5-token window (with scope
breakers) before each sentiment word, flipping the category when negated.
This handles cases like "no material weakness" correctly (negated negative
-> positive signal).

When ``settings.use_finbert_sentiment`` is True, positive/negative scores
are produced by FinBERT (``ProsusAI/finbert``) while uncertainty, litigious,
and constraining scores still come from the Loughran-McDonald dictionary.
"""

import logging
import threading
from pathlib import Path

import pandas as pd

from lexdrift.config import settings
from lexdrift.nlp.tokenizer import tokenize, sentence_split

logger = logging.getLogger(__name__)

LEXICON_PATH = Path("data/Loughran-McDonald_MasterDictionary_1993-2024.csv")

# Sentiment categories from the Loughran-McDonald dictionary
CATEGORIES = ["negative", "positive", "uncertainty", "litigious", "constraining"]

# In-memory lookup: word -> set of categories it belongs to
_lexicon: dict[str, set[str]] = {}
_loaded = False

# Negation words — if one of these appears within 5 tokens before a
# sentiment word (with no scope breaker in between), the sentiment
# category is FLIPPED.
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "without", "lack", "absence",
    "neither", "nor", "fail", "failed", "unable",
    "hardly", "barely", "scarcely", "none", "cannot",
})

# Scope breakers stop negation propagation.
_SCOPE_BREAKERS: frozenset[str] = frozenset({
    ",", ".", ";", "but", "however", "although",
    "nevertheless", "yet", "while",
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


def _is_negated(tokens: list[str], index: int, window: int = 5) -> bool:
    """Check if the token at *index* is preceded by a negation word within *window* tokens.

    Scans backwards from *index* up to *window* tokens, but stops at
    scope breakers (punctuation or conjunctions like *but*, *however*).
    """
    start = max(0, index - window)
    # Walk backwards so we stop at the nearest scope breaker
    for i in range(index - 1, start - 1, -1):
        tok = tokens[i].lower()
        if tok in _SCOPE_BREAKERS:
            return False  # scope breaker reached — no negation applies
        if tok in _NEGATION_WORDS:
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


# ---------------------------------------------------------------------------
# FinBERT-based sentiment (positive / negative / neutral)
# ---------------------------------------------------------------------------

_finbert_pipeline = None
_finbert_lock = threading.Lock()
_finbert_failed = False


def _get_finbert():
    """Lazy-load the FinBERT pipeline with thread-safe double-checked locking."""
    global _finbert_pipeline, _finbert_failed
    if _finbert_pipeline is not None:
        return _finbert_pipeline
    if _finbert_failed:
        return None
    with _finbert_lock:
        if _finbert_pipeline is not None:
            return _finbert_pipeline
        if _finbert_failed:
            return None
        try:
            from transformers import pipeline as hf_pipeline

            logger.info(f"Loading FinBERT model: {settings.finbert_model}")
            _finbert_pipeline = hf_pipeline(
                "sentiment-analysis",
                model=settings.finbert_model,
                truncation=True,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception:
            logger.warning("Failed to load FinBERT model; falling back to dictionary", exc_info=True)
            _finbert_failed = True
            return None
    return _finbert_pipeline


def score_sentiment_finbert(text: str) -> dict[str, float]:
    """Score *text* using FinBERT for positive/negative/neutral.

    Returns a dict with the same keys as the dictionary approach:
    {negative, positive, uncertainty, litigious, constraining}.
    FinBERT produces positive/negative; neutral maps to low values.
    Uncertainty, litigious, and constraining are filled via the
    Loughran-McDonald dictionary.
    """
    classifier = _get_finbert()
    if classifier is None:
        return score_sentiment_contextual(text)

    sentences = sentence_split(text)
    if not sentences:
        sentences = [text] if text.strip() else []
    if not sentences:
        return {cat: 0.0 for cat in CATEGORIES}

    # FinBERT has a 512-token limit; long sentences are truncated by the pipeline.
    results = classifier(sentences)

    pos_total = 0.0
    neg_total = 0.0
    for res in results:
        label = res["label"].lower()
        score = res["score"]
        if label == "positive":
            pos_total += score
        elif label == "negative":
            neg_total += score
        # neutral contributes nothing

    n = len(sentences)
    pos_avg = pos_total / n
    neg_avg = neg_total / n

    # Dictionary scores for categories FinBERT doesn't cover
    dict_scores = score_sentiment_contextual(text)

    return {
        "negative": neg_avg,
        "positive": pos_avg,
        "uncertainty": dict_scores["uncertainty"],
        "litigious": dict_scores["litigious"],
        "constraining": dict_scores["constraining"],
    }


def score_sentiment(text: str) -> dict[str, float]:
    """Score text against Loughran-McDonald categories.

    Returns normalized scores (count / total words) for each category.

    If ``settings.use_finbert_sentiment`` is True and the FinBERT model
    loads successfully, positive/negative scores come from FinBERT while
    uncertainty/litigious/constraining use the dictionary.  Otherwise,
    falls back to the negation-aware dictionary approach.
    """
    if settings.use_finbert_sentiment:
        return score_sentiment_finbert(text)
    return score_sentiment_contextual(text)
