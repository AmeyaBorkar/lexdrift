import logging
from pathlib import Path

import pandas as pd

from lexdrift.nlp.tokenizer import tokenize

logger = logging.getLogger(__name__)

LEXICON_PATH = Path("data/Loughran-McDonald_MasterDictionary_1993-2024.csv")

# Sentiment categories from the Loughran-McDonald dictionary
CATEGORIES = ["negative", "positive", "uncertainty", "litigious", "constraining"]

# In-memory lookup: word -> set of categories it belongs to
_lexicon: dict[str, set[str]] = {}
_loaded = False


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


def score_sentiment(text: str) -> dict[str, float]:
    """Score text against Loughran-McDonald categories.

    Returns normalized scores (count / total words) for each category.
    """
    load_lexicon()

    tokens = tokenize(text)
    total = len(tokens)
    if total == 0:
        return {cat: 0.0 for cat in CATEGORIES}

    counts: dict[str, int] = {cat: 0 for cat in CATEGORIES}
    for token in tokens:
        cats = _lexicon.get(token.upper(), set())
        for cat in cats:
            counts[cat] += 1

    return {cat: counts[cat] / total for cat in CATEGORIES}
