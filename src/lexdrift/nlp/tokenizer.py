import re

# Lightweight tokenizer that avoids requiring spaCy model download at import time.
# For production, swap in spaCy for better sentence splitting.

_WORD_RE = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")


def tokenize(text: str) -> list[str]:
    """Extract lowercase word tokens from text."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def sentence_split(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic."""
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]
