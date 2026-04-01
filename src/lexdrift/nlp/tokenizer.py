import re

# Lightweight tokenizer that avoids requiring spaCy model download at import time.
# For production, swap in spaCy for better sentence splitting.

_WORD_RE = re.compile(r"\b[a-zA-Z]+(?:'[a-zA-Z]+)?\b")

# Common abbreviations found in SEC filings that should NOT trigger sentence splits
_ABBREVIATIONS = frozenset({
    "Dr", "Mr", "Mrs", "Ms", "Inc", "Corp", "Ltd", "Co", "No", "vs",
    "approx", "est", "al", "eg", "ie", "etc", "Vol", "Dept", "Div",
    "Sec", "Gov", "Gen", "Jr", "Sr", "Prof", "Rev",
})

# Build a pattern that matches abbreviations followed by a period
# e.g. "Dr." "Inc." "Corp." etc.
_ABBREV_PATTERN = r"(?:" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\."

# Pattern for multi-dot abbreviations like "U.S." or "S.E.C."
_MULTI_DOT_ABBREV = r"(?:[A-Z]\.){2,}"

# Pattern for decimal numbers like "10.5" or "3.14"
_DECIMAL_NUMBER = r"\d+\.\d+"


def tokenize(text: str) -> list[str]:
    """Extract lowercase word tokens from text."""
    return [m.group().lower() for m in _WORD_RE.finditer(text)]


def sentence_split(text: str) -> list[str]:
    """Split text into sentences with awareness of abbreviations and decimals.

    Handles common pitfalls in SEC filings: abbreviations (Dr., Inc., Corp.),
    multi-dot abbreviations (U.S., S.E.C.), and decimal numbers (10.5%).
    """
    if not text or not text.strip():
        return []

    # Replace known abbreviations with placeholders to protect them from splitting
    protected = text
    placeholders: list[tuple[str, str]] = []

    # Protect multi-dot abbreviations like "U.S." first (more specific)
    for i, match in enumerate(re.finditer(_MULTI_DOT_ABBREV, protected)):
        placeholder = f"\x00MULTIDOT{i}\x00"
        placeholders.append((placeholder, match.group()))
    for placeholder, original in placeholders:
        protected = protected.replace(original, placeholder, 1)

    # Protect decimal numbers like "10.5"
    decimal_placeholders: list[tuple[str, str]] = []
    for i, match in enumerate(re.finditer(_DECIMAL_NUMBER, protected)):
        placeholder = f"\x00DECIMAL{i}\x00"
        decimal_placeholders.append((placeholder, match.group()))
    for placeholder, original in decimal_placeholders:
        protected = protected.replace(original, placeholder, 1)

    # Protect known abbreviations like "Dr." "Inc."
    abbrev_placeholders: list[tuple[str, str]] = []
    for i, match in enumerate(re.finditer(_ABBREV_PATTERN, protected)):
        placeholder = f"\x00ABBREV{i}\x00"
        abbrev_placeholders.append((placeholder, match.group()))
    for placeholder, original in abbrev_placeholders:
        protected = protected.replace(original, placeholder, 1)

    # Now split on sentence-ending punctuation followed by space and uppercase letter
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", protected)

    # Restore all placeholders
    sentences = []
    for part in parts:
        restored = part
        for placeholder, original in abbrev_placeholders:
            restored = restored.replace(placeholder, original)
        for placeholder, original in decimal_placeholders:
            restored = restored.replace(placeholder, original)
        for placeholder, original in placeholders:
            restored = restored.replace(placeholder, original)
        stripped = restored.strip()
        if stripped:
            sentences.append(stripped)

    return sentences
