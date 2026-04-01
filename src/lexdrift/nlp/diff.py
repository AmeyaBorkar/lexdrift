import difflib

from lexdrift.nlp.tokenizer import sentence_split


def unified_diff(prev_text: str, curr_text: str, context_lines: int = 3) -> str:
    """Generate a unified diff between two section texts at the sentence level."""
    prev_sentences = sentence_split(prev_text)
    curr_sentences = sentence_split(curr_text)

    diff = difflib.unified_diff(
        prev_sentences,
        curr_sentences,
        fromfile="previous",
        tofile="current",
        lineterm="",
        n=context_lines,
    )
    return "\n".join(diff)


def diff_stats(prev_text: str, curr_text: str) -> dict:
    """Compute diff statistics between two texts."""
    prev_sentences = sentence_split(prev_text)
    curr_sentences = sentence_split(curr_text)

    matcher = difflib.SequenceMatcher(None, prev_sentences, curr_sentences)
    opcodes = matcher.get_opcodes()

    added = 0
    removed = 0
    changed = 0
    unchanged = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            unchanged += i2 - i1
        elif tag == "insert":
            added += j2 - j1
        elif tag == "delete":
            removed += i2 - i1
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1)

    total_prev = len(prev_sentences)
    total_curr = len(curr_sentences)

    return {
        "sentences_added": added,
        "sentences_removed": removed,
        "sentences_changed": changed,
        "sentences_unchanged": unchanged,
        "total_prev": total_prev,
        "total_curr": total_curr,
        "similarity_ratio": matcher.ratio(),
    }
