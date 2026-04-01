import logging

import numpy as np

from lexdrift.nlp.embeddings import bytes_to_embedding, cosine_distance, encode_text, embedding_to_bytes
from lexdrift.nlp.risk import score_changes
from lexdrift.nlp.sentences import compare_sentences
from lexdrift.nlp.sentiment import score_sentiment
from lexdrift.nlp.tokenizer import tokenize

logger = logging.getLogger(__name__)


def jaccard_distance(tokens_a: list[str], tokens_b: list[str]) -> float:
    """Compute Jaccard distance between two token sets. Higher = more different."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    if not set_a and not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return 1.0 - (intersection / union)


def word_change_counts(tokens_a: list[str], tokens_b: list[str]) -> tuple[int, int]:
    """Count words added and removed between two token lists."""
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    added = len(set_b - set_a)
    removed = len(set_a - set_b)
    return added, removed


def sentiment_delta(scores_a: dict[str, float], scores_b: dict[str, float]) -> dict[str, float]:
    """Compute the change in sentiment scores (current - previous)."""
    return {cat: scores_b.get(cat, 0.0) - scores_a.get(cat, 0.0) for cat in scores_a}


def compute_drift(
    prev_text: str,
    curr_text: str,
    prev_embedding: bytes | None = None,
    curr_embedding: bytes | None = None,
) -> dict:
    """Compute full drift metrics between two section texts.

    Returns a dict with: cosine_distance, jaccard_distance, added_words,
    removed_words, sentiment_delta, curr_embedding_bytes, prev_embedding_bytes.
    """
    # Tokenize
    tokens_prev = tokenize(prev_text)
    tokens_curr = tokenize(curr_text)

    # Jaccard
    jac_dist = jaccard_distance(tokens_prev, tokens_curr)

    # Word changes
    added, removed = word_change_counts(tokens_prev, tokens_curr)

    # Sentiment
    sent_prev = score_sentiment(prev_text)
    sent_curr = score_sentiment(curr_text)
    sent_delta = sentiment_delta(sent_prev, sent_curr)

    # Embeddings and cosine distance
    if prev_embedding:
        emb_prev = bytes_to_embedding(prev_embedding)
    else:
        emb_prev = encode_text(prev_text)

    if curr_embedding:
        emb_curr = bytes_to_embedding(curr_embedding)
    else:
        emb_curr = encode_text(curr_text)

    cos_dist = cosine_distance(emb_prev, emb_curr)

    # Sentence-level semantic comparison (the middle layer)
    sentence_comparison = compare_sentences(prev_text, curr_text)

    # Risk-score every flagged sentence change
    scored_changes = score_changes(sentence_comparison)

    return {
        "cosine_distance": cos_dist,
        "jaccard_distance": jac_dist,
        "added_words": added,
        "removed_words": removed,
        "sentiment_delta": sent_delta,
        "sentiment_prev": sent_prev,
        "sentiment_curr": sent_curr,
        "curr_embedding_bytes": embedding_to_bytes(emb_curr),
        "prev_embedding_bytes": embedding_to_bytes(emb_prev),
        "sentence_changes": scored_changes,
    }
