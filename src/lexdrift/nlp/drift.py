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


def _zero_sentiment() -> dict[str, float]:
    """Return a zero-valued sentiment dict for fallback."""
    return {"negative": 0.0, "positive": 0.0, "uncertainty": 0.0, "litigious": 0.0, "constraining": 0.0}


def _empty_scored_changes() -> dict:
    """Return an empty scored-changes structure for fallback."""
    return {
        "added": [],
        "removed": [],
        "changed": [],
        "likely_replacements": [],
        "unchanged_count": 0,
        "stats": {
            "total_prev": 0, "total_curr": 0,
            "added": 0, "removed": 0, "changed": 0,
        },
        "risk_summary": {
            "max_risk_score": 0.0,
            "max_risk_level": "low",
            "critical_changes": 0,
            "high_risk_changes": 0,
        },
    }


def validated_bytes_to_embedding(data: bytes, expected_dim: int = 384) -> np.ndarray:
    """Deserialize bytes to embedding with validation.

    Checks that byte length is a multiple of 4 (float32) and that the
    resulting array has the expected dimension.

    Raises ValueError if validation fails.
    """
    if len(data) % 4 != 0:
        raise ValueError(
            f"Embedding byte length {len(data)} is not a multiple of 4 (float32). "
            "Data may be corrupted."
        )

    embedding = bytes_to_embedding(data)

    if embedding.shape[0] != expected_dim:
        raise ValueError(
            f"Embedding dimension {embedding.shape[0]} does not match "
            f"expected dimension {expected_dim}. Model mismatch or corrupted data."
        )

    return embedding


def compute_drift(
    prev_text: str,
    curr_text: str,
    prev_embedding: bytes | None = None,
    curr_embedding: bytes | None = None,
) -> dict:
    """Compute full drift metrics between two section texts.

    Returns a dict with: cosine_distance, jaccard_distance, added_words,
    removed_words, sentiment_delta, curr_embedding_bytes, prev_embedding_bytes.

    Core metrics (tokenization, embeddings, cosine/jaccard distance) propagate
    errors on failure since they are essential.  Auxiliary modules (sentiment,
    sentence comparison, risk scoring) degrade gracefully with warnings.
    """
    # Tokenize — core metric, errors propagate
    tokens_prev = tokenize(prev_text)
    tokens_curr = tokenize(curr_text)

    # Jaccard — core metric, errors propagate
    jac_dist = jaccard_distance(tokens_prev, tokens_curr)

    # Word changes — core metric, errors propagate
    added, removed = word_change_counts(tokens_prev, tokens_curr)

    # Sentiment — auxiliary, degrade gracefully
    try:
        sent_prev = score_sentiment(prev_text)
        sent_curr = score_sentiment(curr_text)
        sent_delta = sentiment_delta(sent_prev, sent_curr)
    except Exception:
        logger.warning("Sentiment scoring failed; returning zero-valued sentiment", exc_info=True)
        sent_prev = _zero_sentiment()
        sent_curr = _zero_sentiment()
        sent_delta = _zero_sentiment()

    # Embeddings and cosine distance — core metric, errors propagate
    if prev_embedding:
        emb_prev = validated_bytes_to_embedding(prev_embedding)
    else:
        emb_prev = encode_text(prev_text)

    if curr_embedding:
        emb_curr = validated_bytes_to_embedding(curr_embedding)
    else:
        emb_curr = encode_text(curr_text)

    cos_dist = cosine_distance(emb_prev, emb_curr)

    # Sentence-level semantic comparison — auxiliary, degrade gracefully
    try:
        sentence_comparison = compare_sentences(prev_text, curr_text)
    except Exception:
        logger.warning("Sentence comparison failed; returning empty sentence_changes", exc_info=True)
        sentence_comparison = {
            "added": [], "removed": [], "changed": [], "likely_replacements": [],
            "unchanged_count": 0,
            "stats": {"total_prev": 0, "total_curr": 0, "added": 0, "removed": 0, "changed": 0},
        }

    # Risk-score every flagged sentence change — auxiliary, degrade gracefully
    try:
        scored_changes = score_changes(sentence_comparison)
    except Exception:
        logger.warning("Risk scoring failed; returning unscored changes", exc_info=True)
        scored_changes = sentence_comparison.copy()
        scored_changes.setdefault("risk_summary", {
            "max_risk_score": 0.0,
            "max_risk_level": "low",
            "critical_changes": 0,
            "high_risk_changes": 0,
        })

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
