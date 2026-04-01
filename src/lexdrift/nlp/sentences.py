"""Sentence-level semantic comparison between two filing sections.

This is the middle layer that catches what section-level embeddings miss
(buried changes) and what n-gram analysis misses (paraphrasing, negation,
novel language). It works by:

1. Splitting both sections into individual sentences
2. Embedding every sentence
3. Computing a similarity matrix (all pairs)
4. Aligning sentences using greedy best-match
5. Flagging: new sentences, removed sentences, semantically changed sentences
"""

import logging

import numpy as np

from lexdrift.nlp.tokenizer import sentence_split

logger = logging.getLogger(__name__)

# Thresholds for sentence classification
MATCH_THRESHOLD = 0.85   # above this = "unchanged" (same sentence, maybe minor edits)
CHANGE_THRESHOLD = 0.55  # between this and MATCH_THRESHOLD = "changed" (same topic, different meaning)
                          # below CHANGE_THRESHOLD = "unmatched" (new or removed content)


def _batch_encode_sentences(sentences: list[str]) -> np.ndarray:
    """Encode a list of sentences into embedding vectors.

    Uses the same model as the section-level embeddings but operates
    on individual sentences. Returns shape (n_sentences, embedding_dim).
    """
    if not sentences:
        return np.array([])

    # Import here to avoid circular dependency and allow lazy model loading
    from lexdrift.nlp.embeddings import _get_model

    model = _get_model()
    embeddings = model.encode(sentences, show_progress_bar=False, batch_size=64)
    return embeddings.astype(np.float32)


def _compute_similarity_matrix(
    embeddings_a: np.ndarray, embeddings_b: np.ndarray
) -> np.ndarray:
    """Compute cosine similarity between all pairs of sentences.

    Returns shape (len_a, len_b) where entry [i,j] is the cosine similarity
    between sentence i from section A and sentence j from section B.
    """
    # Normalize rows to unit vectors
    norms_a = np.linalg.norm(embeddings_a, axis=1, keepdims=True)
    norms_b = np.linalg.norm(embeddings_b, axis=1, keepdims=True)

    # Avoid division by zero
    norms_a = np.where(norms_a == 0, 1, norms_a)
    norms_b = np.where(norms_b == 0, 1, norms_b)

    normed_a = embeddings_a / norms_a
    normed_b = embeddings_b / norms_b

    # Dot product of normalized vectors = cosine similarity
    return normed_a @ normed_b.T


def compare_sentences(
    prev_text: str,
    curr_text: str,
    match_threshold: float = MATCH_THRESHOLD,
    change_threshold: float = CHANGE_THRESHOLD,
) -> dict:
    """Compare two section texts at the sentence level.

    Returns:
        {
            "added": [{"index": int, "text": str}],
            "removed": [{"index": int, "text": str}],
            "changed": [{"prev_index": int, "curr_index": int,
                         "prev_text": str, "curr_text": str,
                         "similarity": float}],
            "unchanged_count": int,
            "stats": {"total_prev": int, "total_curr": int,
                      "added": int, "removed": int, "changed": int}
        }
    """
    prev_sentences = sentence_split(prev_text)
    curr_sentences = sentence_split(curr_text)

    if not prev_sentences and not curr_sentences:
        return _empty_result()
    if not prev_sentences:
        return {
            "added": [{"index": i, "text": s} for i, s in enumerate(curr_sentences)],
            "removed": [],
            "changed": [],
            "unchanged_count": 0,
            "stats": _stats(0, len(curr_sentences), len(curr_sentences), 0, 0),
        }
    if not curr_sentences:
        return {
            "added": [],
            "removed": [{"index": i, "text": s} for i, s in enumerate(prev_sentences)],
            "changed": [],
            "unchanged_count": 0,
            "stats": _stats(len(prev_sentences), 0, 0, len(prev_sentences), 0),
        }

    # Embed all sentences
    prev_embeddings = _batch_encode_sentences(prev_sentences)
    curr_embeddings = _batch_encode_sentences(curr_sentences)

    # Build similarity matrix
    sim_matrix = _compute_similarity_matrix(prev_embeddings, curr_embeddings)

    # Greedy alignment: for each sentence, find its best match
    # Track which sentences have been claimed
    prev_matched = set()
    curr_matched = set()
    changed_pairs = []
    unchanged_count = 0

    # Process in order of highest similarity first (greedy best-match)
    flat_indices = np.argsort(sim_matrix.ravel())[::-1]
    for flat_idx in flat_indices:
        i = int(flat_idx // sim_matrix.shape[1])
        j = int(flat_idx % sim_matrix.shape[1])
        sim = float(sim_matrix[i, j])

        # Stop once we're below the change threshold — remaining are unmatched
        if sim < change_threshold:
            break

        if i in prev_matched or j in curr_matched:
            continue

        prev_matched.add(i)
        curr_matched.add(j)

        if sim >= match_threshold:
            unchanged_count += 1
        else:
            # Semantically changed: same topic, different meaning
            changed_pairs.append({
                "prev_index": i,
                "curr_index": j,
                "prev_text": prev_sentences[i],
                "curr_text": curr_sentences[j],
                "similarity": round(sim, 4),
            })

    # Unmatched previous sentences = removed content
    unmatched_prev = [i for i in range(len(prev_sentences)) if i not in prev_matched]
    unmatched_curr = [j for j in range(len(curr_sentences)) if j not in curr_matched]

    removed = [{"index": i, "text": prev_sentences[i]} for i in unmatched_prev]
    added = [{"index": j, "text": curr_sentences[j]} for j in unmatched_curr]

    # Second pass: find likely replacements among unmatched sentences.
    # These are pairs below the change_threshold but still the best available
    # match — suggests the company rephrased something deliberately (euphemisms,
    # softened language, removed specifics). Analysts need to see these links.
    likely_replacements = []
    if unmatched_prev and unmatched_curr:
        sub_matrix = sim_matrix[np.ix_(unmatched_prev, unmatched_curr)]
        used_prev = set()
        used_curr = set()
        flat = np.argsort(sub_matrix.ravel())[::-1]
        for flat_idx in flat:
            pi = int(flat_idx // sub_matrix.shape[1])
            ci = int(flat_idx % sub_matrix.shape[1])
            sim = float(sub_matrix[pi, ci])

            # Below 0.25 there's no meaningful relationship
            if sim < 0.25:
                break
            if pi in used_prev or ci in used_curr:
                continue

            used_prev.add(pi)
            used_curr.add(ci)
            likely_replacements.append({
                "prev_index": unmatched_prev[pi],
                "curr_index": unmatched_curr[ci],
                "prev_text": prev_sentences[unmatched_prev[pi]],
                "curr_text": curr_sentences[unmatched_curr[ci]],
                "similarity": round(sim, 4),
            })

    return {
        "added": added,
        "removed": removed,
        "changed": changed_pairs,
        "likely_replacements": likely_replacements,
        "unchanged_count": unchanged_count,
        "stats": _stats(
            len(prev_sentences), len(curr_sentences),
            len(added), len(removed), len(changed_pairs),
        ),
    }


def _stats(total_prev, total_curr, added, removed, changed):
    return {
        "total_prev": total_prev,
        "total_curr": total_curr,
        "added": added,
        "removed": removed,
        "changed": changed,
    }


def _empty_result():
    return {
        "added": [],
        "removed": [],
        "changed": [],
        "likely_replacements": [],
        "unchanged_count": 0,
        "stats": _stats(0, 0, 0, 0, 0),
    }
