"""Keyphrase extraction and comparison using TF-IDF and KeyBERT.

Replaces the old n-gram frequency + hardcoded blocklist approach with
data-driven methods: TF-IDF for corpus-aware scoring and KeyBERT for
semantic keyphrase extraction.
"""

import json
import logging
import math
import re
import threading
from collections import Counter
from pathlib import Path

from lexdrift.config import settings
from lexdrift.nlp.tokenizer import tokenize

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Priority phrases (curated watchlist — always checked)
# ---------------------------------------------------------------------------

_priority_phrases: set[str] = set()
_loaded = False


def load_priority_phrases(path: Path | None = None) -> set[str]:
    """Load high-priority phrases from the default config file.

    These are phrases that ALWAYS trigger alerts on appearance/disappearance,
    independent of the automatic keyphrase discovery.
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


def check_watchlist_phrases(
    prev_text: str,
    curr_text: str,
    extra_phrases: set[str] | None = None,
) -> dict[str, list[str]]:
    """Check curated watchlist phrases for changes between filings.

    This is the renamed version of check_priority_phrases — makes it clear
    these are a curated watchlist, not the main detection method.
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


# Keep the old name as an alias for backward compatibility
check_priority_phrases = check_watchlist_phrases


# ---------------------------------------------------------------------------
# Corpus-level document frequency for TF-IDF
# ---------------------------------------------------------------------------

_CORPUS_DF_PATH = Path("data/corpus_df.json")
_corpus_df: Counter = Counter()
_corpus_doc_count: int = 0
_corpus_lock = threading.Lock()
_corpus_loaded = False


def _load_corpus_df() -> None:
    """Load the persisted corpus document-frequency counter from disk."""
    global _corpus_df, _corpus_doc_count, _corpus_loaded
    if _corpus_loaded:
        return
    with _corpus_lock:
        if _corpus_loaded:
            return
        if _CORPUS_DF_PATH.exists():
            try:
                data = json.loads(_CORPUS_DF_PATH.read_text())
                _corpus_df = Counter(data.get("df", {}))
                _corpus_doc_count = data.get("doc_count", 0)
                logger.info(
                    "Loaded corpus DF: %d unique n-grams across %d documents",
                    len(_corpus_df), _corpus_doc_count,
                )
            except Exception:
                logger.warning("Failed to load corpus DF from %s", _CORPUS_DF_PATH, exc_info=True)
        _corpus_loaded = True


def _save_corpus_df() -> None:
    """Persist the corpus DF counter to disk."""
    _CORPUS_DF_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {"df": dict(_corpus_df), "doc_count": _corpus_doc_count}
    _CORPUS_DF_PATH.write_text(json.dumps(data))


def _extract_ngrams(tokens: list[str], n: int) -> list[str]:
    """Extract n-grams from a token list."""
    return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def _get_ngram_tf(text: str) -> Counter:
    """Compute term-frequency counter for bigrams and trigrams in a text."""
    tokens = tokenize(text)
    tf: Counter = Counter()
    for n in (2, 3):
        tf.update(_extract_ngrams(tokens, n))
    return tf


def update_corpus(text: str) -> None:
    """Add a filing's n-grams to the corpus document-frequency counter.

    Call this once per filing to build up corpus statistics. Each unique
    n-gram in the filing increments its DF by 1 (regardless of how many
    times it appears in this particular filing).
    """
    global _corpus_doc_count
    _load_corpus_df()

    tf = _get_ngram_tf(text)
    # DF counts unique n-grams per document (presence, not frequency)
    unique_ngrams = set(tf.keys())
    with _corpus_lock:
        for ng in unique_ngrams:
            _corpus_df[ng] += 1
        _corpus_doc_count += 1
        _save_corpus_df()


def extract_keyphrases_tfidf(text: str, top_k: int = 20) -> list[dict]:
    """Extract top keyphrases from text using TF-IDF scoring.

    TF = frequency of n-gram in this filing
    IDF = log(total_docs / (1 + doc_frequency_of_ngram))

    High TF-IDF means the phrase is frequent in THIS filing but rare
    across the corpus — mathematically filters boilerplate without any
    hardcoded lists.

    Returns:
        List of dicts with keys: phrase, score, tf, idf
    """
    _load_corpus_df()
    tf = _get_ngram_tf(text)
    if not tf:
        return []

    total_tokens = sum(tf.values())
    doc_count = max(_corpus_doc_count, 1)  # avoid division by zero

    scored: list[dict] = []
    for ngram, count in tf.items():
        tf_norm = count / total_tokens
        df = _corpus_df.get(ngram, 0)
        idf = math.log(doc_count / (1 + df))
        tfidf = tf_norm * idf
        scored.append({
            "phrase": ngram,
            "score": round(tfidf, 6),
            "tf": count,
            "idf": round(idf, 4),
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# KeyBERT semantic keyphrase extraction
# ---------------------------------------------------------------------------

_keybert_model = None
_keybert_lock = threading.Lock()


def _get_keybert():
    """Lazy-load the KeyBERT model (thread-safe)."""
    global _keybert_model
    if _keybert_model is not None:
        return _keybert_model
    with _keybert_lock:
        if _keybert_model is not None:
            return _keybert_model
        from keybert import KeyBERT
        logger.info("Loading KeyBERT model (uses sentence-transformers)")
        _keybert_model = KeyBERT(model=settings.embedding_model)
        logger.info("KeyBERT model loaded")
    return _keybert_model


def extract_keyphrases_semantic(text: str, top_k: int = 10) -> list[dict]:
    """Extract keyphrases using KeyBERT (semantic / model-driven).

    Uses sentence-transformer embeddings to find phrases most
    representative of the document's meaning.

    Returns:
        List of dicts with keys: phrase, score
    """
    if not text or not text.strip():
        return []

    try:
        kw_model = _get_keybert()
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(2, 3),
            stop_words="english",
            top_n=top_k,
            use_mmr=True,
            diversity=0.5,
        )
        return [{"phrase": kw, "score": round(score, 4)} for kw, score in keywords]
    except Exception:
        logger.warning("KeyBERT extraction failed; returning empty list", exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Keyphrase comparison between filings
# ---------------------------------------------------------------------------

def _phrase_sets(keyphrases: list[dict]) -> dict[str, float]:
    """Convert keyphrase list to {phrase: score} dict."""
    return {kp["phrase"]: kp.get("score", 0.0) for kp in keyphrases}


def _semantic_overlap(phrases_a: set[str], phrases_b: set[str], threshold: float = 0.6) -> dict[str, str]:
    """Find semantically similar phrases between two sets.

    Returns a mapping from phrase_b -> phrase_a for phrases that are
    semantically similar (above threshold) using token-level Jaccard.
    This is a lightweight proxy; for full semantic matching, use embeddings.
    """
    matches: dict[str, str] = {}
    for pb in phrases_b:
        tokens_b = set(pb.lower().split())
        best_sim = 0.0
        best_match = None
        for pa in phrases_a:
            tokens_a = set(pa.lower().split())
            if not tokens_a or not tokens_b:
                continue
            jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
            if jaccard > best_sim:
                best_sim = jaccard
                best_match = pa
        if best_match and best_sim >= threshold:
            matches[pb] = best_match
    return matches


def compare_keyphrases(
    prev_text: str,
    curr_text: str,
    top_k_tfidf: int = 20,
    top_k_semantic: int = 10,
) -> dict:
    """Compare keyphrases between two filings using TF-IDF and semantic matching.

    Returns:
        {
            appeared: [{phrase, score}] — in current but not previous
            disappeared: [{phrase, score}] — in previous but not current
            intensified: [{phrase, prev_score, curr_score}] — higher TF-IDF in current
            diminished: [{phrase, prev_score, curr_score}] — lower TF-IDF in current
            semantic_prev: [{phrase, score}] — KeyBERT results for prev
            semantic_curr: [{phrase, score}] — KeyBERT results for curr
        }
    """
    # TF-IDF keyphrases for both texts
    prev_tfidf = extract_keyphrases_tfidf(prev_text, top_k=top_k_tfidf)
    curr_tfidf = extract_keyphrases_tfidf(curr_text, top_k=top_k_tfidf)

    prev_scores = _phrase_sets(prev_tfidf)
    curr_scores = _phrase_sets(curr_tfidf)

    prev_phrases = set(prev_scores.keys())
    curr_phrases = set(curr_scores.keys())

    # Find semantic overlap for phrases present in both (by token similarity)
    overlap = _semantic_overlap(prev_phrases, curr_phrases)

    # Exact matches
    exact_common = prev_phrases & curr_phrases

    # Purely new / gone (not matched semantically or exactly)
    matched_curr = set(overlap.keys()) | exact_common
    matched_prev = set(overlap.values()) | exact_common

    appeared = [
        {"phrase": p, "score": curr_scores[p]}
        for p in sorted(curr_phrases - matched_curr, key=lambda x: curr_scores[x], reverse=True)
    ]

    disappeared = [
        {"phrase": p, "score": prev_scores[p]}
        for p in sorted(prev_phrases - matched_prev, key=lambda x: prev_scores[x], reverse=True)
    ]

    # Intensified / diminished — phrases present in both (exact or semantic match)
    intensified = []
    diminished = []

    for phrase in exact_common:
        prev_s = prev_scores[phrase]
        curr_s = curr_scores[phrase]
        entry = {"phrase": phrase, "prev_score": prev_s, "curr_score": curr_s}
        if curr_s > prev_s * 1.1:  # 10% threshold to avoid noise
            intensified.append(entry)
        elif curr_s < prev_s * 0.9:
            diminished.append(entry)

    for curr_p, prev_p in overlap.items():
        if curr_p in exact_common:
            continue  # already handled
        prev_s = prev_scores.get(prev_p, 0)
        curr_s = curr_scores.get(curr_p, 0)
        label = f"{curr_p} (was: {prev_p})" if curr_p != prev_p else curr_p
        entry = {"phrase": label, "prev_score": prev_s, "curr_score": curr_s}
        if curr_s > prev_s * 1.1:
            intensified.append(entry)
        elif curr_s < prev_s * 0.9:
            diminished.append(entry)

    intensified.sort(key=lambda x: x["curr_score"], reverse=True)
    diminished.sort(key=lambda x: x["prev_score"], reverse=True)

    # Semantic keyphrases (KeyBERT) — auxiliary enrichment
    try:
        semantic_prev = extract_keyphrases_semantic(prev_text, top_k=top_k_semantic)
        semantic_curr = extract_keyphrases_semantic(curr_text, top_k=top_k_semantic)
    except Exception:
        logger.warning("KeyBERT extraction failed during comparison", exc_info=True)
        semantic_prev = []
        semantic_curr = []

    return {
        "appeared": appeared,
        "disappeared": disappeared,
        "intensified": intensified,
        "diminished": diminished,
        "semantic_prev": semantic_prev,
        "semantic_curr": semantic_curr,
    }


def compare_phrases(
    prev_text: str,
    curr_text: str,
    extra_phrases: set[str] | None = None,
) -> dict:
    """Full phrase comparison: watchlist phrases + TF-IDF/KeyBERT keyphrase discovery.

    Backward-compatible wrapper that returns the same top-level structure
    as the old compare_phrases (priority + discovered) but uses the new
    data-driven approach for the discovered portion.

    Returns:
        {
            priority: {appeared, disappeared, persisted},
            discovered: {appeared: [{phrase, score}], disappeared: [{phrase, score}]},
            keyphrase_changes: {appeared, disappeared, intensified, diminished, ...}
        }
    """
    priority = check_watchlist_phrases(prev_text, curr_text, extra_phrases)
    keyphrase_changes = compare_keyphrases(prev_text, curr_text)

    # Map new keyphrase results into the old "discovered" format for backward compat
    discovered = {
        "appeared": keyphrase_changes["appeared"],
        "disappeared": keyphrase_changes["disappeared"],
    }

    return {
        "priority": priority,
        "discovered": discovered,
        "keyphrase_changes": keyphrase_changes,
    }
