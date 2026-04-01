"""Boilerplate detection via cross-company deduplication.

The core problem: when a new accounting standard takes effect, hundreds of
companies all add nearly identical language. These changes drown out the
company-specific signals analysts actually care about.

Approach:
1. Maintain a rolling index of sentence embeddings from recently analyzed filings.
2. When scoring a new filing's changes, check each flagged sentence against the
   index. If many companies added a similar sentence recently, it's boilerplate.
3. Assign a "uniqueness score" — 1.0 = only this company, 0.0 = everyone added it.

This works because boilerplate text is nearly identical across companies
(lawyers copy-paste), so embedding similarity catches it reliably.
"""

import logging
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# In-memory index of recently added sentence embeddings.
# Maps embedding vectors to the set of company IDs that added them.
# In production this would be a vector database (Pinecone, Qdrant, etc.)
_sentence_index: list[dict] = []
# { "embedding": np.ndarray, "company_id": int, "text": str, "filing_quarter": str }

MAX_INDEX_SIZE = 50000


def add_to_index(
    sentences: list[str],
    embeddings: np.ndarray,
    company_id: int,
    filing_quarter: str,
) -> None:
    """Add analyzed sentences to the cross-company index.

    Called after each filing analysis so future filings can check against it.
    """
    global _sentence_index

    for i, sent in enumerate(sentences):
        _sentence_index.append({
            "embedding": embeddings[i],
            "company_id": company_id,
            "text": sent,
            "filing_quarter": filing_quarter,
        })

    # Evict oldest entries if index is too large
    if len(_sentence_index) > MAX_INDEX_SIZE:
        _sentence_index = _sentence_index[-MAX_INDEX_SIZE:]

    logger.debug(f"Index size: {len(_sentence_index)} sentences from cross-company filings")


def compute_uniqueness(
    sentence_embedding: np.ndarray,
    company_id: int,
    similarity_threshold: float = 0.90,
) -> dict:
    """Check how unique a sentence is across all companies in the index.

    Returns:
        {
            "uniqueness_score": float (1.0 = unique to this company, 0.0 = everyone has it),
            "companies_with_similar": int (count of OTHER companies with similar sentence),
            "is_boilerplate": bool,
        }
    """
    if not _sentence_index:
        # No cross-company data yet — assume unique
        return {"uniqueness_score": 1.0, "companies_with_similar": 0, "is_boilerplate": False}

    # Compute similarities against the index
    index_embeddings = np.array([e["embedding"] for e in _sentence_index])

    # Normalize
    norm_query = sentence_embedding / (np.linalg.norm(sentence_embedding) + 1e-8)
    norms = np.linalg.norm(index_embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normed_index = index_embeddings / norms

    similarities = normed_index @ norm_query

    # Find entries above threshold from OTHER companies
    other_company_matches = set()
    for i, sim in enumerate(similarities):
        if sim >= similarity_threshold and _sentence_index[i]["company_id"] != company_id:
            other_company_matches.add(_sentence_index[i]["company_id"])

    n_similar = len(other_company_matches)

    # Uniqueness score: inverse of how many companies share this sentence
    # 0 companies = 1.0 (unique), 1 company = 0.8, 5 = 0.4, 10+ = near 0
    if n_similar == 0:
        uniqueness = 1.0
    else:
        uniqueness = max(0.0, 1.0 - (n_similar / 10.0))

    is_boilerplate = n_similar >= 3

    return {
        "uniqueness_score": round(uniqueness, 4),
        "companies_with_similar": n_similar,
        "is_boilerplate": is_boilerplate,
    }


def filter_boilerplate(
    scored_changes: dict,
    sentence_embeddings: dict[int, np.ndarray],
    company_id: int,
) -> dict:
    """Enrich scored sentence changes with boilerplate detection.

    Takes the output of risk.score_changes() and adds uniqueness scores.
    Sentences flagged as boilerplate get their risk level downgraded.

    Args:
        scored_changes: Output from risk.score_changes()
        sentence_embeddings: Map of sentence index -> embedding (from batch encoding)
        company_id: Current company's ID
    """
    for entry in scored_changes.get("added", []):
        idx = entry.get("index")
        if idx is not None and idx in sentence_embeddings:
            uniqueness = compute_uniqueness(sentence_embeddings[idx], company_id)
            entry["uniqueness"] = uniqueness
            if uniqueness["is_boilerplate"]:
                entry["risk"]["level"] = "boilerplate"
                entry["risk"]["score"] = 0.05

    for entry in scored_changes.get("removed", []):
        idx = entry.get("index")
        if idx is not None and idx in sentence_embeddings:
            uniqueness = compute_uniqueness(sentence_embeddings[idx], company_id)
            entry["uniqueness"] = uniqueness
            if uniqueness["is_boilerplate"]:
                entry["risk"]["level"] = "boilerplate"
                entry["risk"]["score"] = 0.05

    # Recompute risk summary after boilerplate filtering
    all_scores = (
        [e["risk"]["score"] for e in scored_changes.get("added", [])]
        + [e["risk"]["score"] for e in scored_changes.get("removed", [])]
        + [e.get("risk_curr", {}).get("score", 0) for e in scored_changes.get("changed", [])]
    )
    non_boilerplate = [s for s in all_scores if s > 0.05]

    scored_changes["risk_summary"] = {
        "max_risk_score": round(max(all_scores) if all_scores else 0, 4),
        "max_risk_level": (
            "critical" if any(s >= 0.9 for s in all_scores) else
            "high" if any(s >= 0.6 for s in all_scores) else
            "medium" if any(s >= 0.3 for s in all_scores) else "low"
        ),
        "critical_changes": sum(1 for s in all_scores if s >= 0.9),
        "high_risk_changes": sum(1 for s in all_scores if 0.6 <= s < 0.9),
        "boilerplate_filtered": len(all_scores) - len(non_boilerplate),
        "substantive_changes": len(non_boilerplate),
    }

    return scored_changes


def clear_index() -> None:
    """Clear the cross-company index (for testing)."""
    global _sentence_index
    _sentence_index = []


# ---------------------------------------------------------------------------
# Trained boilerplate classifier integration
# ---------------------------------------------------------------------------

import threading
from pathlib import Path

_boilerplate_model = None
_use_trained_boilerplate: bool | None = None  # None = not yet checked
_boilerplate_lock = threading.Lock()

TRAINED_BOILERPLATE_PATH = "models/boilerplate_classifier.pt"


def _try_load_boilerplate_model():
    """Attempt to load the trained boilerplate classifier (thread-safe, once).

    Sets ``_use_trained_boilerplate`` to True/False and populates
    ``_boilerplate_model`` if the model file exists and loads successfully.
    """
    global _boilerplate_model, _use_trained_boilerplate
    if _use_trained_boilerplate is not None:
        return
    with _boilerplate_lock:
        if _use_trained_boilerplate is not None:
            return
        model_path = Path(TRAINED_BOILERPLATE_PATH)
        if model_path.exists():
            try:
                from lexdrift.training.boilerplate_classifier import (
                    load_boilerplate_classifier,
                )
                _boilerplate_model = load_boilerplate_classifier(str(model_path))
                _use_trained_boilerplate = True
                logger.info(
                    "Trained boilerplate classifier loaded from %s", model_path
                )
            except Exception:
                logger.warning(
                    "Failed to load trained boilerplate classifier from %s; "
                    "falling back to index-based uniqueness scoring",
                    model_path,
                    exc_info=True,
                )
                _use_trained_boilerplate = False
        else:
            logger.debug(
                "No trained boilerplate classifier found at %s; "
                "using index-based uniqueness scoring",
                model_path,
            )
            _use_trained_boilerplate = False


def classify_boilerplate(sentences: list[str]) -> list[dict]:
    """Classify sentences as boilerplate or substantive.

    Tries to use a trained classifier at ``models/boilerplate_classifier.pt``.
    If the model is unavailable, falls back to the in-memory cross-company
    index via ``compute_uniqueness()`` (which requires prior calls to
    ``add_to_index()``).

    Returns:
        A list of dicts with keys: sentence, is_boilerplate, confidence.
    """
    if not sentences:
        return []

    _try_load_boilerplate_model()

    if _use_trained_boilerplate and _boilerplate_model is not None:
        try:
            from lexdrift.training.boilerplate_classifier import predict_boilerplate

            predictions = predict_boilerplate(
                sentences, model_path=TRAINED_BOILERPLATE_PATH
            )
            return [
                {
                    "sentence": p["sentence"],
                    "is_boilerplate": p["is_boilerplate"],
                    "confidence": p["boilerplate_probability"]
                    if p["is_boilerplate"]
                    else 1.0 - p["boilerplate_probability"],
                }
                for p in predictions
            ]
        except Exception:
            logger.warning(
                "Trained boilerplate classifier inference failed; "
                "falling back to index-based uniqueness scoring",
                exc_info=True,
            )

    # Fallback: use compute_uniqueness() with the in-memory index.
    # This requires embeddings, so we encode the sentences here.
    results = []
    try:
        from lexdrift.nlp.embeddings import encode_text

        for sentence in sentences:
            embedding = encode_text(sentence)
            # company_id=0 means "unknown" -- just checking cross-company duplication
            uniqueness = compute_uniqueness(embedding, company_id=0)
            results.append({
                "sentence": sentence,
                "is_boilerplate": uniqueness["is_boilerplate"],
                "confidence": 1.0 - uniqueness["uniqueness_score"],
            })
    except Exception:
        logger.warning(
            "Index-based boilerplate fallback failed; "
            "returning neutral classification",
            exc_info=True,
        )
        for sentence in sentences:
            results.append({
                "sentence": sentence,
                "is_boilerplate": False,
                "confidence": 0.5,
            })

    return results
