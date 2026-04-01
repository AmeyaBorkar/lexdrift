import logging
import threading

import numpy as np

from lexdrift.config import settings

logger = logging.getLogger(__name__)

_model = None
_embedding_dim: int | None = None
_model_lock = threading.Lock()


def _get_model():
    """Lazy-load the sentence-transformer model (thread-safe)."""
    global _model, _embedding_dim
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:  # double-check after acquiring lock
            return _model
        from sentence_transformers import SentenceTransformer

        logger.info(f"Loading embedding model: {settings.embedding_model}")
        _model = SentenceTransformer(settings.embedding_model)
        _embedding_dim = _model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {_embedding_dim}")
    return _model


def get_embedding_dim() -> int:
    """Get the embedding dimension from the loaded model."""
    if _embedding_dim is None:
        _get_model()
    return _embedding_dim


def _chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """Split text into overlapping chunks.

    For short texts (< chunk_size), returns the text as-is.
    For long texts, creates overlapping windows so no content is lost.
    """
    chunk_size = chunk_size or settings.embedding_chunk_size
    overlap = overlap or settings.embedding_chunk_overlap
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def encode_text(text: str) -> np.ndarray:
    """Encode text into an embedding vector using chunked mean-pooling.

    For short texts, encodes directly.
    For long texts (common in SEC filings), splits into overlapping chunks,
    encodes each, and mean-pools the results. This captures semantics from
    the entire section rather than truncating.
    """
    model = _get_model()
    chunks = _chunk_text(text)

    if len(chunks) == 1:
        embedding = model.encode(chunks[0], show_progress_bar=False)
        return embedding.astype(np.float32)

    # Encode all chunks in batch (more efficient than one-by-one)
    embeddings = model.encode(chunks, show_progress_bar=False, batch_size=32)

    # Mean-pool across chunks
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding.astype(np.float32)


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Serialize a numpy embedding to bytes for DB storage."""
    return embedding.tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """Deserialize bytes back to a numpy embedding."""
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - similarity). Higher = more different."""
    return 1.0 - cosine_similarity(a, b)
