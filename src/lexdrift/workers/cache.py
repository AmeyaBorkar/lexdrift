"""Smart caching for expensive NLP operations.

Caches: embedding results, KeyBERT keyphrases, FinBERT sentiment,
intelligence reports. Uses file-based cache (data/cache/) with
content-hash keys and configurable TTL.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_DIR = Path("data/cache")


def _ensure_cache_dir() -> None:
    """Create cache directory if it doesn't exist."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def cache_key(prefix: str, *args) -> str:
    """Generate a deterministic key from prefix + args hash."""
    raw = prefix + ":" + ":".join(str(a) for a in args)
    return prefix + "_" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def cache_get(key: str) -> Any | None:
    """Return cached result if it exists and has not expired."""
    path = _CACHE_DIR / f"{key}.json"
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.debug("Corrupt cache entry %s, removing", key)
        path.unlink(missing_ok=True)
        return None

    expires_at = data.get("expires_at", 0)
    if time.time() > expires_at:
        logger.debug("Cache entry %s expired, removing", key)
        path.unlink(missing_ok=True)
        return None

    return data.get("value")


def cache_set(key: str, value: Any, ttl_hours: int = 24) -> None:
    """Store result with expiry."""
    _ensure_cache_dir()
    path = _CACHE_DIR / f"{key}.json"
    data = {
        "value": value,
        "created_at": time.time(),
        "expires_at": time.time() + ttl_hours * 3600,
    }
    try:
        path.write_text(json.dumps(data, default=str), encoding="utf-8")
    except (OSError, TypeError):
        logger.warning("Failed to write cache entry %s", key, exc_info=True)


def cache_cleanup() -> int:
    """Remove all expired cache entries. Returns count of entries removed."""
    if not _CACHE_DIR.exists():
        return 0

    removed = 0
    now = time.time()
    for path in _CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if now > data.get("expires_at", 0):
                path.unlink()
                removed += 1
        except (json.JSONDecodeError, OSError, KeyError):
            path.unlink(missing_ok=True)
            removed += 1

    if removed:
        logger.info("Cache cleanup: removed %d expired entries", removed)
    return removed


def cached(prefix: str, ttl_hours: int = 24):
    """Decorator for caching function results.

    Usage:
        @cached("intelligence_report", ttl_hours=1)
        def generate_intelligence(db, ticker):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Build cache key from all arguments (skip db sessions)
            key_args = []
            for a in args:
                # Skip SQLAlchemy sessions and other non-serializable objects
                if hasattr(a, "execute"):
                    continue
                key_args.append(a)
            for k, v in sorted(kwargs.items()):
                if hasattr(v, "execute"):
                    continue
                key_args.append(f"{k}={v}")

            ck = cache_key(prefix, *key_args)
            hit = cache_get(ck)
            if hit is not None:
                logger.debug("Cache hit for %s", ck)
                return hit

            result = func(*args, **kwargs)

            # Only cache JSON-serializable results
            try:
                json.dumps(result, default=str)
                cache_set(ck, result, ttl_hours=ttl_hours)
            except (TypeError, ValueError):
                logger.debug("Result not JSON-serializable, skipping cache for %s", ck)

            return result
        return wrapper
    return decorator
