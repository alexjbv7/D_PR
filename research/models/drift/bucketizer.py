"""
BucketEdgesRepository — persists PSI bucket edges in Redis with TTL.

Key format : drift:bucket_edges:{model_version}:{feature_name}
Default TTL: 30 days (see ADR-030)

Anti-leakage guarantee
----------------------
Bucket edges are computed once from training data at model-promotion time
and stored here.  Before computing PSI on recent data, the drift cron
loads these edges via :meth:`load`.  This guarantees that the PSI bucket
structure never adapts to recent data, which would suppress the PSI
signal entirely.

Usage
-----
>>> import redis
>>> r = redis.Redis(host="localhost", port=6379)
>>> repo = BucketEdgesRepository(r)
>>> edges = np.percentile(train_values, np.linspace(0, 100, 11))
>>> edges[0] = -np.inf; edges[-1] = np.inf
>>> repo.save("xgb_swing_v3", "rsi_14", edges)
>>> loaded = repo.load("xgb_swing_v3", "rsi_14")
"""
from __future__ import annotations

import json

import numpy as np

__all__ = ["BucketEdgesRepository"]

_TTL_SECONDS: int = 30 * 24 * 3600  # 30 days


class BucketEdgesRepository:
    """Sync Redis-backed repository for PSI bucket edges.

    Parameters
    ----------
    redis_client : redis.Redis (sync, already connected)
    ttl_seconds  : int
        Time-to-live for each key.  Defaults to 30 days.
    """

    def __init__(self, redis_client: object, ttl_seconds: int = _TTL_SECONDS) -> None:
        self._r   = redis_client
        self._ttl = ttl_seconds

    # ------------------------------------------------------------------
    # Key helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(model_version: str, feature_name: str) -> str:
        return f"drift:bucket_edges:{model_version}:{feature_name}"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self, model_version: str, feature_name: str, edges: np.ndarray
    ) -> None:
        """Serialise and store bucket edges with TTL."""
        key     = self._key(model_version, feature_name)
        payload = json.dumps(edges.tolist())
        self._r.setex(key, self._ttl, payload)  # type: ignore[attr-defined]

    def load(
        self, model_version: str, feature_name: str
    ) -> np.ndarray | None:
        """Load bucket edges.  Returns None if key is absent or expired."""
        key = self._key(model_version, feature_name)
        val = self._r.get(key)  # type: ignore[attr-defined]
        if val is None:
            return None
        return np.array(json.loads(val), dtype=float)

    def delete(self, model_version: str, feature_name: str) -> None:
        """Remove a cached edge set (e.g. after model deprecation)."""
        self._r.delete(self._key(model_version, feature_name))  # type: ignore[attr-defined]

    def exists(self, model_version: str, feature_name: str) -> bool:
        """Return True if the key exists and has not expired."""
        return bool(
            self._r.exists(self._key(model_version, feature_name))  # type: ignore[attr-defined]
        )
