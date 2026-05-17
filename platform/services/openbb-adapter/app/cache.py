"""
Cache Redis multinivel para respuestas de OpenBB.

TTLs por tipo de dato:
  - crypto_1m  :  60 s
  - crypto_1h  :  3 600 s
  - crypto_1d  :  3 600 s
  - macro_daily:  86 400 s
  - macro_month:  604 800 s (1 semana)
  - options    :  300 s
  - sec        :  3 600 s
  - cot        :  604 800 s
  - news       :  300 s
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# TTLs en segundos por prefijo de clave
_TTL_MAP: dict[str, int] = {
    "crypto:1m":      60,
    "crypto:5m":      300,
    "crypto:1h":      3_600,
    "crypto:1d":      3_600,
    "macro:daily":    86_400,
    "macro:monthly":  604_800,
    "options":        300,
    "futures":        300,
    "sec":            3_600,
    "cot":            604_800,
    "news":           300,
    "yield_curve":    14_400,
    "vix":            300,
}

_DEFAULT_TTL = 300


def _ttl_for_key(key: str) -> int:
    """Determina el TTL apropiado basado en el prefijo de la clave."""
    key_lower = key.replace(":", "_").lower()
    for prefix, ttl in _TTL_MAP.items():
        if prefix.replace(":", "_") in key_lower:
            return ttl
    return _DEFAULT_TTL


class ResponseCache:
    """
    Cache Redis para respuestas de OpenBB.

    Serializa a JSON; los valores son siempre list[dict] | dict | float | str.
    Maneja gracefully la ausencia de Redis (devuelve None en todos los gets).
    """

    def __init__(self, redis: Any):
        """
        Parameters
        ----------
        redis : aioredis.Redis | None
            Conexión aioredis. Si es None, el cache está deshabilitado.
        """
        self._redis = redis

    @property
    def available(self) -> bool:
        return self._redis is not None

    async def get(self, key: str) -> Optional[Any]:
        if not self._redis:
            return None
        try:
            raw = await self._redis.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as exc:
            logger.debug("cache.get_error key=%s: %s", key, exc)
            return None

    async def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        if not self._redis:
            return False
        if ttl is None:
            ttl = _ttl_for_key(key)
        try:
            await self._redis.setex(key, ttl, json.dumps(data, default=str))
            return True
        except Exception as exc:
            logger.debug("cache.set_error key=%s: %s", key, exc)
            return False

    async def delete(self, key: str) -> None:
        if not self._redis:
            return
        try:
            await self._redis.delete(key)
        except Exception:
            pass

    async def exists(self, key: str) -> bool:
        if not self._redis:
            return False
        try:
            return bool(await self._redis.exists(key))
        except Exception:
            return False
