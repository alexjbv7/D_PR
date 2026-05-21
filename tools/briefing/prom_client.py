"""Minimal Prometheus HTTP client for briefing metrics."""
from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx


def query_p99_seconds(
    prom_url: str,
    metric: str,
    start: datetime,
    end: datetime,
    *,
    timeout: float = 10.0,
) -> float:
    """Return p99 of *metric* over [start, end] in seconds (0.0 if unavailable)."""
    query = (
        f"histogram_quantile(0.99, "
        f"sum(rate({metric}_bucket[5m])) by (le))"
    )
    params: dict[str, str | float] = {
        "query": query,
        "start": start.timestamp(),
        "end": end.timestamp(),
        "step": 300,
    }
    url = prom_url.rstrip("/") + "/api/v1/query_range"
    try:
        response = httpx.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        if payload.get("status") != "success":
            return 0.0
        result = payload.get("data", {}).get("result", [])
        if not result:
            return 0.0
        values = result[0].get("values", [])
        if not values:
            return 0.0
        samples = [float(v[1]) for v in values if v[1] not in ("NaN", "nan")]
        return max(samples) if samples else 0.0
    except (httpx.HTTPError, ValueError, KeyError, IndexError):
        return 0.0
