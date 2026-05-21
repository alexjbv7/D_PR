"""Prometheus scrape-target health checks."""
from __future__ import annotations

from typing import Any

import httpx


def count_up_targets(prom_url: str, *, timeout: float = 5.0) -> tuple[int, int]:
    """Return (up_count, total_active_targets) from Prometheus API."""
    url = prom_url.rstrip("/") + "/api/v1/targets"
    response = httpx.get(url, timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    targets: list[dict[str, Any]] = payload["data"]["activeTargets"]
    up = sum(1 for t in targets if t.get("health") == "up")
    return up, len(targets)
