"""Build MetricsCollector with Postgres + Prometheus when available."""
from __future__ import annotations

import os

from .metrics_collector import MetricsCollector, StubMetricsBackend
from .postgres_backend import PostgresMetricsBackend

_DEFAULT_DSN = "postgresql://trading:trading@localhost:5432/trading_db"
_DEFAULT_PROM = "http://localhost:9090"


def build_collector() -> MetricsCollector:
    """Prefer Postgres backend; fall back to stub on connection failure."""
    try:
        import psycopg2  # type: ignore[import-untyped]
    except ImportError:
        return MetricsCollector(StubMetricsBackend())

    dsn = os.getenv("POSTGRES_DSN", _DEFAULT_DSN)
    prom = os.getenv("PROMETHEUS_URL", _DEFAULT_PROM)
    try:
        with psycopg2.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception:
        return MetricsCollector(StubMetricsBackend())
    return MetricsCollector(
        PostgresMetricsBackend(dsn, prom_url=prom),
    )
