"""
Post-deploy smoke — run after ``cd platform && make up``:

    pytest -xvs tools/smoke/test_post_deploy_smoke.py

Skips automatically when the stack is not reachable (CI without Docker).
Set ``SMOKE_FORCE=1`` to fail instead of skip when services are down.
"""
from __future__ import annotations

import asyncio
import os
import sys
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from tools.smoke.alerts_check import count_up_targets
from tools.smoke.config import (
    DEFAULT_DB_TABLES,
    HTTP_TIMEOUT_S,
    KAFKA_BOOTSTRAP,
    MIGRATIONS_DIR,
    POSTGRES_DSN,
    PROMETHEUS_URL,
    load_expected_kafka_topics,
    load_expected_services,
)
from tools.smoke.db_schema_check import (
    applied_migration_version,
    latest_migration_prefix,
    migration_artifacts_present,
    table_exists,
)
from tools.smoke.service_health import check_service_health

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXEC_ENGINE = _REPO_ROOT / "platform" / "services" / "execution-engine"
_SHARED = _REPO_ROOT / "shared"
for _p in (_EXEC_ENGINE, _SHARED):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _stack_reachable() -> bool:
    services = load_expected_services()
    if not services:
        return False
    try:
        httpx.get(services[0][1], timeout=1.0)
        return True
    except httpx.HTTPError:
        return False


_FORCE = os.getenv("SMOKE_FORCE", "").lower() in ("1", "true", "yes")
_SKIP_REASON = "stack not reachable — run `cd platform && make up` first"
pytestmark = pytest.mark.skipif(
    not _FORCE and not _stack_reachable(),
    reason=_SKIP_REASON,
)


@pytest.mark.parametrize("service,url", load_expected_services())
def test_service_responds_healthy(service: str, url: str) -> None:
    check_service_health(service, url, timeout=HTTP_TIMEOUT_S)


def test_kafka_topics_exist() -> None:
    pytest.importorskip("confluent_kafka")
    from tools.smoke.kafka_topics_check import missing_kafka_topics as _missing

    expected = load_expected_kafka_topics()
    missing = _missing(KAFKA_BOOTSTRAP, expected)
    assert not missing, f"Missing Kafka topics: {sorted(missing)}"


@pytest.mark.parametrize("schema,table", DEFAULT_DB_TABLES)
def test_db_table_exists(schema: str, table: str) -> None:
    pytest.importorskip("psycopg2")
    assert table_exists(POSTGRES_DSN, schema, table), f"Missing: {schema}.{table}"


def test_alpaca_paper_account_reachable() -> None:
    """Connect to Alpaca paper and verify equity > 0."""
    if not os.getenv("ALPACA_API_KEY"):
        pytest.skip("ALPACA_API_KEY not set")

    from app.brokers.alpaca import AlpacaAdapter, AlpacaConfig  # type: ignore[import-not-found]

    cfg = AlpacaConfig(paper=True)

    async def _check() -> None:
        async with AlpacaAdapter(cfg) as broker:
            account = await broker.get_account()
            assert account.is_paper is True, "Account is NOT paper — refuse to run live"
            assert account.equity > Decimal("0"), f"Equity is {account.equity}"

    asyncio.run(_check())


def test_prometheus_scrapes_targets() -> None:
    try:
        up, total = count_up_targets(PROMETHEUS_URL, timeout=HTTP_TIMEOUT_S)
    except httpx.HTTPError as exc:
        pytest.skip(f"Prometheus not reachable at {PROMETHEUS_URL}: {exc}")
    n_services = len(load_expected_services())
    assert up >= n_services - 1, (
        f"Only {up}/{total} Prometheus targets up (expected >={n_services - 1})"
    )


def test_no_pending_migrations() -> None:
    pytest.importorskip("psycopg2")
    expected_last = latest_migration_prefix(str(MIGRATIONS_DIR))
    if expected_last is None:
        pytest.skip("No migrations found")
    applied = applied_migration_version(POSTGRES_DSN)
    if applied is not None:
        assert int(applied) >= int(expected_last), (
            f"Pending migration: applied={applied}, expected>={expected_last}"
        )
        return
    assert migration_artifacts_present(POSTGRES_DSN, expected_last), (
        f"Migration artifacts for <= {expected_last} missing — run "
        "`cd platform && make db-migrate`"
    )


def test_smoke_detects_dead_service(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate connection refused — error must name the service."""

    def mock_get(*_a: object, **_k: object) -> None:
        raise httpx.ConnectError("Connection refused")

    monkeypatch.setattr(httpx, "get", mock_get)
    with pytest.raises(httpx.ConnectError, match="market-intelligence unreachable"):
        check_service_health(
            "market-intelligence",
            "http://localhost:8001/health",
            timeout=1.0,
        )
