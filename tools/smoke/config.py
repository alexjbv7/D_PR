"""Smoke-test configuration loaded from env + repo paths."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Host ports from platform/docker-compose.yml (host:container).
DEFAULT_SERVICES: list[tuple[str, str]] = [
    ("market-intelligence", "http://localhost:8001/health"),
    ("macroeconomic", "http://localhost:8002/health"),
    ("onchain-analysis", "http://localhost:8003/health"),
    ("context-engine", "http://localhost:8004/health"),
    ("realtime-signal", "http://localhost:8005/health"),
    ("ml-feature-store", "http://localhost:8006/health"),
    ("strategy-orchestrator", "http://localhost:8007/health"),
    ("execution-engine", "http://localhost:8010/health"),
]

DEFAULT_DB_TABLES: list[tuple[str, str]] = [
    ("data", "universe_historical"),
    ("market", "corporate_actions"),
    ("market", "corporate_actions_applied"),
    ("market", "bars_1m_adjusted"),
    ("risk", "allocator_state"),
    ("risk", "allocator_updates"),
    ("risk", "position_actions"),
    ("drift", "psi_history"),
    ("drift", "ece_history"),
    ("audit", "predictions"),
]

# Topics used in code but not yet listed in topics.yml.
_EXTRA_KAFKA_TOPICS: frozenset[str] = frozenset({
    "los_ojos.execution.result",
})

KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
POSTGRES_DSN = os.getenv(
    "POSTGRES_DSN",
    "postgresql://trading:trading@localhost:5432/trading_db",
)
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
HTTP_TIMEOUT_S = float(os.getenv("SMOKE_HTTP_TIMEOUT", "5.0"))
MIGRATIONS_DIR = Path(
    os.getenv(
        "SMOKE_MIGRATIONS_DIR",
        str(_REPO_ROOT / "platform" / "infra" / "sql" / "migrations"),
    ),
)
TOPICS_YAML = Path(
    os.getenv(
        "SMOKE_TOPICS_YAML",
        str(_REPO_ROOT / "infra" / "kafka" / "topics.yml"),
    ),
)


def load_expected_kafka_topics() -> list[str]:
    """Load topic names from infra/kafka/topics.yml plus runtime extras."""
    names: set[str] = set(_EXTRA_KAFKA_TOPICS)
    if TOPICS_YAML.is_file():
        raw: dict[str, Any] = yaml.safe_load(TOPICS_YAML.read_text(encoding="utf-8")) or {}
        for entry in raw.get("topics", []):
            if isinstance(entry, dict) and "name" in entry:
                names.add(str(entry["name"]))
    return sorted(names)


def load_expected_services() -> list[tuple[str, str]]:
    """Return (service_name, health_url) pairs."""
    override = os.getenv("SMOKE_SERVICES_JSON")
    if override:
        import json

        data = json.loads(override)
        return [(str(k), str(v)) for k, v in data.items()]
    return list(DEFAULT_SERVICES)
