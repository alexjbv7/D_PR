"""Tests for Settings (pydantic-settings env loading)."""
from __future__ import annotations

import pytest

from app.settings import Settings


def test_defaults_when_no_env(monkeypatch):
    # Wipe relevant env vars so we exercise the defaults
    for var in (
        "POSTGRES_DSN", "KAFKA_BOOTSTRAP_SERVERS", "ALPACA_ENABLED",
        "CCXT_ENABLED", "RISK_REQUIRE_PAPER",
    ):
        monkeypatch.delenv(var, raising=False)

    s = Settings(_env_file=None)
    assert s.service_name           == "execution-engine"
    assert s.port                   == 8010
    assert s.kafka_signal_topic     == "los_ojos.signals.trading"
    assert s.kafka_result_topic     == "los_ojos.execution.result"
    assert s.alpaca_paper           is True
    assert s.ccxt_testnet           is True
    assert s.risk_require_paper     is True
    assert s.reconciler_interval_sec == 60


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("PORT", "9999")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("RISK_PER_SYMBOL_CAP_PCT", "0.08")

    s = Settings(_env_file=None)
    assert s.port                    == 9999
    assert s.alpaca_paper            is False
    assert s.risk_per_symbol_cap_pct == 0.08


def test_alpaca_creds_optional(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY",    raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)

    s = Settings(_env_file=None)
    assert s.alpaca_api_key    is None
    assert s.alpaca_api_secret is None
