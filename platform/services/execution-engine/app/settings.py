"""
Typed configuration for the execution-engine service.

All settings come from environment variables (12-factor app — CLAUDE.md §1.4).
A ``.env`` file at the repo root is loaded automatically by pydantic-settings.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the execution-engine."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- service ----
    service_name: str  = "execution-engine"
    log_level:    str  = "INFO"
    port:         int  = 8010

    # ---- database ----
    postgres_dsn: str  = "postgresql://trading:trading@postgres:5432/trading_db"
    postgres_min_size: int = 1
    postgres_max_size: int = 5

    # ---- kafka ----
    kafka_bootstrap_servers: str  = "kafka:29092"
    kafka_signal_topic:      str  = "los_ojos.signals.trading"
    kafka_result_topic:      str  = "los_ojos.execution.result"
    kafka_anomaly_topic:     str  = "los_ojos.context.anomaly"
    kafka_consumer_group:    str  = "execution-engine"

    # ---- redis (kill switch flag) ----
    redis_url:              str = "redis://redis:6379/0"
    redis_kill_switch_key:  str = "execution:kill_switch"

    # ---- alpaca ----
    alpaca_enabled:    bool          = True
    alpaca_api_key:    Optional[str] = None
    alpaca_api_secret: Optional[str] = None
    alpaca_paper:      bool          = True
    alpaca_data_feed:  str           = "iex"    # "iex" (free) or "sip" (paid)

    # ---- ccxt ----
    ccxt_enabled:      bool          = False
    ccxt_exchange:     str           = "binance"
    ccxt_api_key:      Optional[str] = None
    ccxt_api_secret:   Optional[str] = None
    ccxt_testnet:      bool          = True
    ccxt_market_type:  str           = "spot"

    # ---- risk gate ----
    risk_per_symbol_cap_pct:  float = 0.05
    risk_per_venue_cap_pct:   float = 0.50
    risk_daily_dd_kill_pct:   float = 0.03
    risk_min_cash_buffer_pct: float = 0.10
    risk_require_paper:       bool  = True

    # ---- reconciler ----
    reconciler_interval_sec:      int = 60
    reconciler_failure_threshold: int = 3

    # ---- account refresh ----
    account_refresh_sec: int = 30


def get_settings() -> Settings:
    """Singleton-ish helper (FastAPI Depends won't see env updates after start)."""
    return Settings()
