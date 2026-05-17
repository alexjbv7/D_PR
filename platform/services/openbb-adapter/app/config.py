"""
Configuración del openbb-adapter via variables de entorno.

Todas las API keys se leen de env; nunca se hardcodean.
Usa pydantic-settings para validación automática.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenBB providers ──────────────────────────────────────────────
    fred_api_key:           str = ""
    fmp_api_key:            str = ""
    polygon_api_key:        str = ""
    coinmarketcap_api_key:  str = ""
    openbb_pat:             str = ""  # Personal Access Token (OpenBB Hub)

    # ── Infraestructura ───────────────────────────────────────────────
    kafka_servers: str = "localhost:9092"
    redis_url:     str = "redis://localhost:6379/0"
    redis_db:      int = 0

    # ── Intervals de polling (segundos) ──────────────────────────────
    crypto_poll_interval:  int = 60       # 1 min — OHLCV crypto
    macro_poll_interval:   int = 3600     # 1 h  — series FRED
    funding_poll_interval: int = 28800    # 8 h  — funding rates
    news_poll_interval:    int = 300      # 5 min — noticias
    options_poll_interval: int = 300      # 5 min — cadena de opciones
    yield_curve_interval:  int = 14400   # 4 h  — yield curve

    # ── Símbolos a monitorizar ────────────────────────────────────────
    crypto_symbols: str = "BTC,ETH,SOL,BNB,XRP"
    equity_symbols: str = "IBIT,GBTC,MSTR,COIN"  # ETFs + equities cripto

    # ── Server ────────────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8009
    log_level: str = "INFO"

    # ── OpenBB ────────────────────────────────────────────────────────
    openbb_disable_telemetry: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def crypto_symbol_list(self) -> list[str]:
        return [s.strip() for s in self.crypto_symbols.split(",") if s.strip()]

    @property
    def equity_symbol_list(self) -> list[str]:
        return [s.strip() for s in self.equity_symbols.split(",") if s.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
