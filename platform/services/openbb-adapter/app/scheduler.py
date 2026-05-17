"""
Scheduler de polling — APScheduler con jobs de cron y de intervalo.

Jobs configurados:
  crypto_ohlcv_1m   — cada 60 s (datos 1m)
  crypto_ohlcv_1h   — cada 3 600 s (datos 1h)
  macro_poll        — cada 3 600 s (series FRED)
  yield_curve       — cada 4 h
  funding_rates     — cada 8 h
"""
from __future__ import annotations

import logging

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from .config import Settings
from .producers.crypto_producer import CryptoProducer
from .producers.macro_producer import MacroProducer

logger = logging.getLogger(__name__)


def setup_scheduler(
    settings: Settings,
    macro_producer: MacroProducer,
    crypto_producer: CryptoProducer,
) -> AsyncIOScheduler:
    """
    Configura y retorna el scheduler de APScheduler.
    No lo inicia — llamar scheduler.start() en el lifespan.
    """
    scheduler = AsyncIOScheduler(timezone="UTC")
    symbols = settings.crypto_symbol_list

    # ── Macro: series FRED ────────────────────────────────────────────
    scheduler.add_job(
        macro_producer.poll_all_series,
        trigger=IntervalTrigger(seconds=settings.macro_poll_interval),
        id="macro_fred_poll",
        name="Macro FRED series",
        replace_existing=True,
        misfire_grace_time=300,
    )

    # ── Macro: yield curve ────────────────────────────────────────────
    scheduler.add_job(
        macro_producer.poll_yield_curve,
        trigger=IntervalTrigger(seconds=settings.yield_curve_interval),
        id="yield_curve_poll",
        name="Yield curve",
        replace_existing=True,
        misfire_grace_time=600,
    )

    # ── Crypto: OHLCV 1h ─────────────────────────────────────────────
    scheduler.add_job(
        crypto_producer.poll_ohlcv,
        trigger=IntervalTrigger(seconds=3600),
        id="crypto_ohlcv_1h",
        name="Crypto OHLCV 1h",
        args=[symbols, "1h"],
        replace_existing=True,
        misfire_grace_time=300,
    )

    # ── Crypto: OHLCV 1d ─────────────────────────────────────────────
    scheduler.add_job(
        crypto_producer.poll_ohlcv,
        trigger=IntervalTrigger(seconds=86400),
        id="crypto_ohlcv_1d",
        name="Crypto OHLCV 1d",
        args=[symbols, "1d"],
        replace_existing=True,
        misfire_grace_time=3600,
    )

    # ── Crypto: Funding rates ─────────────────────────────────────────
    scheduler.add_job(
        crypto_producer.poll_funding_rates,
        trigger=IntervalTrigger(seconds=settings.funding_poll_interval),
        id="funding_rates_poll",
        name="Funding rates",
        args=[symbols],
        replace_existing=True,
        misfire_grace_time=600,
    )

    logger.info(
        "scheduler.configured jobs=%d symbols=%s",
        len(scheduler.get_jobs()),
        symbols,
    )
    return scheduler
