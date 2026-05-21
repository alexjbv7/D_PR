"""
S9 pipeline — wires confirmation → allocator → lock → emit.

Public entry point
------------------
``handle_raw_signal(signal, ctx)``

This module is intentionally decoupled from main.py wiring; ``main.py`` will
construct ``S9PipelineContext`` once on startup and call ``handle_raw_signal``
per RawSignalEvent.  Doing so as a standalone module keeps S9 testable in
isolation and avoids touching the existing FastAPI lifespan during this sprint.

MVP scope (S9)
--------------
A RawSignalEvent carries a single horizon; the allocator receives
``{horizon: signal}`` with one element.  The full multi-horizon aggregation
(buffering signals from intraday/swing/daily that fire on the same symbol
within a 5 s window) is left for S11 — see TODO marker below.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Mapping, Optional

import structlog

from .allocator    import ThompsonAllocator, AllocatorDecision
from .confirmation import MultiFactorConfirmation, ConfirmationResult
from .locks        import LockAcquisitionError, SameSymbolDirectionLock

logger = structlog.get_logger(__name__)


# ----------------------------------------------------------------------
# Context bundle (DI)
# ----------------------------------------------------------------------

@dataclass
class S9PipelineContext:
    """Bundle of dependencies passed to ``handle_raw_signal``.

    Constructed once at service startup and reused across signals.
    """
    confirmation: MultiFactorConfirmation
    allocator:    ThompsonAllocator
    lock:         SameSymbolDirectionLock
    kafka_producer: Any
    final_signal_topic: str = "los_ojos.signals.trading"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _is_crypto(symbol: str) -> bool:
    """Crude crypto heuristic (good enough for MVP).

    A symbol is treated as crypto when it ends with USDT/BUSD/USDC or
    starts with X (BTC, ETH paired against USD on most CEXes).  Equities
    use standard ticker symbols without those suffixes.
    """
    s = symbol.upper()
    if s.endswith(("USDT", "BUSD", "USDC", "USD")):
        return True
    return False


def _build_final_signal_payload(
    signal:    Mapping[str, Any],
    decision:  AllocatorDecision,
    p_correct: Optional[Decimal],
) -> dict:
    """Construct the FinalSignalEvent payload — see schemas/events.py."""
    return {
        "event_type":         "FinalSignalEvent",
        "strategy":           signal.get("strategy", ""),
        "horizon":            decision.chosen_horizon,
        "symbol":             signal.get("symbol", ""),
        "timeframe":          signal.get("timeframe", ""),
        "direction":          int(signal.get("direction", 0)),
        "p_win":              float(signal.get("p_win", 0.0)),
        "p_win_raw":          float(signal.get("p_win_raw", 0.0)),
        "model_version":      signal.get("model_version", ""),
        "feature_set_hash":   signal.get("feature_set_hash", ""),
        "regime":             signal.get("regime", 0),
        "meta_filter_passed": True,
        "bayesian_updated":   False,
        "confidence_tier":    "high",
        "p_correct":          str(p_correct) if p_correct is not None else None,
        "allocator_samples":  {h: str(Decimal(str(v))) for h, v in decision.samples.items()},
        "ts":                 datetime.now(tz=timezone.utc).isoformat(),
    }


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

async def handle_raw_signal(
    signal: Mapping[str, Any],
    ctx:    S9PipelineContext,
) -> Optional[dict]:
    """Execute the S9 pipeline for one RawSignalEvent.

    Returns the final-signal payload that was emitted, or None when the
    pipeline rejected the signal at any stage.
    """
    symbol    = str(signal.get("symbol", ""))
    direction = int(signal.get("direction", 0))
    horizon   = str(signal.get("horizon", ""))
    ts_iso    = signal.get("ts")
    ts        = _parse_ts(ts_iso) or datetime.now(tz=timezone.utc)

    if not symbol or not horizon:
        logger.debug("s9_pipeline.skip.bad_payload", symbol=symbol, horizon=horizon)
        return None

    # 1. Confirmation
    confirmation: ConfirmationResult = await ctx.confirmation.confirm(
        signal=signal,
        symbol=symbol,
        direction=direction,
        ts=ts,
        symbol_is_crypto=_is_crypto(symbol),
    )
    if not confirmation.passed:
        logger.debug(
            "s9_pipeline.confirmation_rejected",
            symbol=symbol, horizon=horizon,
            rejected_by=confirmation.rejected_by,
        )
        return None

    # 2. Allocator (MVP: single-horizon; full multi-horizon aggregation = S11).
    # TODO(@alex 2026-06-15): replace this single-horizon path with a 5-second
    # buffer that aggregates signals across horizons (S11 scope).
    confirmed_horizons = {horizon: signal}
    decision = await ctx.allocator.choose(
        symbol=symbol,
        direction=direction,
        confirmed_horizons=confirmed_horizons,
        ts=ts,
    )
    if decision.chosen_horizon is None:
        logger.debug(
            "s9_pipeline.allocator_rejected",
            symbol=symbol, rejected_by=decision.rejected_by,
        )
        return None

    # 3. Lock
    try:
        async with await ctx.lock.acquire(symbol, direction) as _guard:
            # 4. Emit FinalSignalEvent
            payload = _build_final_signal_payload(
                signal, decision, confirmation.p_correct,
            )
            await _send_kafka(
                ctx.kafka_producer, ctx.final_signal_topic, symbol, payload,
            )
            logger.info(
                "s9_pipeline.emit",
                symbol=symbol, horizon=decision.chosen_horizon,
                direction=direction,
            )
            return payload
    except LockAcquisitionError:
        logger.warning("s9_pipeline.lock_timeout", symbol=symbol, direction=direction)
        return None


# ----------------------------------------------------------------------
# Internal
# ----------------------------------------------------------------------

async def _send_kafka(
    producer: Any,
    topic:    str,
    key:      str,
    payload:  dict,
) -> None:
    if producer is None:
        return
    import json
    try:
        await producer.send_and_wait(
            topic,
            value=json.dumps(payload).encode(),
            key=key.encode(),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("s9_pipeline.kafka.error", topic=topic, error=str(exc))


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            ts = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


__all__ = ["S9PipelineContext", "handle_raw_signal"]
