"""
AllocatorUpdateConsumer — consumes execution.result and updates posteriors.

Lifecycle of an update
----------------------
1. Receive OrderResult from los_ojos.execution.result.
2. Skip if status != FILLED.
3. Skip if not a close (i.e. no realized_pnl available).
4. Extract horizon from OrderResult metadata (strategy field convention).
5. Check idempotency: was_update_applied(trade_id) → skip if true.
6. Under repo.lock_horizon(horizon):
     posterior  = repo.load(horizon)
     new_post   = posterior.update(outcome, ts)
     repo.save(horizon, new_post)
     repo.record_update(...)
7. Emit AllocatorUpdateEvent to los_ojos.allocator.updates.

Win/loss definition (locked decision)
-------------------------------------
realized_pnl > 0  → "win"
realized_pnl ≤ 0  → "loss"

Hooks for the host service
--------------------------
This module exposes a class with a single public entry-point ``handle_event(d)``
that takes an already-deserialised dict (i.e. message.value parsed by the
caller).  The host service owns the consumer loop and DLQ wiring; we keep
the unit test surface small.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, Optional

import structlog

from .posterior  import BetaPosterior
from .repository import AllocatorRepository

logger = structlog.get_logger(__name__)


_VALID_HORIZONS: frozenset[str] = frozenset({"intraday", "swing", "daily"})

# We accept either explicit horizon metadata, or a "strategy" string like
# "xgb_swing_v3" / "deep_mlp_intraday_v1" — extract the horizon token.
_HORIZON_TOKEN_MAP: dict[str, str] = {
    "intraday": "intraday",
    "swing":    "swing",
    "daily":    "daily",
}


def _extract_horizon(payload: dict) -> Optional[str]:
    """Best-effort horizon extraction from execution result metadata.

    Order of attempts:
      1. payload["horizon"]
      2. payload["meta"]["horizon"]
      3. payload["strategy"] containing one of {intraday, swing, daily}
    """
    h = payload.get("horizon")
    if isinstance(h, str) and h in _VALID_HORIZONS:
        return h
    meta = payload.get("meta")
    if isinstance(meta, dict):
        h2 = meta.get("horizon")
        if isinstance(h2, str) and h2 in _VALID_HORIZONS:
            return h2
    strat = payload.get("strategy")
    if isinstance(strat, str):
        s = strat.lower()
        for token, horizon in _HORIZON_TOKEN_MAP.items():
            if token in s:
                return horizon
    return None


def _extract_realized_pnl(payload: dict) -> Optional[Decimal]:
    """Return realized_pnl as Decimal, or None when the result is not a close."""
    pnl = payload.get("realized_pnl")
    if pnl is None:
        # Allow meta.realized_pnl as well (legacy producers stash it there).
        meta = payload.get("meta")
        if isinstance(meta, dict):
            pnl = meta.get("realized_pnl")
    if pnl is None:
        return None
    try:
        return Decimal(str(pnl))
    except Exception:  # noqa: BLE001
        return None


def _outcome(pnl: Decimal) -> Literal["win", "loss"]:
    return "win" if pnl > Decimal("0") else "loss"


class AllocatorUpdateConsumer:
    """Stateless processor: one event in → posterior update + emit out.

    Parameters
    ----------
    repo : AllocatorRepository
    kafka_producer : object, optional
        Async producer used to publish AllocatorUpdateEvent.  Pass None in
        tests that only verify the DB side.
    updates_topic : str
        Default ``los_ojos.allocator.updates``.
    """

    def __init__(
        self,
        repo:           AllocatorRepository,
        kafka_producer: Any = None,
        updates_topic:  str = "los_ojos.allocator.updates",
    ) -> None:
        self._repo     = repo
        self._producer = kafka_producer
        self._topic    = updates_topic

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------

    async def handle_event(self, payload: dict) -> Optional[BetaPosterior]:
        """Apply a single execution.result event.

        Returns the new posterior on a successful update, None otherwise
        (filtered, skipped, or DB error).
        """
        # 1. Status filter — only FILLED results count.
        status = (payload.get("status") or "").lower()
        if status != "filled":
            return None

        # 2. Only realized closes update the posterior.
        pnl = _extract_realized_pnl(payload)
        if pnl is None:
            logger.debug("alloc_update.skip.no_realized_pnl",
                         trade_id=payload.get("result_id"))
            return None

        # 3. Horizon required.
        horizon = _extract_horizon(payload)
        if horizon is None:
            logger.warning("alloc_update.skip.no_horizon",
                           trade_id=payload.get("result_id"))
            return None

        trade_id = payload.get("result_id") or payload.get("trade_id")
        if not trade_id:
            logger.warning("alloc_update.skip.no_trade_id", payload_keys=list(payload))
            return None

        # 4. Idempotency.
        if await self._repo.was_update_applied(trade_id):
            logger.debug("alloc_update.skip.duplicate", trade_id=trade_id)
            return None

        outcome = _outcome(pnl)
        ts_raw  = payload.get("ts_updated") or payload.get("ts")
        ts = _parse_ts(ts_raw) or datetime.now(tz=timezone.utc)

        # 5. Critical section per horizon.
        async with self._repo.lock_horizon(horizon):
            posterior   = await self._repo.load(horizon)
            new_post    = posterior.update(outcome, ts)
            alpha_delta = new_post.alpha - posterior.alpha
            beta_delta  = new_post.beta  - posterior.beta

            await self._repo.save(horizon, new_post)
            await self._repo.record_update(
                update_id    = str(uuid.uuid4()),
                horizon      = horizon,
                trade_id     = trade_id,
                outcome      = outcome,
                realized_pnl = pnl,
                alpha_delta  = alpha_delta,
                beta_delta   = beta_delta,
                alpha_after  = new_post.alpha,
                beta_after   = new_post.beta,
                ts           = ts,
            )

        await self._publish_update(horizon, trade_id, outcome, pnl, new_post)
        return new_post

    # ------------------------------------------------------------------
    # Kafka emit
    # ------------------------------------------------------------------

    async def _publish_update(
        self,
        horizon:      str,
        trade_id:     str,
        outcome:      str,
        realized_pnl: Decimal,
        posterior:    BetaPosterior,
    ) -> None:
        if self._producer is None:
            return
        payload = {
            "event_type":   "AllocatorUpdateEvent",
            "horizon":      horizon,
            "trade_id":     trade_id,
            "outcome":      outcome,
            "realized_pnl": str(realized_pnl),
            "alpha_after":  str(posterior.alpha),
            "beta_after":   str(posterior.beta),
            "ts":           datetime.now(tz=timezone.utc).isoformat(),
        }
        try:
            await self._producer.send_and_wait(
                self._topic,
                value=json.dumps(payload).encode(),
                key=horizon.encode(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "alloc_update.publish.error",
                trade_id=trade_id, error=str(exc),
            )


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


__all__ = ["AllocatorUpdateConsumer"]
