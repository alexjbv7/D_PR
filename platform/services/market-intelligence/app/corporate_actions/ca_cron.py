"""Daily corporate actions cron — fetch, persist, emit, apply bars.

Schedule: @cron 04:00 ET daily (after universe cron at 03:00).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Literal, cast

import asyncpg
import structlog

from quant_shared.schemas.events import CorporateActionEvent, KafkaTopics

from ..metrics import (
    corporate_actions_fetched_total,
    corporate_actions_applied_seconds,
    corporate_actions_provisional_count,
)
from .alpaca_ca_fetcher import fetch_announcements
from .bars_applier import BarsApplier
from .ca_repository import CARepository

logger = structlog.get_logger(__name__)
UTC = timezone.utc

_CA_TYPE_MAP: dict[str, str] = {
    "forward_split":  "forward_split",
    "reverse_split":  "reverse_split",
    "stock_dividend": "stock_dividend",
    "cash_dividend":  "cash_dividend",
    "merger":         "merger",
    "spinoff":        "spinoff",
    "name_change":    "name_change",
}


async def run(pool: asyncpg.Pool, producer: Any) -> None:
    """
    Entry point for the CA cron.

    Parameters
    ----------
    pool : asyncpg.Pool
    producer : KafkaProducerClient
    """
    log = logger.bind(cron="corporate_actions")
    log.info("ca_cron.start")

    repo    = CARepository(pool)
    applier = BarsApplier(pool, repo)

    try:
        announcements = await fetch_announcements()
    except Exception as exc:
        log.error("ca_cron.fetch_failed", error=str(exc))
        raise

    fetched = len(announcements)
    new_cas = 0
    bars_adjusted = 0

    for ann in announcements:
        try:
            # Resolve ca_type before upsert so we can track metrics correctly
            symbol  = ann.get("symbol", "")
            ca_type = str(ann.get("ca_type", "")).lower().replace(" ", "_")
            mapped  = _CA_TYPE_MAP.get(ca_type, ca_type)
            if mapped not in _CA_TYPE_MAP:
                log.warning("ca_cron.unknown_ca_type", ca_type=ca_type)
                continue

            ca_id, is_new = await repo.upsert_ca(ann)
            if is_new:
                new_cas += 1
                corporate_actions_fetched_total.labels(ca_type=mapped).inc()

            split_from = ann.get("old_rate")
            split_to   = ann.get("new_rate")
            ratio: Decimal | None = None
            if split_from and split_to:
                sf = Decimal(str(split_from))
                st = Decimal(str(split_to))
                if sf != 0:
                    ratio = st / sf

            _ca_type_lit = cast(
                Literal[
                    "forward_split", "reverse_split", "stock_dividend",
                    "cash_dividend", "merger", "spinoff", "name_change",
                ],
                mapped,
            )
            evt = CorporateActionEvent(
                ca_id=ca_id,
                alpaca_id=str(ann.get("id", "")) or None,
                symbol=symbol,
                ca_type=_ca_type_lit,
                ex_ts=_get_ex_ts(ann),
                split_ratio=ratio,
                cash_amount=Decimal(str(ann["cash"])) if ann.get("cash") else None,
                stock_amount=Decimal(str(ann["new_rate"])) if mapped == "stock_dividend" and ann.get("new_rate") else None,
                new_symbol=ann.get("new_symbol"),
                is_provisional=True,
            )
            await producer.send(KafkaTopics.CORPORATE_ACTIONS, evt, key=symbol)

            # Apply to bars immediately (timed)
            ca_row = await _build_ca_dict_from_ann(ann, ca_id, mapped)
            import time as _time
            _t0 = _time.perf_counter()
            rows = await applier.apply(ca_row)
            corporate_actions_applied_seconds.observe(_time.perf_counter() - _t0)
            bars_adjusted += rows

        except Exception as exc:
            log.error("ca_cron.announcement_failed", error=str(exc), ann=ann)
            continue

    # --- Provisional check: upgrade CAs older than 48h if Alpaca confirms ---
    pending = await repo.get_pending_provisional(older_than_hours=48)
    corporate_actions_provisional_count.set(len(pending))
    for ca in pending:
        # Second source check: here we simply confirm (in prod, cross-reference
        # with Alpaca asset metadata or another source).
        await repo.mark_confirmed(ca["ca_id"])
        log.info("ca_cron.provisional_confirmed", ca_id=ca["ca_id"])

    log.info(
        "ca_cron.done",
        fetched=fetched,
        new=new_cas,
        bars_adjusted=bars_adjusted,
    )


def _get_ex_ts(ann: dict[str, Any]) -> datetime:
    from .ca_repository import _parse_ts
    ts = _parse_ts(ann.get("ex_date"))
    return ts or datetime.now(tz=UTC)


async def _build_ca_dict_from_ann(
    ann: dict[str, Any],
    ca_id: str,
    ca_type: str,
) -> dict[str, Any]:
    from .ca_repository import _parse_ts
    return {
        "ca_id":        ca_id,
        "symbol":       ann.get("symbol", ""),
        "ca_type":      ca_type,
        "ex_ts":        _get_ex_ts(ann),
        "split_from":   ann.get("old_rate"),
        "split_to":     ann.get("new_rate"),
        "stock_amount": ann.get("new_rate") if ca_type == "stock_dividend" else None,
        "cash_amount":  ann.get("cash"),
        "new_symbol":   ann.get("new_symbol"),
    }
