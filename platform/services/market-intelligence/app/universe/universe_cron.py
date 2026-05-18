"""Daily universe cron — fetch /v2/assets, detect delistings, emit events.

Schedule: @cron 03:00 ET daily (wired in main.py lifespan).

Edge cases handled:
- Trading halts vs permanent delisting: confirmed only after 3 days inactive.
- Symbols returning to active (false alarm): candidate entry removed.
- Metadata changes (fractionable / tradable / exchange): emitted as metadata_update.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import asyncpg
import structlog

from quant_shared.schemas.events import KafkaTopics, UniverseUpdateEvent

from ..metrics import (
    universe_cron_runs_total,
    universe_changes_total,
)
from .alpaca_assets_fetcher import fetch_all_assets
from .universe_repository import UniverseRepository

logger = structlog.get_logger(__name__)

UTC = timezone.utc
_DELISTING_BUFFER_DAYS = 3


async def run(pool: asyncpg.Pool, producer: Any) -> None:
    """
    Entry point for the universe cron.

    Parameters
    ----------
    pool : asyncpg.Pool
    producer : KafkaProducerClient
    """
    log = logger.bind(cron="universe")
    log.info("universe_cron.start")

    repo = UniverseRepository(pool)

    try:
        assets = await fetch_all_assets()
    except Exception as exc:
        log.error("universe_cron.fetch_failed", error=str(exc))
        universe_cron_runs_total.labels(result="failure").inc()
        raise

    # Split by Alpaca status field
    active_symbols:   set[str] = set()
    inactive_symbols: set[str] = set()
    for a in assets:
        sym = a.get("symbol", "")
        if not sym:
            continue
        if a.get("status") == "active":
            active_symbols.add(sym)
        else:
            inactive_symbols.add(sym)

    known_active = await repo.get_active_symbols()

    new_listings  = 0
    meta_updates  = 0
    delistings    = 0

    # --- 1. Upsert all assets into universe_historical ---
    for asset in assets:
        sym = asset.get("symbol", "")
        if not sym:
            continue
        result = await repo.upsert_asset(asset)

        if result == "inserted" and asset.get("status") == "active":
            new_listings += 1
            evt = UniverseUpdateEvent(
                symbol=sym,
                asset_class=asset.get("class", "us_equity"),
                change_type="new_listing",
            )
            await producer.send(KafkaTopics.UNIVERSE_UPDATES, evt, key=sym)
            universe_changes_total.labels(type="new_listing").inc()
            log.info("universe.new_listing", symbol=sym)
        elif result == "updated":
            meta_updates += 1
            universe_changes_total.labels(type="metadata_update").inc()

    # --- 2. Detect newly inactive symbols (may be halt or real delist) ---
    newly_inactive = (known_active - active_symbols) & inactive_symbols
    for sym in newly_inactive:
        await repo.add_delisting_candidate(sym)
        log.info("universe.delisting_candidate", symbol=sym)

    # --- 3. Symbols that came back to active: remove from candidates ---
    came_back = known_active & active_symbols
    for sym in came_back:
        await repo.remove_delisting_candidate(sym)

    # --- 4. Confirm delistings older than buffer days ---
    confirmed = await repo.confirm_delisting_candidates_older_than_days(_DELISTING_BUFFER_DAYS)
    for sym in confirmed:
        now = datetime.now(tz=UTC)
        await repo.mark_delisted(sym, "us_equity", now)
        delistings += 1
        evt = UniverseUpdateEvent(
            symbol=sym,
            asset_class="us_equity",
            change_type="delisting",
            delisted_ts=now,
        )
        await producer.send(KafkaTopics.UNIVERSE_UPDATES, evt, key=sym)
        universe_changes_total.labels(type="delisting").inc()
        log.info("universe.confirmed_delisting", symbol=sym)

    universe_cron_runs_total.labels(result="success").inc()
    log.info(
        "universe_cron.done",
        total=len(assets),
        new_listings=new_listings,
        meta_updates=meta_updates,
        delistings=delistings,
    )
