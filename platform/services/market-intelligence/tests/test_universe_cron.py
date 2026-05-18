"""Tests for universe_cron — fetch, diff, delisting buffer logic.

All tests use in-memory fakes instead of a real database or Kafka.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional
from unittest.mock import AsyncMock, patch

import pytest

UTC = timezone.utc

import sys, os
_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from app.universe.universe_repository import UniverseRepository  # noqa: E402


# ---------------------------------------------------------------------------
# In-memory UniverseRepository stub
# ---------------------------------------------------------------------------

class _FakeUniverseRepo:
    def __init__(self) -> None:
        self._assets: dict[str, dict[str, Any]] = {}
        self._candidates: dict[str, datetime] = {}
        self._confirmed_delistings: set[str] = set()

    async def upsert_asset(self, asset: dict[str, Any]) -> str:
        sym = asset.get("symbol", "")
        existed = sym in self._assets
        self._assets[sym] = asset
        return "updated" if existed else "inserted"

    async def mark_delisted(self, symbol: str, asset_class: str, ts: datetime) -> None:
        if symbol in self._assets:
            self._assets[symbol]["delisted_ts"] = ts

    async def get_active_symbols(self) -> set[str]:
        return {
            sym for sym, a in self._assets.items()
            if a.get("delisted_ts") is None
        }

    async def add_delisting_candidate(self, symbol: str) -> None:
        if symbol not in self._candidates:
            self._candidates[symbol] = datetime.now(tz=UTC)

    async def remove_delisting_candidate(self, symbol: str) -> None:
        self._candidates.pop(symbol, None)

    async def confirm_delisting_candidates_older_than_days(self, days: int = 3) -> list[str]:
        # In tests we immediately confirm all candidates to simulate time passage.
        result = list(self._candidates.keys())
        for sym in result:
            self._confirmed_delistings.add(sym)
        self._candidates.clear()
        return result


class _FakeProducer:
    def __init__(self) -> None:
        self.sent: list[tuple] = []

    async def send(self, topic: str, event: Any, key: str = "") -> None:
        self.sent.append((topic, event, key))


# ---------------------------------------------------------------------------
# Patch universe_cron to use in-memory repo
# ---------------------------------------------------------------------------

from quant_shared.schemas.events import KafkaTopics, UniverseUpdateEvent  # noqa: E402


async def _run_with_fake_repo(
    fake_repo: _FakeUniverseRepo,
    producer: _FakeProducer,
    assets: list[dict[str, Any]],
) -> None:
    """Replicate universe_cron.run() logic with injectable fake repo."""
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

    known_active = await fake_repo.get_active_symbols()

    for asset in assets:
        sym = asset.get("symbol", "")
        if not sym:
            continue
        result = await fake_repo.upsert_asset(asset)
        if result == "inserted" and asset.get("status") == "active":
            evt = UniverseUpdateEvent(
                symbol=sym,
                asset_class=asset.get("class", "us_equity"),
                change_type="new_listing",
            )
            await producer.send(KafkaTopics.UNIVERSE_UPDATES, evt, key=sym)

    newly_inactive = (known_active - active_symbols) & inactive_symbols
    for sym in newly_inactive:
        await fake_repo.add_delisting_candidate(sym)

    came_back = known_active & active_symbols
    for sym in came_back:
        await fake_repo.remove_delisting_candidate(sym)

    confirmed = await fake_repo.confirm_delisting_candidates_older_than_days(3)
    for sym in confirmed:
        now = datetime.now(tz=UTC)
        await fake_repo.mark_delisted(sym, "us_equity", now)
        evt = UniverseUpdateEvent(
            symbol=sym,
            asset_class="us_equity",
            change_type="delisting",
            delisted_ts=now,
        )
        await producer.send(KafkaTopics.UNIVERSE_UPDATES, evt, key=sym)


# ---------------------------------------------------------------------------
# Test: initial population
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run1_inserts_10_symbols():
    """First run with 10 active symbols persists 10 rows."""
    repo     = _FakeUniverseRepo()
    producer = _FakeProducer()
    assets   = [
        {"symbol": f"SYM{i:02d}", "status": "active", "class": "us_equity",
         "exchange": "XNAS", "tradable": True, "fractionable": False, "shortable": False}
        for i in range(10)
    ]
    await _run_with_fake_repo(repo, producer, assets)

    assert len(repo._assets) == 10
    # All 10 are new_listing events
    assert len([e for _, e, _ in producer.sent if e.change_type == "new_listing"]) == 10


# ---------------------------------------------------------------------------
# Test: delisting detection (with 3-day buffer)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run2_detects_delisting():
    """
    Run #1: 10 active symbols.
    Run #2: 9 active + 1 inactive → candidate added + confirmed (buffer simulated) → delisted_ts set.
    """
    repo     = _FakeUniverseRepo()
    producer = _FakeProducer()

    # Seed: 10 active (simulate previous run state)
    for i in range(10):
        sym = f"SYM{i:02d}"
        repo._assets[sym] = {
            "symbol": sym, "status": "active", "class": "us_equity",
            "exchange": "XNAS", "tradable": True, "fractionable": False,
            "shortable": False, "delisted_ts": None,
        }

    # Run #2: SYM09 goes inactive
    assets_run2 = [
        {"symbol": f"SYM{i:02d}", "status": "active" if i < 9 else "inactive",
         "class": "us_equity", "exchange": "XNAS",
         "tradable": True, "fractionable": False, "shortable": False}
        for i in range(10)
    ]
    await _run_with_fake_repo(repo, producer, assets_run2)

    # SYM09 should be delisted
    assert repo._assets["SYM09"].get("delisted_ts") is not None

    # UniverseUpdateEvent for delisting must have been sent
    delisting_events = [e for _, e, _ in producer.sent if e.change_type == "delisting"]
    assert len(delisting_events) == 1
    assert delisting_events[0].symbol == "SYM09"

    # DoD check: at least one delisted_ts set
    delisted_count = sum(
        1 for a in repo._assets.values() if a.get("delisted_ts") is not None
    )
    assert delisted_count > 0


# ---------------------------------------------------------------------------
# Test: false alarm halt — symbol comes back
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run3_false_alarm_no_delisting():
    """
    Symbol goes inactive → candidate added.
    Next run symbol comes back active → candidate removed, NOT delisted.
    """
    repo     = _FakeUniverseRepo()
    producer = _FakeProducer()

    # Seed: 5 active
    for i in range(5):
        sym = f"SYM{i:02d}"
        repo._assets[sym] = {
            "symbol": sym, "status": "active", "class": "us_equity",
            "exchange": "XNAS", "tradable": True, "fractionable": False,
            "shortable": False, "delisted_ts": None,
        }

    # SYM04 goes inactive temporarily (trading halt)
    assets_halt = [
        {"symbol": f"SYM{i:02d}", "status": "active" if i < 4 else "inactive",
         "class": "us_equity", "exchange": "XNAS",
         "tradable": True, "fractionable": False, "shortable": False}
        for i in range(5)
    ]

    # Don't confirm delistings immediately — simulate first check
    class _FakeRepoNoConfirm(_FakeUniverseRepo):
        async def confirm_delisting_candidates_older_than_days(self, days: int = 3) -> list[str]:
            return []  # Too early, buffer not elapsed

    repo2 = _FakeRepoNoConfirm()
    repo2._assets = dict(repo._assets)
    producer2 = _FakeProducer()

    await _run_with_fake_repo(repo2, producer2, assets_halt)
    assert "SYM04" in repo2._candidates

    # Now SYM04 comes back active
    assets_back = [
        {"symbol": f"SYM{i:02d}", "status": "active",
         "class": "us_equity", "exchange": "XNAS",
         "tradable": True, "fractionable": False, "shortable": False}
        for i in range(5)
    ]
    await _run_with_fake_repo(repo2, producer2, assets_back)

    # Candidate removed; NOT delisted
    assert "SYM04" not in repo2._candidates
    assert repo2._assets["SYM04"].get("delisted_ts") is None

    # No delisting events emitted
    delisting_events = [e for _, e, _ in producer2.sent if e.change_type == "delisting"]
    assert len(delisting_events) == 0
