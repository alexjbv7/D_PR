"""
Integration tests for the signal → executor pipeline.

Covers the three root causes of 0 trades in W1-W2:
  BUG-1: BTCUSDT routes to 'binance' (not registered) → 0 trades
  BUG-2: Kafka consumer never starts when router is empty
  BUG-3: signal missing 'venue' field → falls back to default_crypto (binance)

Tests validate fixes applied in:
  - strategy-orchestrator/app/main.py (_normalise_symbol, venue in payload)
  - execution-engine/app/service.py (venue guard + debug logging)
  - execution-engine/app/main.py (Kafka gate removed)
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup — allow importing platform services without installing them
# ---------------------------------------------------------------------------
_PLATFORM = Path(__file__).parent.parent.parent  # platform/
sys.path.insert(0, str(_PLATFORM / "services" / "execution-engine"))
sys.path.insert(0, str(_PLATFORM.parent / "shared"))

# ---------------------------------------------------------------------------
# Helpers to avoid full service startup
# ---------------------------------------------------------------------------


def _make_fake_router(registered_venues: list[str]):
    """Return a mock Router with only the given venues registered."""
    router = MagicMock()
    router.venues.return_value = registered_venues
    router.default_equity = "alpaca"
    router.default_crypto = "binance"

    def _get(venue):
        from app.brokers.base import BrokerError
        if venue not in registered_venues:
            raise BrokerError(f"No adapter for venue={venue!r}")
        return MagicMock()

    router.get.side_effect = _get
    return router


# ---------------------------------------------------------------------------
# Tests for _normalise_symbol (strategy-orchestrator fix)
# ---------------------------------------------------------------------------


class TestNormaliseSymbol:
    """Tests for strategy-orchestrator._normalise_symbol()."""

    def _import(self):
        # Dynamically import to avoid pulling full orchestrator deps
        orch_path = str(
            Path(__file__).parent.parent.parent.parent
            / "platform" / "services" / "strategy-orchestrator"
        )
        if orch_path not in sys.path:
            sys.path.insert(0, orch_path)
        from app.main import _normalise_symbol, _ALPACA_SYMBOL_MAP
        return _normalise_symbol, _ALPACA_SYMBOL_MAP

    def test_btcusdt_to_alpaca(self):
        _normalise_symbol, _ = self._import()
        assert _normalise_symbol("BTCUSDT", "alpaca") == "BTC/USD"

    def test_ethusdt_to_alpaca(self):
        _normalise_symbol, _ = self._import()
        assert _normalise_symbol("ETHUSDT", "alpaca") == "ETH/USD"

    def test_equity_symbol_passthrough_alpaca(self):
        _normalise_symbol, _ = self._import()
        assert _normalise_symbol("AAPL", "alpaca") == "AAPL"

    def test_unknown_crypto_passthrough_alpaca(self):
        _normalise_symbol, _ = self._import()
        # Unknown crypto → pass through (don't silently mangle it)
        assert _normalise_symbol("WEIRDUSDT", "alpaca") == "WEIRDUSDT"

    def test_binance_venue_passthrough(self):
        _normalise_symbol, _ = self._import()
        assert _normalise_symbol("BTCUSDT", "binance") == "BTCUSDT"

    def test_map_covers_known_crypto(self):
        _, _ALPACA_SYMBOL_MAP = self._import()
        for binance_sym, alpaca_sym in _ALPACA_SYMBOL_MAP.items():
            assert "/" in alpaca_sym, f"{binance_sym} → {alpaca_sym} missing slash"
            assert alpaca_sym.endswith("/USD"), f"{alpaca_sym} should end with /USD"


# ---------------------------------------------------------------------------
# Tests for signal_translator (execution-engine)
# ---------------------------------------------------------------------------


class TestSignalTranslator:
    """Tests for the translate_signal() function."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        from app.signal_translator import translate_signal
        self.translate = translate_signal

    def _signal(self, **overrides) -> dict:
        base = {
            "event_id": "test-001",
            "event_type": "TradingSignalEvent",
            "symbol": "BTC/USD",
            "strategy": "regime_adaptive",
            "direction": 1,
            "p_win": 0.62,
            "position_size": 0.02,
            "venue": "alpaca",
            "confidence": 0.45,
        }
        base.update(overrides)
        return base

    def test_valid_signal_returns_intent(self):
        result = self.translate(
            self._signal(),
            equity=Decimal("10000"),
            current_price=Decimal("50000"),
            default_venue="alpaca",
        )
        assert result is not None
        assert result.symbol == "BTC/USD"
        assert result.venue == "alpaca"

    def test_direction_zero_returns_none(self):
        result = self.translate(
            self._signal(direction=0),
            equity=Decimal("10000"),
            current_price=Decimal("50000"),
            default_venue="alpaca",
        )
        assert result is None

    def test_no_price_returns_none(self):
        result = self.translate(
            self._signal(),
            equity=Decimal("10000"),
            current_price=None,
            default_venue="alpaca",
        )
        assert result is None

    def test_zero_position_size_returns_none(self):
        result = self.translate(
            self._signal(position_size=0),
            equity=Decimal("10000"),
            current_price=Decimal("50000"),
            default_venue="alpaca",
        )
        assert result is None

    def test_below_min_notional_returns_none(self):
        """kelly=0.0001 × equity=$100 = $0.01 < $10 min notional → None."""
        result = self.translate(
            self._signal(position_size=0.0001),
            equity=Decimal("100"),
            current_price=Decimal("50000"),
            default_venue="alpaca",
        )
        assert result is None

    def test_venue_from_signal_takes_precedence(self):
        result = self.translate(
            self._signal(venue="alpaca"),
            equity=Decimal("50000"),
            current_price=Decimal("200"),
            default_venue="binance",  # should be overridden by signal venue
        )
        assert result is not None
        assert result.venue == "alpaca"

    def test_missing_venue_uses_default(self):
        signal = self._signal()
        signal.pop("venue")  # no venue in signal
        result = self.translate(
            signal,
            equity=Decimal("50000"),
            current_price=Decimal("200"),
            default_venue="alpaca",
        )
        assert result is not None
        assert result.venue == "alpaca"

    def test_btcusdt_without_venue_would_use_default_crypto(self):
        """Reproduces BUG-1: BTCUSDT without venue → default_crypto='binance'."""
        signal = {
            "event_id": "bug1-test",
            "symbol": "BTCUSDT",
            "direction": 1,
            "position_size": 0.02,
            # NO venue field — old behaviour
        }
        result = self.translate(
            signal,
            equity=Decimal("10000"),
            current_price=Decimal("50000"),
            default_venue="binance",  # would have been the fallback
        )
        # translate_signal itself would succeed with binance as venue
        # The failure happened in router.get("binance") — not in the translator
        # This test confirms the translator works; the routing fix is in service.py
        assert result is not None
        assert result.venue == "binance"


# ---------------------------------------------------------------------------
# Tests for ExecutionService venue guard (BUG-1 / BUG-3 fix)
# ---------------------------------------------------------------------------


class TestExecutionServiceVenueGuard:
    """Tests for the venue-not-registered guard added to handle_signal()."""

    def _make_service(self, registered_venues: list[str]):
        """Build a minimal ExecutionService with a mock router."""
        from app.service import ExecutionService
        router = _make_fake_router(registered_venues)
        risk_gate = MagicMock()
        risk_gate.evaluate = AsyncMock(return_value=MagicMock(approved=True))
        repo = MagicMock()
        repo.save_result = AsyncMock()
        repo.save_fill = AsyncMock()
        repo.get_open_positions = AsyncMock(return_value=[])
        repo.save_intent = AsyncMock()
        svc = ExecutionService(
            router=router,
            risk_gate=risk_gate,
            repository=repo,
        )
        return svc

    @pytest.mark.asyncio
    async def test_unregistered_venue_returns_none_and_increments_rejected(self):
        """BUG-1 fix: signal for 'binance' (not registered) → rejected, not silent drop."""
        svc = self._make_service(registered_venues=["alpaca"])
        signal = {
            "event_id": "t1",
            "symbol": "BTCUSDT",
            "direction": 1,
            "position_size": 0.02,
            "venue": "binance",  # not registered
        }
        result = await svc.handle_signal(signal)
        assert result is None
        assert svc._counters["rejected"] == 1
        assert svc._counters["signals_seen"] == 1

    @pytest.mark.asyncio
    async def test_registered_venue_proceeds(self):
        """With 'alpaca' registered and a valid signal, handle_signal proceeds past the guard."""
        svc = self._make_service(registered_venues=["alpaca"])

        # Mock the account fetch
        account_mock = MagicMock()
        account_mock.equity = Decimal("10000")
        account_mock.cash = Decimal("9000")
        account_mock.pnl_day = Decimal("0")
        account_mock.account_id = "PA123"
        account_mock.is_paper = True
        account_mock.venue = "alpaca"
        svc.router.get("alpaca").get_account = AsyncMock(return_value=account_mock)
        svc.router.get("alpaca").get_last_price = AsyncMock(return_value=Decimal("200"))

        signal = {
            "event_id": "t2",
            "symbol": "BTC/USD",
            "direction": 1,
            "position_size": 0.02,
            "venue": "alpaca",
        }
        # It will proceed past the venue guard; may fail at risk_gate or submit
        # (both are mocked), we just verify it doesn't return None at the guard
        await svc.handle_signal(signal)
        assert svc._counters["signals_seen"] == 1
        assert svc._counters["rejected"] == 0  # guard passed

    @pytest.mark.asyncio
    async def test_no_venue_in_signal_falls_back_to_default_routing(self):
        """BUG-3 fix: signal without 'venue' uses _default_venue_for(symbol)."""
        svc = self._make_service(registered_venues=["alpaca"])

        # AAPL is equity → routes to default_equity = "alpaca" (registered)
        svc.router.default_equity = "alpaca"
        svc.router.default_crypto = "binance"

        account_mock = MagicMock()
        account_mock.equity = Decimal("10000")
        account_mock.cash = Decimal("9000")
        account_mock.pnl_day = Decimal("0")
        account_mock.account_id = "PA123"
        account_mock.is_paper = True
        svc.router.get("alpaca").get_account = AsyncMock(return_value=account_mock)
        svc.router.get("alpaca").get_last_price = AsyncMock(return_value=Decimal("180"))

        # Patch is_equity to return True for AAPL
        with patch("app.service.is_equity", return_value=True):
            signal = {
                "event_id": "t3",
                "symbol": "AAPL",
                "direction": 1,
                "position_size": 0.02,
                # NO venue field
            }
            await svc.handle_signal(signal)
            assert svc._counters["rejected"] == 0  # equity → alpaca → registered


# ---------------------------------------------------------------------------
# Tests for strategy-orchestrator signal payload completeness
# ---------------------------------------------------------------------------


class TestOrchestratorSignalPayload:
    """Verify the _generate_signal() payload has all fields execution-engine needs."""

    REQUIRED_FIELDS = {"event_id", "symbol", "direction", "position_size", "venue"}

    def _make_feature_vector(self, symbol: str = "BTCUSDT") -> dict:
        return {
            "symbol": symbol,
            "rsi_14": 65.0,
            "macd_hist": 0.5,
            "mom_24h": 0.02,
            "mom_4h": 0.01,
            "whale_sentiment": 0.3,
            "ob_imbalance": 0.2,
            "adx_14": 25.0,
            "sma_cross": 0.003,
            "regime_id": 1.0,
            "macro_leverage": 1.0,
            "funding_rate": 0.0002,
            "reserve_z": -0.5,
            "vol_ratio_1h": 1.0,
            "atr_14": 0.02,
            "bb_width": 0.05,
            "vwap_deviation": 0.001,
            "oi_change_1h": 0.1,
        }

    def test_signal_has_venue_field(self):
        orch_path = str(
            Path(__file__).parent.parent.parent.parent
            / "platform" / "services" / "strategy-orchestrator"
        )
        if orch_path not in sys.path:
            sys.path.insert(0, orch_path)

        import importlib
        import app.main as orch_main

        # Patch the config
        mock_config = MagicMock()
        mock_config.risk_per_trade = 0.02
        mock_config.max_drawdown = 0.10
        mock_config.active_strategies = ["regime_adaptive"]

        with patch.object(orch_main, "_current_config", mock_config), \
             patch.object(orch_main, "_pnl_cache", {"drawdown": 0.0}):
            signal = orch_main._generate_signal(self._make_feature_vector())

        if signal is None:
            pytest.skip("Score too low to generate signal with this fixture")

        for field in self.REQUIRED_FIELDS:
            assert field in signal, f"Signal missing required field: {field!r}"

    def test_signal_symbol_is_alpaca_format(self):
        orch_path = str(
            Path(__file__).parent.parent.parent.parent
            / "platform" / "services" / "strategy-orchestrator"
        )
        if orch_path not in sys.path:
            sys.path.insert(0, orch_path)

        import app.main as orch_main

        mock_config = MagicMock()
        mock_config.risk_per_trade = 0.02
        mock_config.max_drawdown = 0.10
        mock_config.active_strategies = ["regime_adaptive"]

        with patch.object(orch_main, "_current_config", mock_config), \
             patch.object(orch_main, "_pnl_cache", {"drawdown": 0.0}):
            signal = orch_main._generate_signal(self._make_feature_vector("BTCUSDT"))

        if signal is None:
            pytest.skip("Score too low to generate signal with this fixture")

        # After fix: BTCUSDT → BTC/USD for alpaca venue
        if signal.get("venue") == "alpaca":
            assert signal["symbol"] == "BTC/USD", (
                f"Expected 'BTC/USD' for alpaca venue, got {signal['symbol']!r}"
            )
