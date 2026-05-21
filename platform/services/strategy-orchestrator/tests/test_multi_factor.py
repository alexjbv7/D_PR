"""
Tests — confirmation/multi_factor.py
====================================

5 cases verifying the AND-aggregation + early-reject order:
  1. test_early_reject_regime_unstable    (meta-labeler must NOT be called)
  2. test_early_reject_macro_incoherent   (meta-labeler must NOT be called)
  3. test_all_factors_pass                (full path; p_correct populated)
  4. test_crypto_skips_macro_check        (BTCUSDT bypasses macro coherence)
  5. test_meta_label_threshold            (0.54 → reject; 0.55 → pass)
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock

import pytest

from app.confirmation.factors      import MacroSnapshot, RegimeSnapshot
from app.confirmation.multi_factor import MultiFactorConfirmation


def _now() -> datetime:
    return datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)


def _make_confirmation(
    *,
    regime_stability: Decimal = Decimal("0.8"),
    macro_regime:     str = "neutral",
    meta_p:           Decimal = Decimal("0.70"),
    regime_label:     str = "bull_trend",
) -> tuple[MultiFactorConfirmation, AsyncMock, AsyncMock, AsyncMock]:
    """Build a confirmation with mocked dependencies; return mocks for assertions."""
    regime_repo = AsyncMock()
    regime_repo.get_current.return_value = RegimeSnapshot(
        stability_60bar=regime_stability, label=regime_label,
    )

    macro_repo = AsyncMock()
    macro_repo.get_current.return_value = MacroSnapshot(regime=macro_regime)

    meta = AsyncMock()
    meta.predict.return_value = meta_p

    confirmation = MultiFactorConfirmation(
        regime_repo=regime_repo,
        macro_repo=macro_repo,
        meta_labeler=meta,
    )
    return confirmation, regime_repo, macro_repo, meta


# ----------------------------------------------------------------------
# Early-reject tests (verify meta-labeler is NOT called)
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_early_reject_regime_unstable() -> None:
    """stability_60bar = 0.4 → rejected by regime factor; meta NOT called."""
    confirmation, regime_repo, macro_repo, meta = _make_confirmation(
        regime_stability=Decimal("0.4"),
    )
    result = await confirmation.confirm(
        signal={}, symbol="AAPL", direction=1, ts=_now(),
    )
    assert result.passed is False
    assert result.rejected_by == "regime_unstable"
    meta.predict.assert_not_called()
    # Macro check should NOT happen either (early reject before it).
    macro_repo.get_current.assert_not_called()


@pytest.mark.asyncio
async def test_early_reject_macro_incoherent() -> None:
    """Long equity in risk_off macro → rejected; meta NOT called."""
    confirmation, _, _, meta = _make_confirmation(
        regime_stability=Decimal("0.9"),
        macro_regime="risk_off",
    )
    result = await confirmation.confirm(
        signal={}, symbol="AAPL", direction=1, ts=_now(),
        symbol_is_crypto=False,
    )
    assert result.passed is False
    assert result.rejected_by == "macro_incoherent"
    meta.predict.assert_not_called()


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_factors_pass() -> None:
    confirmation, _, _, meta = _make_confirmation(
        regime_stability=Decimal("0.9"),
        macro_regime="risk_on",
        meta_p=Decimal("0.7"),
    )
    result = await confirmation.confirm(
        signal={}, symbol="AAPL", direction=1, ts=_now(),
    )
    assert result.passed is True
    assert result.rejected_by is None
    assert result.p_correct == Decimal("0.7")
    meta.predict.assert_awaited_once()


# ----------------------------------------------------------------------
# Crypto bypass
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_crypto_skips_macro_check() -> None:
    """BTCUSDT + macro=risk_off → still coherent because symbol_is_crypto=True."""
    confirmation, _, macro_repo, meta = _make_confirmation(
        regime_stability=Decimal("0.9"),
        macro_regime="risk_off",
        meta_p=Decimal("0.7"),
    )
    result = await confirmation.confirm(
        signal={}, symbol="BTCUSDT", direction=1, ts=_now(),
        symbol_is_crypto=True,
    )
    assert result.passed is True
    # Macro is still fetched (factor 3 runs), but the rule allows risk_off for crypto.
    macro_repo.get_current.assert_awaited_once()
    meta.predict.assert_awaited_once()


# ----------------------------------------------------------------------
# Meta-label threshold boundary
# ----------------------------------------------------------------------

@pytest.mark.asyncio
@pytest.mark.parametrize(
    "p, expect_pass",
    [
        (Decimal("0.54"), False),
        (Decimal("0.55"), True),
        (Decimal("0.99"), True),
    ],
)
async def test_meta_label_threshold(p: Decimal, expect_pass: bool) -> None:
    confirmation, _, _, _ = _make_confirmation(
        regime_stability=Decimal("0.9"),
        macro_regime="risk_on",
        meta_p=p,
    )
    result = await confirmation.confirm(
        signal={}, symbol="AAPL", direction=1, ts=_now(),
    )
    assert result.passed is expect_pass
    if not expect_pass:
        assert result.rejected_by == "meta_low_confidence"
