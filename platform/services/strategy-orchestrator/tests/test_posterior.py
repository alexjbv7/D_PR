"""
Tests — allocator/posterior.py
==============================

5 cases:
  1. decay_zero_days_returns_same
  2. decay_90d_reduces_by_factor   (0.99^90 ≈ 0.404)
  3. decay_floors_at_min            (α, β ≥ 1.0)
  4. update_win_increments_alpha
  5. update_decays_first_then_increments
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import numpy as np
import pytest

from app.allocator.posterior import (
    BetaPosterior,
    DECAY_PER_DAY,
    MIN_ALPHA,
    MIN_BETA,
)


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_decay_zero_days_returns_same() -> None:
    ts = _now()
    p  = BetaPosterior(alpha=Decimal("20"), beta=Decimal("20"), last_update_ts=ts)
    q  = p.decayed_to(ts)
    assert q.alpha == p.alpha
    assert q.beta  == p.beta


def test_decay_90d_reduces_by_factor() -> None:
    ts0 = _now()
    p   = BetaPosterior(alpha=Decimal("100"), beta=Decimal("100"), last_update_ts=ts0)
    q   = p.decayed_to(ts0 + timedelta(days=90))

    expected_factor = DECAY_PER_DAY ** Decimal("90")
    # Tolerance: Decimal ** Decimal is exact in our pure-Python path; use small abs tol.
    assert abs(q.alpha - Decimal("100") * expected_factor) < Decimal("1e-6")
    assert abs(q.beta  - Decimal("100") * expected_factor) < Decimal("1e-6")
    # Sanity: 0.99^90 ≈ 0.4047
    assert float(expected_factor) == pytest.approx(0.4047, abs=0.01)


def test_decay_floors_at_min() -> None:
    """Aggressive decay must not drive α or β below MIN_*."""
    ts0 = _now()
    p   = BetaPosterior(alpha=Decimal("1.5"), beta=Decimal("1.5"), last_update_ts=ts0)
    # 1000 days of decay → 0.99^1000 ≈ 4.3e-5 → without floor α would be ~6.4e-5
    q   = p.decayed_to(ts0 + timedelta(days=1000))
    assert q.alpha == MIN_ALPHA
    assert q.beta  == MIN_BETA


def test_update_win_increments_alpha() -> None:
    ts = _now()
    p  = BetaPosterior(alpha=Decimal("20"), beta=Decimal("20"), last_update_ts=ts)
    q  = p.update("win", ts)
    # No time elapsed → no decay; α += 1
    assert q.alpha == Decimal("21")
    assert q.beta  == Decimal("20")

    r = p.update("loss", ts)
    assert r.alpha == Decimal("20")
    assert r.beta  == Decimal("21")


def test_update_decays_first_then_increments() -> None:
    """The order is decay-then-add, not add-then-decay."""
    ts0 = _now()
    p   = BetaPosterior(alpha=Decimal("100"), beta=Decimal("100"), last_update_ts=ts0)
    ts1 = ts0 + timedelta(days=10)

    q = p.update("win", ts1)

    # Manually compute the expected
    factor = DECAY_PER_DAY ** Decimal("10")
    expected_alpha = Decimal("100") * factor + Decimal("1")
    expected_beta  = Decimal("100") * factor
    assert abs(q.alpha - expected_alpha) < Decimal("1e-6")
    assert abs(q.beta  - expected_beta)  < Decimal("1e-6")
    assert q.last_update_ts == ts1
