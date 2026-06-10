"""
Tests for the real-data DRL loader (data/drl_dataset.py).

Alpaca fetch is monkeypatched with synthetic OHLCV, so these run offline with
no credentials. The key assertions are (1) exact env column contract, (2) no
NaN, and (3) the GMM regime is fitted on the TRAIN slice only (anti-leakage).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_RESEARCH = Path(__file__).parents[1]
for _p in (str(_RESEARCH), str(_RESEARCH.parent / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data import drl_dataset  # noqa: E402
from data.drl_dataset import _MARKET_FEATURES, _REGIME_FEATURES  # noqa: E402


def _synthetic_ohlcv(n: int = 400, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, n)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, n)))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


@pytest.fixture
def patched_fetch(monkeypatch):
    df = _synthetic_ohlcv()
    monkeypatch.setattr(drl_dataset, "_fetch_ohlcv", lambda *a, **k: df)
    return df


def test_exact_column_contract(patched_fetch):
    out = drl_dataset.build_drl_dataset("TEST", "2020-01-01", "2021-06-01")
    assert list(out.columns) == ["close", *_MARKET_FEATURES, *_REGIME_FEATURES]
    assert str(out.index.tz) == "UTC"
    assert not out.isna().any().any(), "dataset must be NaN-free after warmup trim"


def test_equity_placeholders_and_session(patched_fetch):
    out = drl_dataset.build_drl_dataset("TEST", "2020-01-01", "2021-06-01")
    assert (out["session_rth"] == 1.0).all()
    for col in ("ob_imbalance", "spread_bps", "funding_z_60"):
        assert (out[col] == 0.0).all()
    # Only 3 GMM components → slots 3 and 4 stay zero.
    assert (out["regime_prob_3"] == 0.0).all()
    assert (out["regime_prob_4"] == 0.0).all()


def test_regime_probs_are_a_distribution(patched_fetch):
    out = drl_dataset.build_drl_dataset("TEST", "2020-01-01", "2021-06-01")
    probs = out[[f"regime_prob_{k}" for k in range(5)]].sum(axis=1)
    assert np.allclose(probs.to_numpy(), 1.0, atol=1e-6)


def test_gmm_fitted_on_train_slice_only(patched_fetch, monkeypatch):
    """Anti-leakage: GMM.fit must see fewer rows than the full clean dataset."""
    import features.regime_gmm as rg

    seen = {}
    orig_fit = rg.GMMRegimeDetector.fit

    def spy_fit(self, close, atr):
        seen["n_fit"] = len(close)
        return orig_fit(self, close, atr)

    monkeypatch.setattr(rg.GMMRegimeDetector, "fit", spy_fit)

    out = drl_dataset.build_drl_dataset("TEST", "2020-01-01", "2021-06-01", train_frac=0.7)
    # The fit slice must be the training portion only — strictly less than the
    # full (pre-dropna) clean series, and close to 70% of it.
    assert "n_fit" in seen
    assert seen["n_fit"] < len(out), "GMM saw the whole series → leakage!"
    assert seen["n_fit"] >= int(0.6 * len(out))  # ~70% minus warmup trim


def test_invalid_train_frac_raises(patched_fetch):
    with pytest.raises(ValueError):
        drl_dataset.build_drl_dataset("TEST", "2020-01-01", "2021-06-01", train_frac=1.5)
