"""
HorizonConfig — pre-decided ML settings per trading horizon.

Decisions documented in docs/adr/028-multi-horizon-config.md.
DO NOT modify without updating that ADR.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from typing import Literal


@dataclass(frozen=True)
class HorizonConfig:
    """
    Immutable configuration for one training horizon.

    Parameters
    ----------
    name : str
        Horizon identifier used as report key and registry tag.
    bar_size : str
        Pandas frequency string for bar resampling.
    tp_pct : Decimal
        Take-profit barrier as fraction of entry price.
    sl_pct : Decimal
        Stop-loss barrier as fraction of entry price.
    timeout_bars : int
        Vertical barrier — maximum bars to hold position.
    embargo : timedelta
        Minimum gap between last train bar and first test bar.
        Must be >= max(horizon_timeout_duration, label_barrier_window)
        to prevent label overlap (anti-leakage rule).
    train_lookback : timedelta
        How far back to pull data for each walk-forward fold.
    feature_set : str
        Name of the feature-set constant in feature_sets.py.
    model_name : str
        "xgb" or "mlp".
    n_optuna_trials : int
        Hyperopt budget per horizon (50 default; reduce to 25 under time pressure).
    """

    name:           Literal["intraday", "swing", "daily"]
    bar_size:       str
    tp_pct:         Decimal
    sl_pct:         Decimal
    timeout_bars:   int
    embargo:        timedelta
    train_lookback: timedelta
    feature_set:    str
    model_name:     Literal["xgb", "mlp"]
    n_optuna_trials: int = 50


INTRADAY = HorizonConfig(
    name="intraday",
    bar_size="5min",
    tp_pct=Decimal("0.005"),
    sl_pct=Decimal("0.005"),
    timeout_bars=3,
    # 2h embargo >> 15min (3 bars × 5min) timeout  →  no label overlap
    embargo=timedelta(hours=2),
    train_lookback=timedelta(days=365),
    feature_set="INTRADAY_FEATURES",
    model_name="xgb",
)

SWING = HorizonConfig(
    name="swing",
    bar_size="4H",
    tp_pct=Decimal("0.02"),
    sl_pct=Decimal("0.02"),
    timeout_bars=30,
    # 24h embargo >> 5d (30 bars × 4h) timeout window  →  safe gap
    embargo=timedelta(hours=24),
    train_lookback=timedelta(days=730),
    feature_set="SWING_FEATURES",
    model_name="xgb",
)

DAILY = HorizonConfig(
    name="daily",
    bar_size="1D",
    tp_pct=Decimal("0.04"),
    sl_pct=Decimal("0.03"),
    timeout_bars=20,
    # 5 trading days embargo >= 20-day timeout window
    embargo=timedelta(days=5),
    train_lookback=timedelta(days=1095),
    feature_set="DAILY_FEATURES",
    model_name="mlp",
)

ALL_HORIZONS: list[HorizonConfig] = [INTRADAY, SWING, DAILY]

# Total optuna trials across all horizons — used for cross-horizon DSR correction.
# See docs/adr/029-dsr-n-trials-correction.md
TOTAL_OPTUNA_TRIALS: int = sum(h.n_optuna_trials for h in ALL_HORIZONS)
