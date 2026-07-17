"""
A-003 regression: OOS calibrator must wire into DqnAlphaAgent serve path.

- from_checkpoint_calibrated → p_win_calibrated=True and p_win != p_win_raw (unless identity)
- sidecar auto-load via from_checkpoint
- 🔴 Changes serve p_win → invalidates prior signal/Kelly validation
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

torch = pytest.importorskip("torch")

from alpha.agents.dqn_agent import (  # noqa: E402
    DqnAlphaAgent,
    calibrator_sidecar_path,
)
from envs.trading_env import EnvironmentConfig  # noqa: E402
from models.drl.dqn import TradingDQN  # noqa: E402
from models.drl.dqn_trainer import DQNConfig, DQNTrainer  # noqa: E402
from quant_shared.contracts import MarketContext, PortfolioState  # noqa: E402


def _tiny_env_frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1D", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n))
    df = pd.DataFrame({"close": close}, index=idx)
    # Optional market/regime cols default to 0 in assemble_observation
    return df


def _context(seed: int = 1) -> MarketContext:
    rng = np.random.default_rng(seed)
    features = {f"f{i}": float(rng.normal()) for i in range(3)}
    # Use real feature names so obs is non-degenerate
    from data.drl_dataset import _MARKET_FEATURES, _REGIME_FEATURES

    features = {n: float(rng.normal()) for n in (*_MARKET_FEATURES, *_REGIME_FEATURES)}
    return MarketContext(
        symbol="SPY",
        features=features,
        portfolio=PortfolioState(position=0.0, equity=1.0),
    )


@pytest.fixture
def trained_ckpt(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    net = TradingDQN()
    trainer = DQNTrainer(net, DQNConfig(device="cpu", min_buffer=10, batch_size=8))
    return trainer._save_checkpoint(tmp_path, episode=1)


def test_a003_from_checkpoint_without_sidecar_uncalibrated(trained_ckpt: Path) -> None:
    agent = DqnAlphaAgent.from_checkpoint(trained_ckpt, load_sidecar_calibrator=True)
    sig = agent.predict(_context())
    assert sig.p_win_calibrated is False
    assert sig.p_win == pytest.approx(sig.p_win_raw)


def test_a003_from_checkpoint_calibrated_wires_hook(trained_ckpt: Path) -> None:
    """Canonical serve path: fit on TRAIN_calib → p_win_calibrated=True."""
    calib = _tiny_env_frame(100, seed=2)
    cfg = EnvironmentConfig(episode_length=40, fee_bps=5.0)
    # Force enough directional pairs: random init may go flat; use low min_samples
    agent = DqnAlphaAgent.from_checkpoint_calibrated(
        trained_ckpt,
        calib,
        cfg,
        seed=3,
        method="sigmoid",  # works with small n
        save_sidecar=True,
    )
    assert agent._calibrator is not None
    sig = agent.predict(_context(seed=9))
    assert sig.p_win_calibrated is True
    assert 0.0 <= sig.p_win <= 1.0
    assert calibrator_sidecar_path(trained_ckpt).is_file()

    # Auto-load from sidecar (default serve path after train_drl)
    agent2 = DqnAlphaAgent.from_checkpoint(trained_ckpt)
    assert agent2._calibrator is not None
    sig2 = agent2.predict(_context(seed=9))
    assert sig2.p_win_calibrated is True
    assert sig2.p_win == pytest.approx(sig.p_win)


def test_a003_explicit_none_skips_sidecar(trained_ckpt: Path) -> None:
    calib = _tiny_env_frame(80, seed=4)
    cfg = EnvironmentConfig(episode_length=30)
    DqnAlphaAgent.from_checkpoint_calibrated(
        trained_ckpt, calib, cfg, seed=1, method="sigmoid", save_sidecar=True
    )
    raw = DqnAlphaAgent.from_checkpoint(
        trained_ckpt, calibrator=None, load_sidecar_calibrator=False
    )
    assert raw._calibrator is None
    assert raw.predict(_context()).p_win_calibrated is False
