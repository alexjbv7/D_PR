"""
Tests for the AlphaAgent adapters (ADR-042) — CPU; DQN skips without torch.

The kelly/size test is THE one separating the alpha layer from the portfolio
layer: an agent that sizes capital is an architecture violation (ADR-042 §3.1,
ADR-009), not a tuning issue.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO = Path(__file__).parents[2]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.drl_dataset import _MARKET_FEATURES, _REGIME_FEATURES
from models.zoo import XGBoostClassifier
from quant_shared.contracts import AlphaAgent, MarketContext, PortfolioState
from quant_shared.schemas.signals import SignalDirection, TradeSignal
from alpha.agents.xgb_agent import XgbAlphaAgent

_FEATURE_NAMES = [*_MARKET_FEATURES, *_REGIME_FEATURES]  # las 21 del env-frame

_DIRECTION_BY_LABEL = {
    -1: SignalDirection.SHORT,
    0: SignalDirection.FLAT,
    1: SignalDirection.LONG,
}


def _make_context(seed: int = 0) -> MarketContext:
    rng = np.random.default_rng(seed)
    features = {name: float(rng.normal()) for name in _FEATURE_NAMES}
    return MarketContext(
        symbol="SPY",
        features=features,
        portfolio=PortfolioState(position=0.0, equity=1.0),
    )


@pytest.fixture(scope="module")
def xgb_agent() -> XgbAlphaAgent:
    """Tiny fitted XGBoost (no retraining concerns — synthetic, seconds)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        rng.standard_normal((200, len(_FEATURE_NAMES))), columns=_FEATURE_NAMES
    )
    y = pd.Series(rng.choice([-1, 0, 1], size=200))
    model = XGBoostClassifier(n_estimators=10, max_depth=2, n_jobs=1)
    model.fit(X, y, all_classes=[-1, 0, 1])
    return XgbAlphaAgent(model, model_version="test-0.1")


# ---------------------------------------------------------------------------
# XGBoost adapter
# ---------------------------------------------------------------------------


class TestXgbAlphaAgent:
    def test_is_alpha_agent(self, xgb_agent: XgbAlphaAgent) -> None:
        assert isinstance(xgb_agent, AlphaAgent)

    def test_predict_returns_valid_signal(self, xgb_agent: XgbAlphaAgent) -> None:
        context = _make_context(seed=1)
        signal = xgb_agent.predict(context)

        assert isinstance(signal, TradeSignal)
        assert signal.symbol == "SPY"
        assert 0.0 <= signal.p_win <= 1.0
        assert 0.0 <= signal.p_win_raw <= 1.0
        assert signal.strategy == xgb_agent.hypothesis.id

        # Direction coherent with the model's winning class on the same row.
        X = pd.DataFrame([{f: context.features[f] for f in _FEATURE_NAMES}])
        expected_label = int(xgb_agent._model.predict(X)[0])
        assert signal.direction == _DIRECTION_BY_LABEL[expected_label]

    def test_agent_never_sizes_capital(self, xgb_agent: XgbAlphaAgent) -> None:
        signal = xgb_agent.predict(_make_context(seed=2))
        assert signal.kelly_fraction == 0.0
        assert signal.size_usd == 0.0

    def test_intrinsic_risk_comes_from_config(self, xgb_agent: XgbAlphaAgent) -> None:
        signal = xgb_agent.predict(_make_context(seed=3))
        assert signal.stop_loss_pct == xgb_agent.config.intrinsic_stop_pct
        assert signal.take_profit_pct == xgb_agent.config.intrinsic_target_pct

    def test_module_brings_its_own_fee_model(self, xgb_agent: XgbAlphaAgent) -> None:
        fees = xgb_agent.config.fees
        assert fees is not None
        assert fees.taker_bps > 0.0       # equities: coste por lado
        assert fees.borrow_bps > 0.0      # equities: short borrow
        assert fees.funding is False      # NO es un módulo de crypto perps

    def test_missing_features_raise(self, xgb_agent: XgbAlphaAgent) -> None:
        incomplete = MarketContext(symbol="SPY", features={"rsi_14": 0.5})
        with pytest.raises(ValueError, match="missing"):
            xgb_agent.predict(incomplete)

    def test_requires_fitted_model(self) -> None:
        with pytest.raises(ValueError, match="FITTED"):
            XgbAlphaAgent(XGBoostClassifier(n_estimators=5))

    def test_hypothesis_is_falsifiable(self, xgb_agent: XgbAlphaAgent) -> None:
        h = xgb_agent.hypothesis
        assert h.id == "stock.position.xgb_directional"
        assert h.invalidation  # obligatorio: qué resultado mata la hipótesis
        assert h.benchmark.value == "buy_and_hold"


# ---------------------------------------------------------------------------
# DQN adapter — skips on torch-less CPU envs
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dqn_agent():
    torch = pytest.importorskip("torch")
    from alpha.agents.dqn_agent import DqnAlphaAgent
    from models.drl.dqn import TradingDQN

    torch.manual_seed(42)
    return DqnAlphaAgent(TradingDQN(), model_version="rand-init-test")


class TestDqnAlphaAgent:
    def test_is_alpha_agent(self, dqn_agent) -> None:
        assert isinstance(dqn_agent, AlphaAgent)

    def test_predict_returns_valid_signal(self, dqn_agent) -> None:
        import torch

        context = _make_context(seed=1)
        signal = dqn_agent.predict(context)

        assert isinstance(signal, TradeSignal)
        assert signal.symbol == "SPY"
        assert 0.0 <= signal.p_win <= 1.0
        assert 0.0 <= signal.p_win_raw <= 1.0
        assert signal.strategy == dqn_agent.hypothesis.id

        # Direction coherent with the greedy action on the same observation
        # (select_action(epsilon=0) == argmax Q).
        obs = dqn_agent._build_observation(context)
        state = torch.from_numpy(obs)
        expected_action = dqn_agent._net.select_action(state, epsilon=0.0)
        expected = {0: SignalDirection.SHORT, 1: SignalDirection.FLAT,
                    2: SignalDirection.LONG}[expected_action]
        assert signal.direction == expected

        # p_win_raw is the softmax of the Q-values at the greedy action.
        with torch.no_grad():
            q = dqn_agent._net(state.unsqueeze(0))[0]
        assert signal.p_win_raw == pytest.approx(
            float(torch.softmax(q, dim=-1)[expected_action].item())
        )

    def test_agent_never_sizes_capital(self, dqn_agent) -> None:
        signal = dqn_agent.predict(_make_context(seed=2))
        assert signal.kelly_fraction == 0.0
        assert signal.size_usd == 0.0

    def test_intrinsic_risk_comes_from_config(self, dqn_agent) -> None:
        signal = dqn_agent.predict(_make_context(seed=3))
        assert signal.stop_loss_pct == dqn_agent.config.intrinsic_stop_pct
        assert signal.take_profit_pct == dqn_agent.config.intrinsic_target_pct

    def test_module_brings_its_own_fee_model(self, dqn_agent) -> None:
        fees = dqn_agent.config.fees
        assert fees is not None
        assert fees.taker_bps > 0.0
        assert fees.funding is False

    def test_observation_matches_env_layout(self, dqn_agent) -> None:
        """The rebuilt obs must be 42-dim, clipped, with portfolio block live."""
        context = MarketContext(
            symbol="SPY",
            features={name: 10.0 for name in _FEATURE_NAMES},  # fuera de rango → clip
            portfolio=PortfolioState(
                position=1.0, equity=1.0, unrealized_pnl=0.05, holding_bars=10
            ),
        )
        obs = dqn_agent._build_observation(context)
        assert obs.shape == (42,)
        assert obs.dtype == np.float32
        assert obs.max() <= 3.0 and obs.min() >= -3.0
        assert obs[22] == pytest.approx(1.0)             # position slot
        assert obs[23] == pytest.approx(0.05 / 0.10)     # unrealized / 0.10
        assert np.all(obs[27:42] == 0.0)                 # reserved block

    def test_calibrator_is_applied(self, dqn_agent) -> None:
        from alpha.agents.dqn_agent import DqnAlphaAgent

        calibrated = DqnAlphaAgent(
            dqn_agent._net, model_version="cal", calibrator=lambda p: p / 2.0
        )
        context = _make_context(seed=4)
        raw = dqn_agent.predict(context)
        cal = calibrated.predict(context)
        assert cal.p_win == pytest.approx(raw.p_win_raw / 2.0)
        assert cal.p_win_raw == pytest.approx(raw.p_win_raw)

    def test_from_checkpoint_roundtrip(self, dqn_agent, tmp_path: Path) -> None:
        """Checkpoint written by DQNTrainer loads into an equivalent agent."""
        import torch

        from alpha.agents.dqn_agent import DqnAlphaAgent
        from models.drl.dqn import TradingDQN
        from models.drl.dqn_trainer import DQNConfig, DQNTrainer

        torch.manual_seed(7)
        trainer = DQNTrainer(TradingDQN(), DQNConfig(device="cpu"))
        ckpt = trainer._save_checkpoint(tmp_path, episode=1)  # canonical writer

        agent = DqnAlphaAgent.from_checkpoint(ckpt)
        assert isinstance(agent, AlphaAgent)
        assert agent._model_version == ckpt.name

        context = _make_context(seed=5)
        obs = torch.from_numpy(agent._build_observation(context)).unsqueeze(0)
        trainer.online_net.eval()  # comparar inferencia vs inferencia (BatchNorm)
        with torch.no_grad():
            q_loaded = agent._net(obs)
            q_source = trainer.online_net(obs)
        assert torch.allclose(q_loaded, q_source)
