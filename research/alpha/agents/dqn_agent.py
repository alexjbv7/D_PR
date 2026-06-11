"""
DqnAlphaAgent — AlphaAgent adapter for the existing TradingDQN (ADR-042).

Wraps a trained ``models.drl.dqn.TradingDQN`` (loaded from a ``DQNTrainer``
checkpoint) behind the ``AlphaAgent`` Protocol. No retraining, no changes to
the net, the env or the gate: the adapter only maps ``MarketContext`` → the
42-dim observation the policy was trained on, and the greedy action →
``TradeSignal``.

Self-contained module (ADR-042 §3.1): declares its own ``AlphaHypothesis``
(falsifiable, with explicit invalidation) and its own ``StrategyConfig`` with
the fee model of ITS market (US equities) and its intrinsic stop/target. The
agent PROPOSES that intrinsic risk in the signal; capital sizing and firm
limits are external (``PositionSizer`` / ``RiskGate``, ADR-009) —
``kelly_fraction`` and ``size_usd`` are ALWAYS 0.0 here.

Input dependencies (declared, ADR-042 §3.1)
-------------------------------------------
RL family: consumes ``MarketContext.features`` AND ``MarketContext.portfolio``
to rebuild the env observation (layout imported from ``envs.trading_env`` —
single source of truth, ADR-037):

- market block (15): ``_MARKET_COLS`` from ``features``; absent names read 0.0
  (same convention as the env for placeholder columns).
- regime block (7): ``_REGIME_COLS`` from ``features``.
- portfolio block (5): from ``PortfolioState`` — ``position``,
  ``unrealized_pnl`` (as fraction of equity), ``holding_bars``. The env's
  ``daily_pnl_pct`` slot has NO source in ``PortfolioState`` and is set to
  0.0 — declared limitation, extend ``PortfolioState`` additively if a future
  policy is sensitive to it (ADR-042 §3.1 "límite honesto").
- reserved block (15): zeros, as in the env.

``p_win`` = softmax over the Q-values at the greedy action — an ORDINAL
confidence proxy, NOT a calibrated probability (Q-values are returns, not
log-odds). Pass ``calibrator`` (fitted on OOS outcomes) to map it to a real
probability; without it, downstream consumers must not treat ``p_win`` as
frequency-calibrated.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from envs.trading_env import (
    _MARKET_COLS,
    _MARKET_DIM,
    _OBS_DIM,
    _PORTFOLIO_DIM,
    _REGIME_COLS,
    _REGIME_DIM,
)
from models.drl.dqn import TradingDQN
from quant_shared.contracts import (
    AlphaHypothesis,
    AssetClass,
    Benchmark,
    FeeModel,
    MarketContext,
    StrategyConfig,
    TradingStyle,
)
from quant_shared.schemas.signals import SignalDirection, TradeSignal

#: Falsifiable identity of this module (ADR-042; CATALOGO_ALPHA_HIPOTESIS.md).
DQN_HYPOTHESIS = AlphaHypothesis(
    id="stock.position.dqn_directional",
    asset_class=AssetClass.STOCK,
    style=TradingStyle.POSITION,
    thesis=(
        "Una política DQN entrenada con reward mark-to-market (ADR-041) sobre "
        "features técnicos diarios + régimen GMM captura tendencias multi-día "
        "en equities US mejor que mantener el subyacente."
    ),
    horizon_bars=20,
    benchmark=Benchmark.BUY_AND_HOLD,
    invalidation=(
        "DSR deflactado <= 0.4 sobre OOS concatenado del gate ADR-040, o "
        "Sharpe OOS <= Sharpe buy-and-hold en el mismo walk-forward → "
        "hipótesis muerta; el agente no se promueve."
    ),
)

#: Intrinsic strategy parameters + THIS market's fee model (US equities).
#: Un módulo crypto-perp futuro declararía FeeModel(funding=True, ...) propio.
DQN_CONFIG = StrategyConfig(
    fees=FeeModel(
        taker_bps=5.0,      # coste efectivo por lado usado en el gate (EnvironmentConfig.fee_bps)
        slippage_bps=2.0,
        borrow_bps=50.0,    # short borrow anualizado, general collateral típico
    ),
    intrinsic_stop_pct=0.05,    # estilo position: stop ancho para montar tendencia
    intrinsic_target_pct=0.10,  # R:R 2:1
    max_holding_bars=252,
    bar_size="1d",
    params={"obs_dim": float(_OBS_DIM), "episode_length": 252.0},
)

#: Greedy action index {0=SELL, 1=HOLD, 2=BUY} → signal direction.
_ACTION_TO_DIRECTION: dict[int, SignalDirection] = {
    0: SignalDirection.SHORT,
    1: SignalDirection.FLAT,
    2: SignalDirection.LONG,
}


class DqnAlphaAgent:
    """
    AlphaAgent adapter over a trained ``TradingDQN`` policy.

    Parameters
    ----------
    net : TradingDQN
        Trained policy network (eval mode is enforced). Build it via
        ``DqnAlphaAgent.from_checkpoint`` to load ``DQNTrainer`` artifacts.
    model_version : str
        Version stamp for traceability (e.g. checkpoint filename). Travels in
        ``TradeSignal.model_version``.
    calibrator : callable, optional
        Maps the raw softmax confidence to a calibrated probability
        (``p_cal = calibrator(p_raw)``). ``None`` → ``p_win = p_win_raw``.
    feature_set_hash : str
        Hash of the observation spec. Default: sha256 over the env's market +
        regime column layout.

    Examples
    --------
    >>> agent = DqnAlphaAgent.from_checkpoint(Path("artifacts/drl/dqn_ep00500.pt"))
    >>> signal = agent.predict(context)
    """

    hypothesis: AlphaHypothesis = DQN_HYPOTHESIS
    config: StrategyConfig = DQN_CONFIG

    def __init__(
        self,
        net: TradingDQN,
        model_version: str = "",
        calibrator: Optional[Callable[[float], float]] = None,
        feature_set_hash: str = "",
    ) -> None:
        self._net = net.eval()
        self._model_version = model_version
        self._calibrator = calibrator
        self._feature_set_hash = feature_set_hash or _hash_obs_layout()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        net: TradingDQN | None = None,
        calibrator: Optional[Callable[[float], float]] = None,
    ) -> "DqnAlphaAgent":
        """
        Load a ``DQNTrainer`` checkpoint and wrap its online net.

        Reuses ``DQNTrainer.load_checkpoint`` so the checkpoint format has a
        single owner (§20.2 — no reimplementation).

        Parameters
        ----------
        path : Path
            Checkpoint file written by ``DQNTrainer`` (``dqn_ep*.pt``).
        net : TradingDQN, optional
            Pre-built net matching the checkpoint architecture; ``None`` uses
            the default ``TradingDQN()`` dimensions.
        calibrator : callable, optional
            See ``__init__``.
        """
        from models.drl.dqn_trainer import DQNTrainer

        trainer = DQNTrainer.load_checkpoint(Path(path), online_net=net)
        return cls(
            trainer.online_net,
            model_version=Path(path).name,
            calibrator=calibrator,
        )

    def predict(self, context: MarketContext) -> TradeSignal:
        """
        Rebuild the env observation, take the greedy action, emit the signal.

        The greedy action (``argmax_a Q(s, a)``, identical to
        ``TradingDQN.select_action(state, epsilon=0.0)``) maps {SELL, HOLD,
        BUY} → {SHORT, FLAT, LONG}. ``p_win_raw`` is the softmax of the
        Q-values at that action; ``p_win`` applies the calibrator if present.
        The agent does NOT size capital: ``kelly_fraction`` / ``size_usd``
        stay 0.0 (ADR-042 §3.1, ADR-009).
        """
        obs = self._build_observation(context)
        with torch.no_grad():
            q_values = self._net(torch.from_numpy(obs).unsqueeze(0))[0]
            action = int(q_values.argmax(dim=-1).item())   # greedy (eps=0)
            p_win_raw = float(torch.softmax(q_values, dim=-1)[action].item())

        p_win = p_win_raw
        if self._calibrator is not None:
            p_win = float(min(1.0, max(0.0, self._calibrator(p_win_raw))))

        return TradeSignal(
            symbol=context.symbol,
            direction=_ACTION_TO_DIRECTION[action],
            p_win=p_win,
            p_win_raw=p_win_raw,
            # Regla dura ADR-042 §3.1 / ADR-009: el agente NUNCA dimensiona capital.
            kelly_fraction=0.0,
            size_usd=0.0,
            strategy=self.hypothesis.id,
            model_version=self._model_version,
            feature_set_hash=self._feature_set_hash,
            # Riesgo INTRÍNSECO de la estrategia — el agente PROPONE, la firma dispone.
            stop_loss_pct=self.config.intrinsic_stop_pct,
            take_profit_pct=self.config.intrinsic_target_pct,
        )

    def _build_observation(self, context: MarketContext) -> np.ndarray:
        """
        Rebuild the 42-dim env observation (mirrors
        ``TradingEnvironment._build_observation`` — layout constants imported
        from the env, ADR-037). See the module docstring for the declared
        input dependencies and the ``daily_pnl_pct`` limitation.
        """
        obs = np.zeros(_OBS_DIM, dtype=np.float32)
        features = context.features

        # Market block (15)
        for i, col in enumerate(_MARKET_COLS):
            if col is not None:
                obs[i] = float(features.get(col, 0.0))

        # Regime block (7)
        base = _MARKET_DIM
        for j, col in enumerate(_REGIME_COLS):
            obs[base + j] = float(features.get(col, 0.0))

        # Portfolio block (5) — from PortfolioState
        base += _REGIME_DIM
        pf = context.portfolio
        unrealized = float(pf.unrealized_pnl)  # fraction of equity
        max_holding = max(1.0, float(self.config.params.get("episode_length", 252.0)))
        holding_norm = min(1.0, float(pf.holding_bars) / max_holding)
        daily_pnl_pct = 0.0  # no source in PortfolioState — declared limitation
        cash_ratio = 1.0 if pf.position == 0.0 else max(0.0, 1.0 - abs(unrealized))
        obs[base : base + _PORTFOLIO_DIM] = np.array(
            [
                float(pf.position),
                np.clip(unrealized / 0.10, -1.0, 1.0),
                holding_norm,
                np.clip(daily_pnl_pct / 0.05, -1.0, 1.0),
                cash_ratio,
            ],
            dtype=np.float32,
        )
        # Reserved block (15) stays zero, as in the env.

        return np.clip(obs, -3.0, 3.0).astype(np.float32)


def _hash_obs_layout() -> str:
    """Deterministic hash of the env observation layout (market + regime)."""
    names = [c or "_reserved" for c in _MARKET_COLS] + list(_REGIME_COLS)
    return hashlib.sha256(",".join(names).encode("utf-8")).hexdigest()[:16]
