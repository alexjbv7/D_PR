"""
Gymnasium-compatible trading environment for DRL (ADR-037, reward per ADR-041).

State vector layout (42 dims): market (15) | regime (7) | portfolio (5) | reserved (15).

Reward modes (ADR-041)
----------------------
``"mtm"`` (default): per-bar mark-to-market reward aligned with the gate's
return definition (``models.drl.dsr_gate.positions_to_returns``, ADR-040 §3.3).
With ``w_ret = w_cost = 1`` and the penalty terms untriggered, the per-step
reward equals ``positions_to_returns`` bit-for-bit over the same path.

``"realized"``: legacy ADR-037 reward (realized P&L only, cost scaled by
price, idle penalty on holding). Preserved unchanged for A/B shadow runs.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import numpy as np
import pandas as pd

# Actions (ADR-037 MVP)
ACTION_SELL = 0
ACTION_HOLD = 1
ACTION_BUY = 2

# Observation layout indices
_MARKET_COLS: tuple[str | None, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_realized_20",
    "vol_z_60",
    "rsi_14",
    "macd_signal",
    "atr_14",
    "bb_pct",
    "volume_z_20",
    "ob_imbalance",
    "spread_bps",
    "funding_z_60",
    "session_rth",
    None,  # placeholder
)

_REGIME_COLS: tuple[str, ...] = (
    "regime_prob_0",
    "regime_prob_1",
    "regime_prob_2",
    "regime_prob_3",
    "regime_prob_4",
    "regime_stability",
    "vol_regime",
)

_OBS_DIM = 42
_MARKET_DIM = 15
_REGIME_DIM = 7
_PORTFOLIO_DIM = 5

# Portfolio-block normalizers (ADR-037 §1, bloque 3)
_UNREALIZED_NORM = 0.10
_DAILY_PNL_NORM = 0.05


def assemble_observation(
    values: "Mapping[str, float] | pd.Series",
    *,
    position: float,
    unrealized_pnl_pct: float,
    holding_bars: int,
    max_holding_bars: int,
    daily_pnl_pct: float,
) -> np.ndarray:
    """
    Assemble the 42-dim observation (pure function — single source of truth).

    Used by BOTH ``TradingEnvironment._build_observation`` (training) and
    ``alpha.agents.dqn_agent.DqnAlphaAgent`` (serving), so the layout,
    normalization and clipping cannot drift between train and serve.

    Layout (ADR-037): market (15) | regime (7) | portfolio (5) | reserved (15).

    Parameters
    ----------
    values : Mapping[str, float] | pd.Series
        Market + regime feature values by name (``_MARKET_COLS`` /
        ``_REGIME_COLS``); absent names read 0.0 (placeholder convention).
    position : float
        Current position, sign = direction.
    unrealized_pnl_pct : float
        Unrealized P&L as a fraction of equity (normalized by
        ``_UNREALIZED_NORM``, clipped to ±1).
    holding_bars : int
        Bars since the last entry while in a position.
    max_holding_bars : int
        Normalizer for ``holding_bars`` (the env uses ``episode_length``).
    daily_pnl_pct : float
        Day P&L as a fraction of equity (normalized by ``_DAILY_PNL_NORM``,
        clipped to ±1).

    Returns
    -------
    np.ndarray
        float32 observation of shape ``(_OBS_DIM,)``, clipped to [-3, 3].
    """
    obs = np.zeros(_OBS_DIM, dtype=np.float32)

    # Market block (15)
    for i, col in enumerate(_MARKET_COLS):
        if col is not None:
            obs[i] = float(values.get(col, 0.0))

    # Regime block (7)
    base = _MARKET_DIM
    for j, col in enumerate(_REGIME_COLS):
        obs[base + j] = float(values.get(col, 0.0))

    # Portfolio block (5)
    base += _REGIME_DIM
    holding_norm = min(1.0, holding_bars / max(1, max_holding_bars))
    cash_ratio = 1.0 if position == 0 else max(0.0, 1.0 - abs(unrealized_pnl_pct))
    obs[base : base + _PORTFOLIO_DIM] = np.array(
        [
            float(position),
            np.clip(unrealized_pnl_pct / _UNREALIZED_NORM, -1.0, 1.0),
            holding_norm,
            np.clip(daily_pnl_pct / _DAILY_PNL_NORM, -1.0, 1.0),
            cash_ratio,
        ],
        dtype=np.float32,
    )
    # Reserved block (15) stays zero (macro, paso 2 de ADR-037)

    return np.clip(obs, -3.0, 3.0).astype(np.float32)


@dataclass
class EnvironmentConfig:
    """Parámetros del environment — dataclass con defaults."""

    # Reward mode (ADR-041): "mtm" = mark-to-market por barra (default);
    # "realized" = legacy ADR-037, preservado para A/B shadow.
    reward_mode: Literal["mtm", "realized"] = "mtm"

    # Pesos del reward MTM (ADR-041 §3) — espacio de búsqueda de Optuna (§5)
    w_ret: float = 1.0
    w_cost: float = 1.0
    w_dd: float = 2.0
    w_vol: float = 0.5
    w_idle: float = 0.001
    max_flat_bars: int = 20   # barras FLAT toleradas antes de penalizar (mtm)

    # Compartido por ambos modos
    dd_threshold: float = 0.02
    # Target de vol REALIZADA por barra (misma escala que feature ``vol_realized_20``:
    # std de log-returns diarios ~0.01 ≈ 16% anualizado). NO debe igualarse a
    # vol_realized en el mismo bar — eso anula el término w_vol (Y-002).
    vol_target: float = 0.01

    # Reward legacy (ADR-037, reward_mode="realized")
    lambda_dd: float = 2.0
    lambda_vol: float = 0.5
    idle_penalty: float = 0.001
    max_idle_bars: int = 20
    fee_bps: float = 5.0  # 5 bps maker fee Alpaca

    # Episode
    episode_length: int = 252  # barras por episodio
    obs_dim: int = 42


def compute_reward(
    pnl_realized: float,
    equity: float,
    drawdown: float,
    vol_realized: float,
    vol_target: float,
    delta_position: float,
    price: float,
    fee_bps: float,
    holding_bars: int,
    *,
    lambda_dd: float = 2.0,
    dd_threshold: float = 0.02,
    lambda_vol: float = 0.5,
    max_idle_bars: int = 20,
    idle_penalty: float = 0.001,
) -> float:
    """
  Compute step reward (pure function — ADR-037).

  Parameters
  ----------
  pnl_realized : float
      Realized P&L this step (currency units).
  equity : float
      Current equity (must be > 0).
  drawdown : float
      Current drawdown fraction [0, 1].
  vol_realized : float
      Realized volatility at t.
  vol_target : float
      Target / reference volatility.
  delta_position : float
      Absolute position change (e.g. 0, 1, or 2 for discrete {-1,0,1}).
  price : float
      Current price (for transaction cost scaling).
  fee_bps : float
      Fee in basis points.
  holding_bars : int
      Bars since last entry while in a position.
  lambda_dd, dd_threshold, lambda_vol, max_idle_bars, idle_penalty
      Reward shaping hyperparameters.

  Returns
  -------
  float
      Scalar reward for the transition.
  """
    if equity <= 0.0:
        equity = 1e-8

    r_pnl = pnl_realized / equity
    r_risk = (
        -lambda_dd * max(0.0, drawdown - dd_threshold)
        - lambda_vol * max(0.0, vol_realized - vol_target)
    )
    r_cost = -(fee_bps / 10_000.0) * abs(delta_position) * price
    r_idle = -idle_penalty if holding_bars > max_idle_bars else 0.0

    return float(r_pnl + r_risk + r_cost + r_idle)


def compute_reward_mtm(
    prev_position: float,
    price_return: float,
    delta_position: float,
    fee_bps: float,
    drawdown: float,
    vol_realized: float,
    vol_target: float,
    position: float,
    flat_bars: int,
    *,
    w_ret: float = 1.0,
    w_cost: float = 1.0,
    w_dd: float = 2.0,
    w_vol: float = 0.5,
    w_idle: float = 0.001,
    dd_threshold: float = 0.02,
    max_flat_bars: int = 20,
) -> float:
    """
    Compute per-bar mark-to-market step reward (pure function — ADR-041 §3).

    ``r_t = w_ret · (pos_{t-1} · price_return_t)
          − w_cost · (fee_bps / 1e4) · |Δpos_t|
          − w_dd · max(0, drawdown_t − dd_threshold)
          − w_vol · max(0, vol_realized_t − vol_target)
          − w_idle · 1[pos_t == 0 ∧ flat_bars > max_flat_bars]``

    Contract with the promotion gate (ADR-041 §7): with ``w_ret = w_cost = 1``
    and the penalty terms zero/untriggered, the reward equals
    ``models.drl.dsr_gate.positions_to_returns`` bit-for-bit on the same path.
    The cost term is in RETURN units — no ``·price`` factor (the legacy scale
    bug that made fees dominate the reward for high-priced underlyings).

    This is NOT the ADR-037 unrealized-P&L anti-pattern: a losing position
    bleeds negative reward every bar, creating pressure to exit (ADR-041 §4).

    Parameters
    ----------
    prev_position : float
        Position held during the bar, ``pos_{t-1}`` ∈ [-1, 1].
    price_return : float
        ``close_t / close_{t-1} − 1``; 0.0 on the first bar of an episode.
    delta_position : float
        Absolute position change ``|pos_t − pos_{t-1}|``.
    fee_bps : float
        Proportional fee in basis points per unit of position change.
    drawdown : float
        Current drawdown fraction [0, 1].
    vol_realized : float
        Realized volatility at t.
    vol_target : float
        Target / reference volatility.
    position : float
        Position AFTER acting at t, ``pos_t``.
    flat_bars : int
        Consecutive bars spent flat (``pos == 0``) up to and including t.
    w_ret, w_cost, w_dd, w_vol, w_idle, dd_threshold, max_flat_bars
        Reward shaping weights (ADR-041 §3; Optuna search space §5).

    Returns
    -------
    float
        Scalar reward for the transition.
    """
    r_ret = w_ret * prev_position * price_return
    r_cost = -w_cost * (fee_bps / 10_000.0) * abs(delta_position)
    r_dd = -w_dd * max(0.0, drawdown - dd_threshold)
    r_vol = -w_vol * max(0.0, vol_realized - vol_target)
    r_idle = -w_idle if (position == 0.0 and flat_bars > max_flat_bars) else 0.0

    return float(r_ret + r_cost + r_dd + r_vol + r_idle)


class TradingEnvironment(gym.Env):
    """
    Environment DRL para trading algorítmico.

    Parameters
    ----------
    data : pd.DataFrame
        DEBE tener columnas: close, ret_1, ret_5, ret_20, vol_realized_20,
        vol_z_60, rsi_14, macd_signal, atr_14, bb_pct, volume_z_20,
        ob_imbalance (puede ser 0 si no hay LOB), spread_bps (puede ser 0),
        funding_z_60 (puede ser 0), session_rth,
        regime_prob_0..4, regime_stability, vol_regime.
        Index: DatetimeIndex UTC.
    config : EnvironmentConfig
        Parámetros de episodio y reward.
    seed : int
        Semilla para reset aleatorio de ventanas de episodio.

    Observation space: Box(-3, 3, shape=(42,), float32)
    Action space MVP: Discrete(3)  — 0=SELL 1=HOLD 2=BUY
    """

    metadata: dict[str, Any] = {"render_modes": []}

    action_space = gym.spaces.Discrete(3)
    observation_space = gym.spaces.Box(
        low=-3.0,
        high=3.0,
        shape=(_OBS_DIM,),
        dtype=np.float32,
    )

    def __init__(
        self,
        data: pd.DataFrame,
        config: EnvironmentConfig | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self._validate_data(data)
        self.data = data.sort_index()
        self.config = config or EnvironmentConfig()
        if self.config.obs_dim != _OBS_DIM:
            raise ValueError(f"obs_dim must be {_OBS_DIM}, got {self.config.obs_dim}")
        if self.config.reward_mode not in ("mtm", "realized"):
            raise ValueError(
                f"reward_mode must be 'mtm' or 'realized', got {self.config.reward_mode!r}"
            )

        self._rng = np.random.default_rng(seed)

        self._bar_idx: int = 0
        self._episode_step: int = 0
        self._position: int = 0
        self._entry_price: float = 0.0
        self._equity: float = 1.0
        self._peak_equity: float = 1.0
        self._holding_bars: int = 0
        self._flat_bars: int = 0
        self._daily_pnl: float = 0.0
        self._episode_start_idx: int = 0

    @staticmethod
    def _validate_data(data: pd.DataFrame) -> None:
        if "close" not in data.columns:
            raise ValueError("data must contain column 'close'")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("data index must be a DatetimeIndex")
        if data.index.tz is None:
            raise ValueError("data index must be timezone-aware (UTC)")
        if str(data.index.tz) != "UTC":
            # Normalize to UTC for contract compliance
            pass

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        ep_len = self.config.episode_length
        max_start = len(self.data) - ep_len - 1
        if max_start < 0:
            raise ValueError(
                f"data length {len(self.data)} insufficient for episode_length {ep_len}"
            )
        self._episode_start_idx = int(self._rng.integers(0, max_start + 1))
        self._bar_idx = self._episode_start_idx
        self._episode_step = 0
        self._position = 0
        self._entry_price = 0.0
        self._equity = 1.0
        self._peak_equity = 1.0
        self._holding_bars = 0
        self._flat_bars = 0
        self._daily_pnl = 0.0

        obs = self._build_observation()
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if action not in (ACTION_SELL, ACTION_HOLD, ACTION_BUY):
            raise ValueError(f"invalid action {action}")

        row = self.data.iloc[self._bar_idx]
        price = float(row["close"])
        old_position = self._position

        # price_return_t = close_t / close_{t-1} − 1 (ADR-041 §3); 0.0 on the
        # very first bar — same convention as positions_to_returns (r_0 has no
        # price term, and pos_{t-1} = 0 there regardless).
        if self._bar_idx > 0:
            prev_close = float(self.data.iloc[self._bar_idx - 1]["close"])
            price_return = price / prev_close - 1.0
        else:
            price_return = 0.0

        new_position, pnl_realized = self._apply_action(action, price)
        delta_position = float(abs(new_position - old_position))

        self._position = new_position
        if self._position != 0:
            if old_position == 0:
                self._holding_bars = 0
            self._holding_bars += 1
            self._flat_bars = 0
        else:
            self._holding_bars = 0
            self._flat_bars += 1

        self._equity += pnl_realized
        self._daily_pnl += pnl_realized
        self._peak_equity = max(self._peak_equity, self._equity)
        drawdown = 0.0
        if self._peak_equity > 0.0:
            drawdown = max(0.0, (self._peak_equity - self._equity) / self._peak_equity)

        vol_realized = float(row.get("vol_realized_20", 0.0))
        # Y-002: fixed (or configured) target — never equal to realized-of-same-bar
        # (that made r_vol ≡ 0 and w_vol dead).
        vol_target = float(self.config.vol_target)

        if self.config.reward_mode == "mtm":
            reward = compute_reward_mtm(
                prev_position=float(old_position),
                price_return=price_return,
                delta_position=delta_position,
                fee_bps=self.config.fee_bps,
                drawdown=drawdown,
                vol_realized=vol_realized,
                vol_target=vol_target,
                position=float(self._position),
                flat_bars=self._flat_bars,
                w_ret=self.config.w_ret,
                w_cost=self.config.w_cost,
                w_dd=self.config.w_dd,
                w_vol=self.config.w_vol,
                w_idle=self.config.w_idle,
                dd_threshold=self.config.dd_threshold,
                max_flat_bars=self.config.max_flat_bars,
            )
        else:
            reward = compute_reward(
                pnl_realized=pnl_realized,
                equity=self._equity,
                drawdown=drawdown,
                vol_realized=vol_realized,
                vol_target=vol_target,
                delta_position=delta_position,
                price=price,
                fee_bps=self.config.fee_bps,
                holding_bars=self._holding_bars,
                lambda_dd=self.config.lambda_dd,
                dd_threshold=self.config.dd_threshold,
                lambda_vol=self.config.lambda_vol,
                max_idle_bars=self.config.max_idle_bars,
                idle_penalty=self.config.idle_penalty,
            )

        self._episode_step += 1
        self._bar_idx += 1

        terminated = self._episode_step >= self.config.episode_length
        truncated = False

        if (
            terminated
            and self._position != 0
            and self.config.reward_mode == "realized"
        ):
            # Legacy episode-end forced close (ADR-037). In "mtm" the episode
            # ends with the position open: the per-bar rewards already paid
            # its returns, and a forced close would charge a phantom exit fee
            # the gate's return series never sees — `info["position"]` must
            # report the agent's true position for positions_to_returns to
            # match the training reward bit-for-bit (ADR-041 §7). The gate
            # gives buy-and-hold the same no-liquidation treatment.
            final_price = float(self.data.iloc[min(self._bar_idx, len(self.data) - 1)]["close"])
            mtm = self._close_position(final_price)
            self._equity += mtm
            self._daily_pnl += mtm
            reward += compute_reward(
                pnl_realized=mtm,
                equity=max(self._equity, 1e-8),
                drawdown=drawdown,
                vol_realized=vol_realized,
                vol_target=vol_target,
                delta_position=0.0,
                price=final_price,
                fee_bps=self.config.fee_bps,
                holding_bars=self._holding_bars,
                lambda_dd=self.config.lambda_dd,
                dd_threshold=self.config.dd_threshold,
                lambda_vol=self.config.lambda_vol,
                max_idle_bars=self.config.max_idle_bars,
                idle_penalty=self.config.idle_penalty,
            )
            self._position = 0
            self._holding_bars = 0

        obs = self._build_observation()
        info: dict[str, Any] = {
            "equity": self._equity,
            "position": self._position,
            "pnl_realized": pnl_realized,
        }
        return obs, reward, terminated, truncated, info

    def _apply_action(self, action: int, price: float) -> tuple[int, float]:
        """Map discrete action to target position; return (position, realized_pnl)."""
        if action == ACTION_HOLD:
            return self._position, 0.0
        target = -1 if action == ACTION_SELL else 1
        if target == self._position:
            return self._position, 0.0
        pnl = 0.0
        if self._position != 0:
            pnl = self._close_position(price)
        self._entry_price = price
        return target, pnl

    def _close_position(self, price: float) -> float:
        if self._position == 0:
            return 0.0
        ret = (price - self._entry_price) / self._entry_price
        pnl = float(self._position * ret * self._equity)
        self._entry_price = 0.0
        return pnl

    def _build_observation(self) -> np.ndarray:
        """Assemble the 42-dim state via ``assemble_observation`` (ADR-037)."""
        idx = self._bar_idx
        if idx >= len(self.data):
            idx = len(self.data) - 1
        row = self.data.iloc[idx]

        unrealized = 0.0
        if self._position != 0 and self._entry_price > 0.0:
            price = float(row["close"])
            ret = (price - self._entry_price) / self._entry_price
            unrealized = float(self._position * ret)

        return assemble_observation(
            row,
            position=float(self._position),
            unrealized_pnl_pct=unrealized,
            holding_bars=self._holding_bars,
            max_holding_bars=max(1, self.config.episode_length),
            daily_pnl_pct=self._daily_pnl / max(self._equity, 1e-8),
        )

    def render(self) -> None:
        """No-op render (MVP)."""
