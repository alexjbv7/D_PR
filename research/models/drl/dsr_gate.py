"""
Walk-forward DSR promotion gate for DRL agents (ADR-040).

Replaces the fragile single-split heuristic ``edge = oos_reward > 0`` in
``cli/train_drl.py`` with a statistically robust criterion: the agent's
Deflated Sharpe Ratio over CONCATENATED out-of-sample walk-forward folds,
compared against buy-and-hold and an XGBoost supervised baseline.

Promotion requires ALL THREE conditions (ADR-040 §3.2):

1. ``dsr_agent > dsr_threshold``  (default 0.4; KPI target §1.1 is 0.6)
2. ``sharpe_agent > sharpe_buyhold``  (beats holding the underlying)
3. ``dsr_agent > dsr_xgb``  (beats the supervised baseline, CLAUDE.md §6.10)

Anti-leakage contract (ADR-040 §4 — non-negotiable)
----------------------------------------------------
- The regime GMM is re-fitted PER FOLD on that fold's train bars only
  (``data.drl_dataset.build_env_frame`` with explicit ``gmm_train_idx``).
- ``splitter.embargo >= MIN_EMBARGO_BARS`` (60 — the longest feature window,
  ``vol_z_60``) is enforced; violating splitters raise ``ValueError``.
- Test-fold evaluation is GREEDY (``epsilon=0``) — no exploration OOS.
- Baselines use the SAME folds, embargo, and per-bar return definition
  (§3.3) as the agent, so the comparison is fair.

Per-bar return definition (§3.3 — single source of truth)
---------------------------------------------------------
::

    r_t = position_{t-1} * price_return_t - fee_bps/1e4 * |Δposition_t|
    price_return_t = close_t / close_{t-1} - 1

implemented once in ``positions_to_returns`` and shared by the agent and
both baselines. Per fold the series covers the first ``len(test_k) - 1``
test bars (the bars on which the env can act), identically for all three.

Design notes
------------
- Reuses (does NOT reimplement): ``probabilistic_sharpe_ratio`` /
  ``deflated_sharpe_ratio`` (``models.walk_forward_runner``),
  ``WalkForwardSplitter`` (``models.validation``), ``XGBoostClassifier``
  (``models.zoo``), feature builders (``data.drl_dataset``).
- ``torch`` and the DQN trainer are imported lazily inside
  ``walk_forward_oos_returns`` so the gate module (and its tests 1-6) stay
  importable on torch-less CPU environments.
- ``n_trials`` deflates the AGENT's DSR (number of configs/seeds actually
  searched — do not inflate or under-report, ADR-040 §6). The XGBoost
  baseline uses ``n_trials=1`` (its DSR == PSR): a single default config was
  fitted, and deflating it by the agent's trial count would artificially
  lower the bar the agent has to beat.

Cost: walk-forward = N folds x training. For gating, N=3-5 folds with
reduced episodes is enough (ADR-040 §6); wall-clock scales linearly.
"""
from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Optional, Tuple

import numpy as np
import pandas as pd

from data.drl_dataset import (
    _MARKET_FEATURES,
    _REGIME_FEATURES,
    build_env_frame,
    clean_close_series,
    n_clean_bars,
)
from envs import EnvironmentConfig, TradingEnvironment
from models.validation import WalkForwardSplitter
from models.walk_forward_runner import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)

if TYPE_CHECKING:
    from models.drl.dqn_trainer import DQNConfig

logger = logging.getLogger(__name__)

#: Longest rolling feature window (vol_z_60) — minimum embargo (ADR-040 §4.3).
MIN_EMBARGO_BARS: int = 60

_FEATURE_COLS: tuple[str, ...] = (*_MARKET_FEATURES, *_REGIME_FEATURES)


# =====================================================================
# Public dataclasses
# =====================================================================


@dataclass(frozen=True)
class AgentSpec:
    """
    Specification of the DRL agent the gate must train/evaluate per fold.

    Parameters
    ----------
    algo : str
        Algorithm id. MVP supports ``"dqn"`` only (ADR-040 §6: validate the
        gate with DQN first; the contract is algo-agnostic).
    episodes : int
        Training episodes per fold (keep small for gating — cost is
        ``n_folds * episodes``).
    seed : int
        Base random seed; fold ``k`` uses ``seed + k``.
    device : str
        Torch device string ("cpu" or "cuda").
    config : DQNConfig, optional
        Trainer hyperparameters; ``None`` uses ``DQNConfig`` defaults with
        ``device`` overridden.
    """

    algo: str = "dqn"
    episodes: int = 100
    seed: int = 42
    device: str = "cpu"
    config: Optional["DQNConfig"] = None


@dataclass(frozen=True)
class GateResult:
    """
    Verdict of the ADR-040 promotion gate.

    Parameters
    ----------
    dsr_agent : float
        Deflated Sharpe Ratio of the agent on concatenated OOS returns.
    psr_agent : float
        Probabilistic Sharpe Ratio (n_trials=1 view of the same series).
    sharpe_agent : float
        Annualized Sharpe of the agent's OOS returns.
    sharpe_buyhold : float
        Annualized Sharpe of buy-and-hold over the same OOS bars.
    dsr_xgb : float
        DSR (== PSR, n_trials=1) of the XGBoost baseline on the same folds.
    n_trials : int
        Number of agent configs/seeds searched (deflation input).
    n_oos_bars : int
        Total concatenated OOS bars.
    passed : bool
        True only if all three §3.2 conditions hold.
    reason : str
        Human-readable explanation of the verdict (which condition failed).
    """

    dsr_agent: float
    psr_agent: float
    sharpe_agent: float
    sharpe_buyhold: float
    dsr_xgb: float
    n_trials: int
    n_oos_bars: int
    passed: bool
    reason: str


# =====================================================================
# Shared helpers (single return definition — §3.3)
# =====================================================================


def positions_to_returns(
    positions: np.ndarray,
    closes: np.ndarray,
    fee_bps: float,
) -> np.ndarray:
    """
    Per-bar strategy returns from a position path (ADR-040 §3.3).

    ``r_t = pos_{t-1} * (close_t/close_{t-1} - 1) - fee_bps/1e4 * |Δpos_t|``
    with ``pos_{-1} = 0`` (entering the first position pays the fee).

    Parameters
    ----------
    positions : np.ndarray
        Position per bar, values in [-1, 1]; length n.
    closes : np.ndarray
        Close per bar, same length n (aligned with ``positions``).
    fee_bps : float
        Proportional fee in basis points per unit of position change.

    Returns
    -------
    np.ndarray
        Length-n return series. ``r_0`` has no price term (no prior close).
    """
    pos = np.asarray(positions, dtype=float)
    px = np.asarray(closes, dtype=float)
    if pos.shape != px.shape:
        raise ValueError(f"positions {pos.shape} and closes {px.shape} must align")
    n = len(pos)
    if n == 0:
        return np.empty(0, dtype=float)
    prev = np.concatenate([[0.0], pos[:-1]])
    price_ret = np.zeros(n, dtype=float)
    if n > 1:
        price_ret[1:] = px[1:] / px[:-1] - 1.0
    fee = fee_bps / 1e4
    return prev * price_ret - fee * np.abs(pos - prev)


def make_wf_splitter(
    raw_ohlcv: pd.DataFrame,
    n_folds: int,
    *,
    env_cfg: Optional[EnvironmentConfig] = None,
    embargo: int = MIN_EMBARGO_BARS,
) -> WalkForwardSplitter:
    """
    Size an expanding ``WalkForwardSplitter`` over the CLEAN bars of a series.

    Sizing rule: first-fold train = max(episode_length + 1, 30% of clean
    bars); the remaining bars minus the embargo tile into ``n_folds`` equal
    test windows.

    Parameters
    ----------
    raw_ohlcv : pd.DataFrame
        Raw OHLCV frame (warmup bars are discounted via ``n_clean_bars``).
    n_folds : int
        Number of OOS test folds.
    env_cfg : EnvironmentConfig, optional
        For the episode-length lower bound on the first train fold.
    embargo : int
        Bars excluded between train and test (>= ``MIN_EMBARGO_BARS``).

    Returns
    -------
    WalkForwardSplitter
        Expanding splitter producing exactly ``n_folds`` folds.

    Raises
    ------
    ValueError
        If there are not enough clean bars for the requested folds/embargo.
    """
    if n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")
    if embargo < MIN_EMBARGO_BARS:
        raise ValueError(
            f"embargo {embargo} < MIN_EMBARGO_BARS {MIN_EMBARGO_BARS} (ADR-040 §4.3)"
        )
    cfg = env_cfg or EnvironmentConfig()
    n = n_clean_bars(raw_ohlcv)
    min_train = max(cfg.episode_length + 1, int(0.3 * n))
    test_size = (n - min_train - embargo) // n_folds
    if test_size < 10:
        raise ValueError(
            f"insufficient clean bars ({n}) for {n_folds} folds with "
            f"min_train={min_train} and embargo={embargo}"
        )
    train_size = n - embargo - n_folds * test_size
    return WalkForwardSplitter(
        train_size=train_size,
        test_size=test_size,
        expanding=True,
        embargo=embargo,
    )


def _validated_folds(
    raw_ohlcv: pd.DataFrame,
    splitter: WalkForwardSplitter,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (train_idx, test_idx) over the clean frame, enforcing ADR-040 §4.

    Raises
    ------
    ValueError
        If ``splitter.embargo < MIN_EMBARGO_BARS`` or no fold fits the data.
    """
    if splitter.embargo < MIN_EMBARGO_BARS:
        raise ValueError(
            f"splitter.embargo={splitter.embargo} < {MIN_EMBARGO_BARS} bars "
            f"(longest feature window vol_z_60 — ADR-040 §4.3)"
        )
    n = n_clean_bars(raw_ohlcv)
    sized = pd.DataFrame(index=np.arange(n))
    got_any = False
    for train_idx, test_idx in splitter.split(sized):
        got_any = True
        yield train_idx, test_idx
    if not got_any:
        raise ValueError(
            f"splitter produced no folds over {n} clean bars "
            f"(train={splitter.train_size}, test={splitter.test_size}, "
            f"embargo={splitter.embargo})"
        )


# =====================================================================
# OOS return series — agent + baselines (same folds, same return def)
# =====================================================================


def walk_forward_oos_returns(
    agent_spec: AgentSpec,
    raw_ohlcv: pd.DataFrame,
    splitter: WalkForwardSplitter,
    env_cfg: EnvironmentConfig,
    *,
    seed: int = 42,
) -> np.ndarray:
    """
    Train the agent per fold on train_k, evaluate GREEDY (eps=0) on test_k,
    and concatenate the per-bar test returns. No train/test overlap.

    Per fold the regime GMM is re-fitted on exactly that fold's train bars
    (ADR-040 §4.1) via ``build_env_frame(raw, gmm_train_idx=train_idx)``.

    Parameters
    ----------
    agent_spec : AgentSpec
        Algorithm, episodes, seed, device, trainer config.
    raw_ohlcv : pd.DataFrame
        Raw OHLCV (no features) — features are built per fold.
    splitter : WalkForwardSplitter
        Fold generator; ``embargo >= MIN_EMBARGO_BARS`` enforced.
    env_cfg : EnvironmentConfig
        Environment/reward parameters (``fee_bps`` is reused in §3.3).
    seed : int
        Base seed for torch/env; fold ``k`` uses ``seed + k``.

    Returns
    -------
    np.ndarray
        Concatenated OOS per-bar returns across all folds.
    """
    if agent_spec.algo != "dqn":
        raise NotImplementedError(
            f"gate supports algo='dqn' for now (got {agent_spec.algo!r}); "
            f"PPO/SAC gating is a follow-up (ADR-040 §6)"
        )

    import torch  # heavy import — keep the module torch-free (tests 1-6)

    from models.drl.dqn import TradingDQN
    from models.drl.dqn_trainer import DQNConfig, DQNTrainer

    fold_returns: list[np.ndarray] = []
    for k, (train_idx, test_idx) in enumerate(_validated_folds(raw_ohlcv, splitter)):
        frame = build_env_frame(raw_ohlcv, gmm_train_idx=train_idx)
        train_df = frame.iloc[train_idx]
        test_df = frame.iloc[test_idx]
        if len(train_df) < 2 or len(test_df) < 2:
            raise ValueError(
                f"fold {k}: train={len(train_df)} test={len(test_df)} too small"
            )

        torch.manual_seed(seed + k)
        train_cfg = dataclasses.replace(
            env_cfg,
            episode_length=min(env_cfg.episode_length, len(train_df) - 1),
        )
        train_env = TradingEnvironment(train_df, config=train_cfg, seed=seed + k)

        net = TradingDQN(obs_dim=env_cfg.obs_dim)
        dqn_cfg = agent_spec.config or DQNConfig(device=agent_spec.device)
        trainer = DQNTrainer(net, dqn_cfg)
        trainer.train(
            train_env,
            n_episodes=agent_spec.episodes,
            checkpoint_dir=None,
            log_every=0,
        )

        positions = _greedy_positions(trainer, test_df, env_cfg, seed=seed + k)
        closes = test_df["close"].to_numpy()[: len(positions)]
        r = positions_to_returns(positions, closes, env_cfg.fee_bps)
        fold_returns.append(r)
        logger.info(
            "gate fold %d: train=%d test=%d oos_bars=%d mean_r=%.6f",
            k, len(train_df), len(test_df), len(r), float(np.mean(r)),
        )

    return np.concatenate(fold_returns)


def _greedy_positions(
    trainer: "object",
    test_df: pd.DataFrame,
    env_cfg: EnvironmentConfig,
    *,
    seed: int,
) -> np.ndarray:
    """
    Deterministic greedy (epsilon=0) rollout over the whole test slice.

    The eval env's ``episode_length`` is pinned to ``len(test_df) - 1`` so
    ``reset`` has exactly one admissible start (bar 0) — a single
    deterministic pass, no random episode windows.

    Returns
    -------
    np.ndarray
        Position in {-1, 0, +1} for the first ``len(test_df) - 1`` test bars.
    """
    import torch

    eval_cfg = dataclasses.replace(env_cfg, episode_length=len(test_df) - 1)
    env = TradingEnvironment(test_df, config=eval_cfg, seed=seed)
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    positions: list[int] = []
    device = getattr(trainer, "device", "cpu")
    while True:
        action = trainer.online_net.select_action(state.to(device), epsilon=0.0)
        obs, _, terminated, truncated, info = env.step(action)
        state = torch.tensor(obs, dtype=torch.float32)
        positions.append(int(info["position"]))
        if terminated or truncated:
            break
    return np.asarray(positions, dtype=float)


def buyhold_oos_returns(
    raw_ohlcv: pd.DataFrame,
    splitter: WalkForwardSplitter,
    *,
    fee_bps: float | None = None,
) -> np.ndarray:
    """
    Buy-and-hold baseline over the SAME OOS folds as the agent (§3.3).

    ``position_t = +1`` on every test bar; the only position change is the
    initial entry of each fold, so per fold the series equals the close-to-
    close returns minus a one-off entry fee.

    Parameters
    ----------
    raw_ohlcv : pd.DataFrame
        Raw OHLCV frame.
    splitter : WalkForwardSplitter
        Same splitter used for the agent (embargo enforced).
    fee_bps : float, optional
        Fee in bps; default ``EnvironmentConfig().fee_bps`` for consistency
        with the env.

    Returns
    -------
    np.ndarray
        Concatenated OOS per-bar returns.
    """
    fee = EnvironmentConfig().fee_bps if fee_bps is None else fee_bps
    closes_all = clean_close_series(raw_ohlcv).to_numpy()  # no GMM needed
    out: list[np.ndarray] = []
    for _, test_idx in _validated_folds(raw_ohlcv, splitter):
        n_t = len(test_idx) - 1
        closes = closes_all[test_idx[:n_t]]
        positions = np.ones(n_t, dtype=float)
        out.append(positions_to_returns(positions, closes, fee))
    return np.concatenate(out)


def xgb_oos_returns(
    raw_ohlcv: pd.DataFrame,
    splitter: WalkForwardSplitter,
    *,
    fee_bps: float | None = None,
    seed: int = 42,
    xgb_params: dict | None = None,
) -> np.ndarray:
    """
    XGBoost supervised baseline over the SAME OOS folds (CLAUDE.md §6.10).

    Per fold: fit ``XGBoostClassifier`` on the train bars with 3-class
    labels in {-1, 0, +1}: ``sign(next-bar return)``, where moves within the
    fee (``|ret| <= fee_bps/1e4``) label as 0 — a cost-aware deadband that
    keeps the flat class populated (XGBoost >= 1.6 requires every declared
    class present in ``y``). The 1-bar label lookforward is covered by the
    >= 60-bar embargo. Test-bar predictions map directly to positions
    (predicted class == ``sign(argmax_proba - 1)`` when all 3 classes are
    present; if a fold's train slice lacks a class, the model degrades to
    the classes present — predictions remain in {-1, 0, +1}). The regime
    features the model sees are fitted per fold, same as the agent.

    Parameters
    ----------
    raw_ohlcv : pd.DataFrame
        Raw OHLCV frame.
    splitter : WalkForwardSplitter
        Same splitter used for the agent (embargo enforced).
    fee_bps : float, optional
        Fee in bps; default ``EnvironmentConfig().fee_bps``.
    seed : int
        ``random_state`` for XGBoost.
    xgb_params : dict, optional
        Overrides for ``XGBoostClassifier`` (default: shallow anti-overfit
        config, ``max_depth=3``, ``n_estimators=200``).

    Returns
    -------
    np.ndarray
        Concatenated OOS per-bar returns.
    """
    from models.zoo import XGBoostClassifier

    fee = EnvironmentConfig().fee_bps if fee_bps is None else fee_bps
    params = {"max_depth": 3, "n_estimators": 200, "random_state": seed}
    params.update(xgb_params or {})

    out: list[np.ndarray] = []
    for k, (train_idx, test_idx) in enumerate(_validated_folds(raw_ohlcv, splitter)):
        frame = build_env_frame(raw_ohlcv, gmm_train_idx=train_idx)
        closes = frame["close"]
        ret_next = closes.shift(-1) / closes - 1.0
        # Cost-aware deadband: sub-fee moves are class 0 (flat).
        labels = pd.Series(
            np.where(ret_next.abs() <= fee / 1e4, 0.0, np.sign(ret_next)),
            index=frame.index,
        ).where(ret_next.notna())

        X_train = frame.iloc[train_idx][list(_FEATURE_COLS)]
        y_train = labels.iloc[train_idx]
        valid = y_train.notna()
        present = sorted(np.unique(y_train.loc[valid]))

        n_t = len(test_idx) - 1
        X_test = frame.iloc[test_idx[:n_t]][list(_FEATURE_COLS)]
        if len(present) < 2:
            # Degenerate fold (single class): constant position, no model.
            positions = np.full(n_t, present[0] if present else 0.0, dtype=float)
        else:
            model = XGBoostClassifier(**params)
            model.fit(X_train.loc[valid], y_train.loc[valid], all_classes=present)
            positions = model.predict(X_test).astype(float)  # == sign(argmax-1)
        r = positions_to_returns(
            positions, closes.iloc[test_idx[:n_t]].to_numpy(), fee
        )
        out.append(r)
        logger.info("xgb fold %d: oos_bars=%d mean_r=%.6f", k, len(r), float(np.mean(r)))
    return np.concatenate(out)


# =====================================================================
# Gate verdict
# =====================================================================


def _annualized_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    """Annualized Sharpe; 0.0 for degenerate (constant) series, NaN if empty."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    sigma = float(r.std(ddof=1))
    if sigma < 1e-12:
        return 0.0
    return float(r.mean() / sigma * np.sqrt(periods_per_year))


def evaluate_drl_gate(
    agent_returns: np.ndarray,
    buyhold_returns: np.ndarray,
    xgb_returns: np.ndarray,
    n_trials: int,
    dsr_threshold: float = 0.4,
    periods_per_year: int = 252,
) -> GateResult:
    """
    Apply the three ADR-040 §3.2 promotion conditions and explain the verdict.

    Parameters
    ----------
    agent_returns : np.ndarray
        Concatenated OOS per-bar returns of the agent.
    buyhold_returns : np.ndarray
        Same-fold buy-and-hold returns.
    xgb_returns : np.ndarray
        Same-fold XGBoost baseline returns.
    n_trials : int
        Number of agent configs/seeds actually searched (selection-bias
        deflation; 1 if a single config was trained — DSR == PSR then).
    dsr_threshold : float
        Minimum deflated Sharpe for condition 1 (default 0.4).
    periods_per_year : int
        Annualization factor (252 for daily bars).

    Returns
    -------
    GateResult
        Frozen verdict with metrics and a human-readable ``reason``.
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    agent_r = np.asarray(agent_returns, dtype=float)
    agent_r = agent_r[~np.isnan(agent_r)]
    n_oos = int(len(agent_r))

    if n_oos < 4:
        return GateResult(
            dsr_agent=0.0, psr_agent=0.0, sharpe_agent=float("nan"),
            sharpe_buyhold=_annualized_sharpe(buyhold_returns, periods_per_year),
            dsr_xgb=deflated_sharpe_ratio(
                np.asarray(xgb_returns, dtype=float), 1, periods_per_year
            ),
            n_trials=n_trials, n_oos_bars=n_oos, passed=False,
            reason=f"FAIL: only {n_oos} OOS bars — not enough to estimate DSR",
        )

    psr_agent = probabilistic_sharpe_ratio(agent_r, 0.0, periods_per_year)
    dsr_agent = deflated_sharpe_ratio(agent_r, n_trials, periods_per_year)
    sharpe_agent = _annualized_sharpe(agent_r, periods_per_year)
    sharpe_buyhold = _annualized_sharpe(
        np.asarray(buyhold_returns, dtype=float), periods_per_year
    )
    # Baseline fitted once with a default config — n_trials=1 (DSR == PSR).
    # Deflating it by the agent's trial count would lower the bar unfairly.
    dsr_xgb = deflated_sharpe_ratio(
        np.asarray(xgb_returns, dtype=float), 1, periods_per_year
    )

    failures: list[str] = []
    if not dsr_agent > dsr_threshold:
        failures.append(
            f"dsr_agent={dsr_agent:.3f} <= dsr_threshold={dsr_threshold:.2f}"
        )
    if not sharpe_agent > sharpe_buyhold:
        failures.append(
            f"sharpe_agent={sharpe_agent:.3f} <= sharpe_buyhold={sharpe_buyhold:.3f}"
        )
    if not dsr_agent > dsr_xgb:
        failures.append(f"dsr_agent={dsr_agent:.3f} <= dsr_xgb={dsr_xgb:.3f}")

    passed = not failures
    if passed:
        reason = (
            f"PASS: dsr_agent={dsr_agent:.3f} > {dsr_threshold:.2f}, "
            f"sharpe_agent={sharpe_agent:.3f} > sharpe_buyhold={sharpe_buyhold:.3f}, "
            f"dsr_agent > dsr_xgb={dsr_xgb:.3f} "
            f"(n_trials={n_trials}, n_oos_bars={n_oos})"
        )
    else:
        reason = (
            f"FAIL: {'; '.join(failures)} "
            f"(n_trials={n_trials}, n_oos_bars={n_oos})"
        )

    return GateResult(
        dsr_agent=float(dsr_agent),
        psr_agent=float(psr_agent),
        sharpe_agent=float(sharpe_agent),
        sharpe_buyhold=float(sharpe_buyhold),
        dsr_xgb=float(dsr_xgb),
        n_trials=int(n_trials),
        n_oos_bars=n_oos,
        passed=passed,
        reason=reason,
    )
