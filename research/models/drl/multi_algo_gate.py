"""Walk-forward DSR gate generalized to DQN / PPO / SAC (ADR-040 §6, ADR-039).

``models.drl.dsr_gate`` implements the audited ADR-040 gate but only trains DQN
(it raises ``NotImplementedError`` for other algos). This module is the
"PPO/SAC follow-up" promised in that file's docstring: it REUSES every
torch-free helper of ``dsr_gate`` unchanged —

    positions_to_returns, make_wf_splitter, buyhold_oos_returns,
    xgb_oos_returns, evaluate_drl_gate, _validated_folds, _concat_fold_returns,
    AgentSpec, GateResult

— and only adds the per-fold *training + greedy rollout* for the actor-critic
algorithms. The DQN path delegates to ``dsr_gate.walk_forward_oos_returns`` so
its behaviour is bit-identical (zero regression risk to the audited module).

Why a sibling module instead of editing dsr_gate
------------------------------------------------
The DQN gate is covered by the repo's torch-less tests 1-6. Editing its
internals to add algo dispatch risks those guarantees and cannot be re-verified
on a torch-less CI. Adding capability in a new module that *composes* the gate
keeps the audited path frozen while satisfying "cablear PPO/SAC al gate": the
return definition (§3.3), folds, embargo, anti-leakage GMM, baselines and the
three §3.2 promotion conditions are all the gate's own code.

Action-space note
-----------------
All three agents act on the same ``Discrete(3)`` env ({SELL,HOLD,BUY}); greedy
evaluation is ``epsilon=0`` (DQN) / ``deterministic=True`` (PPO actor-critic,
SAC actor). Training budget is normalized to ``episodes`` across algos:
DQN ``n_episodes=episodes``; PPO ``n_updates=episodes`` (rollout ``n_steps`` =
``episode_length``); SAC ``n_steps = episodes * episode_length``. This keeps the
env-step budget comparable so the cross-algo DSR comparison is fair.

Cross-algo promotion (ADR-039): ``DSR(SAC) > DSR(PPO) > DSR(XGBoost)``.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data.drl_dataset import build_env_frame
from envs import EnvironmentConfig, TradingEnvironment
from models.drl.dsr_gate import (
    AgentSpec,
    GateResult,
    MIN_EMBARGO_BARS,
    _concat_fold_returns,
    _validated_folds,
    buyhold_oos_returns,
    evaluate_drl_gate,
    make_wf_splitter,
    positions_to_returns,
    walk_forward_oos_returns as _dqn_walk_forward_oos_returns,
    xgb_oos_returns,
)
from models.validation import WalkForwardSplitter

logger = logging.getLogger(__name__)

#: Algorithms this gate can train and evaluate walk-forward.
SUPPORTED_ALGOS: tuple[str, ...] = ("dqn", "ppo", "sac")

#: Supervised zoo models usable as additional baselines on the same folds.
SUPERVISED_MODELS: tuple[str, ...] = ("logistic", "xgboost", "res_mlp", "lstm")


# =====================================================================
# Per-fold agent training + greedy rollout (PPO / SAC; DQN delegated)
# =====================================================================


def _train_agent_greedy_fn(
    algo: str,
    train_env: "TradingEnvironment",
    env_cfg: EnvironmentConfig,
    episodes: int,
    seed: int,
    device: str,
) -> Callable[[object], int]:
    """Train ``algo`` on ``train_env`` and return a greedy ``state -> action`` fn.

    Heavy imports are local so this module stays importable without torch.
    """
    import torch

    torch.manual_seed(seed)

    if algo == "ppo":
        from models.drl.ppo import PPOConfig, PPOTrainer, TradingActorCritic

        cfg = PPOConfig(
            device=device,
            obs_dim=env_cfg.obs_dim,
            n_steps=max(64, int(env_cfg.episode_length)),
        )
        net = TradingActorCritic(obs_dim=env_cfg.obs_dim)
        trainer = PPOTrainer(net, cfg)
        trainer.train(train_env, n_updates=episodes, checkpoint_dir=None, log_every=0)
        dev = trainer.device

        def greedy(state: object) -> int:
            action, _, _ = net.act(state.to(dev), deterministic=True)  # type: ignore[attr-defined]
            return int(action.item())

        return greedy

    if algo == "sac":
        from models.drl.sac import SACConfig, SACTrainer, TradingDiscreteActor

        cfg = SACConfig(device=device, obs_dim=env_cfg.obs_dim)
        actor = TradingDiscreteActor(obs_dim=env_cfg.obs_dim)
        trainer = SACTrainer(actor, cfg)
        n_steps = max(int(env_cfg.episode_length), episodes * max(1, int(env_cfg.episode_length)))
        trainer.train(train_env, n_steps=n_steps, checkpoint_dir=None, log_every=0)
        dev = trainer.device

        def greedy(state: object) -> int:
            return actor.select_action(state.to(dev), deterministic=True)  # type: ignore[attr-defined]

        return greedy

    if algo == "dqn":  # used only if someone calls the per-fold path directly
        from models.drl.dqn import TradingDQN
        from models.drl.dqn_trainer import DQNConfig, DQNTrainer

        net = TradingDQN(obs_dim=env_cfg.obs_dim)
        trainer = DQNTrainer(net, DQNConfig(device=device))
        trainer.train(train_env, n_episodes=episodes, checkpoint_dir=None, log_every=0)
        dev = trainer.device

        def greedy(state: object) -> int:
            return net.select_action(state.to(dev), epsilon=0.0)  # type: ignore[attr-defined]

        return greedy

    raise NotImplementedError(f"algo={algo!r} not in {SUPPORTED_ALGOS}")


def _greedy_positions(
    greedy_fn: Callable[[object], int],
    test_df: pd.DataFrame,
    env_cfg: EnvironmentConfig,
    *,
    seed: int,
) -> np.ndarray:
    """Deterministic greedy rollout over the whole test slice (mirrors dsr_gate).

    ``episode_length`` is pinned to ``len(test_df) - 1`` so ``reset`` has a
    single admissible start (bar 0): one deterministic pass, no random windows.
    """
    import torch

    eval_cfg = dataclasses.replace(env_cfg, episode_length=len(test_df) - 1)
    env = TradingEnvironment(test_df, config=eval_cfg, seed=seed)
    obs, _ = env.reset()
    state = torch.tensor(obs, dtype=torch.float32)
    positions: list[int] = []
    while True:
        action = greedy_fn(state)
        obs, _, terminated, truncated, info = env.step(action)
        state = torch.tensor(obs, dtype=torch.float32)
        positions.append(int(info["position"]))
        if terminated or truncated:
            break
    return np.asarray(positions, dtype=float)


def _train_eval_one_fold(
    k: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    raw_ohlcv: pd.DataFrame,
    env_cfg: EnvironmentConfig,
    agent_spec: AgentSpec,
    seed: int,
    threads_per_worker: int = 0,
) -> Tuple[int, np.ndarray]:
    """Train PPO/SAC on fold ``k``; return ``(k, greedy OOS returns)``.

    Module-level (picklable) for process-level fold parallelism. The regime GMM
    is re-fitted on exactly this fold's train bars (anti-leakage, ADR-040 §4.1),
    identical to ``dsr_gate._train_eval_one_fold``.
    """
    import random

    import torch

    if threads_per_worker > 0:
        torch.set_num_threads(threads_per_worker)

    random.seed(seed + k)
    np.random.seed((seed + k) % (2**32))
    torch.manual_seed(seed + k)

    frame = build_env_frame(raw_ohlcv, gmm_train_idx=train_idx)
    train_df = frame.iloc[train_idx]
    test_df = frame.iloc[test_idx]
    if len(train_df) < 2 or len(test_df) < 2:
        raise ValueError(f"fold {k}: train={len(train_df)} test={len(test_df)} too small")

    train_cfg = dataclasses.replace(
        env_cfg, episode_length=min(env_cfg.episode_length, len(train_df) - 1)
    )
    train_env = TradingEnvironment(train_df, config=train_cfg, seed=seed + k)

    greedy = _train_agent_greedy_fn(
        agent_spec.algo, train_env, env_cfg, agent_spec.episodes, seed + k, agent_spec.device
    )
    positions = _greedy_positions(greedy, test_df, env_cfg, seed=seed + k)
    closes = test_df["close"].to_numpy()[: len(positions)]
    r = positions_to_returns(positions, closes, env_cfg.fee_bps)
    logger.info(
        "gate[%s] fold %d: train=%d test=%d oos=%d mean_r=%.6f",
        agent_spec.algo, k, len(train_df), len(test_df), len(r), float(np.mean(r)),
    )
    return k, r


def walk_forward_oos_returns(
    agent_spec: AgentSpec,
    raw_ohlcv: pd.DataFrame,
    splitter: WalkForwardSplitter,
    env_cfg: EnvironmentConfig,
    *,
    seed: int = 42,
    n_jobs: int = 1,
    threads_per_worker: Optional[int] = None,
) -> np.ndarray:
    """Concatenated greedy OOS returns for any supported algo.

    ``dqn`` delegates to ``dsr_gate.walk_forward_oos_returns`` (bit-identical);
    ``ppo``/``sac`` use the per-fold trainer above. Same folds, embargo and
    per-bar return definition as the gate baselines.
    """
    if agent_spec.algo == "dqn":
        return _dqn_walk_forward_oos_returns(
            agent_spec, raw_ohlcv, splitter, env_cfg,
            seed=seed, n_jobs=n_jobs, threads_per_worker=threads_per_worker,
        )
    if agent_spec.algo not in SUPPORTED_ALGOS:
        raise NotImplementedError(f"algo={agent_spec.algo!r} not in {SUPPORTED_ALGOS}")

    jobs = [
        (k, tr, te)
        for k, (tr, te) in enumerate(_validated_folds(raw_ohlcv, splitter))
    ]
    n_jobs = max(1, min(n_jobs, len(jobs)))
    if threads_per_worker is None:
        threads_per_worker = 1 if n_jobs > 1 else 0

    if n_jobs == 1:
        results = [
            _train_eval_one_fold(
                k, tr, te, raw_ohlcv, env_cfg, agent_spec, seed, threads_per_worker
            )
            for k, tr, te in jobs
        ]
    else:
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from pathlib import Path

        # spawn (Windows) does not inherit sys.path; put research/ + shared/ on
        # PYTHONPATH so workers can import this module. Harmless on Linux (fork).
        _research = Path(__file__).resolve().parents[2]
        _paths = [str(_research), str(_research.parent / "shared")]
        _existing = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join([*_paths, _existing]).strip(os.pathsep)

        results = []
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = [
                pool.submit(
                    _train_eval_one_fold,
                    k, tr, te, raw_ohlcv, env_cfg, agent_spec, seed, threads_per_worker,
                )
                for k, tr, te in jobs
            ]
            for fut in as_completed(futures):
                results.append(fut.result())

    return _concat_fold_returns(results)


# =====================================================================
# Supervised baselines on the same folds (generalizes dsr_gate.xgb_oos_returns)
# =====================================================================


def supervised_oos_returns(
    raw_ohlcv: pd.DataFrame,
    splitter: WalkForwardSplitter,
    model_name: str,
    *,
    fee_bps: Optional[float] = None,
    seed: int = 42,
) -> np.ndarray:
    """OOS returns of a supervised zoo model on the SAME folds as the agents.

    Generalizes ``dsr_gate.xgb_oos_returns`` to any ``models.zoo`` classifier
    (``logistic``/``xgboost``/``res_mlp``/``lstm``). Same 3-class cost-aware
    deadband labels (``sign(next-ret)`` with sub-fee moves = flat), per-fold
    regime GMM, and ``positions_to_returns``. The 1-bar label lookforward is
    covered by the >= 60-bar embargo.
    """
    from data.drl_dataset import _MARKET_FEATURES, _REGIME_FEATURES

    feature_cols = list(_MARKET_FEATURES) + list(_REGIME_FEATURES)
    fee = EnvironmentConfig().fee_bps if fee_bps is None else fee_bps
    model_cls = _zoo_model_class(model_name)

    out: list[np.ndarray] = []
    for k, (train_idx, test_idx) in enumerate(_validated_folds(raw_ohlcv, splitter)):
        frame = build_env_frame(raw_ohlcv, gmm_train_idx=train_idx)
        closes = frame["close"]
        ret_next = closes.shift(-1) / closes - 1.0
        labels = pd.Series(
            np.where(ret_next.abs() <= fee / 1e4, 0.0, np.sign(ret_next)),
            index=frame.index,
        ).where(ret_next.notna())

        X_train = frame.iloc[train_idx][feature_cols]
        y_train = labels.iloc[train_idx]
        valid = y_train.notna()
        present = sorted(np.unique(y_train.loc[valid]))

        n_t = len(test_idx) - 1
        X_test = frame.iloc[test_idx[:n_t]][feature_cols]
        if len(present) < 2:
            positions = np.full(n_t, present[0] if present else 0.0, dtype=float)
        else:
            model = model_cls(random_state=seed) if model_name == "xgboost" else model_cls()
            _fit_supervised(model, X_train.loc[valid], y_train.loc[valid], present)
            positions = np.asarray(model.predict(X_test), dtype=float)
        r = positions_to_returns(
            positions, closes.iloc[test_idx[:n_t]].to_numpy(), fee
        )
        out.append(r)
        logger.info("supervised[%s] fold %d: oos=%d mean_r=%.6f", model_name, k, len(r), float(np.mean(r)))
    return np.concatenate(out)


def _zoo_model_class(model_name: str):
    """Resolve a ``models.zoo`` class by name (lazy import — torch for nets)."""
    from models import zoo

    mapping = {
        "logistic": zoo.LogisticBaseline,
        "xgboost": zoo.XGBoostClassifier,
        "res_mlp": zoo.ResMLPClassifier,
        "lstm": zoo.LSTMClassifier,
    }
    if model_name not in mapping:
        raise ValueError(f"unknown supervised model {model_name!r}; options: {sorted(mapping)}")
    return mapping[model_name]


def _fit_supervised(model, X: pd.DataFrame, y: pd.Series, present: list) -> None:
    """Fit a zoo model, passing ``all_classes`` when the model accepts it."""
    try:
        model.fit(X, y, all_classes=present)  # XGBoost keeps class alignment across folds
    except TypeError:
        model.fit(X, y)


# =====================================================================
# Single-asset multi-algo gate verdict
# =====================================================================


def _ann_sharpe(returns: np.ndarray, periods_per_year: int) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    sigma = float(r.std(ddof=1))
    return 0.0 if sigma < 1e-12 else float(r.mean() / sigma * np.sqrt(periods_per_year))


def run_gate(
    raw_ohlcv: pd.DataFrame,
    algo: str,
    *,
    n_folds: int,
    episodes: int,
    seeds: Sequence[int],
    env_cfg: EnvironmentConfig,
    dsr_threshold: float = 0.4,
    periods_per_year: int = 252,
    device: str = "cpu",
    n_jobs: int = 1,
) -> dict:
    """Train ``algo`` over ``seeds`` and apply the ADR-040 gate for one asset.

    Returns a dict with the agent/XGBoost/buy-and-hold DSR & Sharpe, the median
    seed used as the agent series, and the §3.2 PASS/FAIL verdict — all annualized
    at ``periods_per_year`` (4H crypto 2190 / FX 1560), not the gate's daily 252.
    """
    splitter = make_wf_splitter(raw_ohlcv, n_folds, env_cfg=env_cfg)
    xgb_r = xgb_oos_returns(raw_ohlcv, splitter, fee_bps=env_cfg.fee_bps, seed=seeds[0])
    bh_r = buyhold_oos_returns(raw_ohlcv, splitter, fee_bps=env_cfg.fee_bps)

    returns: list[np.ndarray] = []
    sharpes: list[float] = []
    for s in seeds:
        spec = AgentSpec(algo=algo, episodes=episodes, seed=s, device=device)
        r = walk_forward_oos_returns(spec, raw_ohlcv, splitter, env_cfg, seed=s, n_jobs=n_jobs)
        returns.append(r)
        sharpes.append(_ann_sharpe(r, periods_per_year))
        logger.info("gate[%s] seed=%d sharpe=%.3f oos=%d", algo, s, sharpes[-1], len(r))

    order = sorted(range(len(sharpes)), key=lambda i: sharpes[i])
    med = order[(len(order) - 1) // 2]
    agent_r = returns[med]

    gate = evaluate_drl_gate(
        agent_r, bh_r, xgb_r,
        n_trials=len(seeds), dsr_threshold=dsr_threshold, periods_per_year=periods_per_year,
    )
    return {
        "algo": algo,
        "n_seeds": len(seeds),
        "dsr_agent": gate.dsr_agent,
        "psr_agent": gate.psr_agent,
        "sharpe_agent": gate.sharpe_agent,
        "sharpe_median_seed": float(sharpes[med]),
        "sharpe_seed_mean": float(np.nanmean(sharpes)),
        "sharpe_seed_lb95": float(np.nanpercentile(sharpes, 5)) if len(sharpes) > 1 else float(sharpes[med]),
        "sharpe_buyhold": gate.sharpe_buyhold,
        "sharpe_xgb": _ann_sharpe(xgb_r, periods_per_year),
        "dsr_xgb": gate.dsr_xgb,
        "n_oos_bars": gate.n_oos_bars,
        "passed": gate.passed,
        "reason": gate.reason,
    }
