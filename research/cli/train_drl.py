"""
CLI entry point for DRL training (ADR-006/036/039).

Wires TradingEnvironment + a DRL trainer (DQN → PPO → SAC ladder) with a
temporal train/eval split, checkpointing to artifacts/drl/, and exit codes
suitable for CI / Cursor Cloud background agents.

Usage
-----
    # DQN (current rung of the ladder, ADR-006)
    python -m research.cli.train_drl --algo dqn --episodes 500 --seed 42

    # PPO / SAC (next rungs)
    python -m research.cli.train_drl --algo ppo --updates 200
    python -m research.cli.train_drl --algo sac --steps 100000

    # Validate wiring without training (cheap; run this first in the cloud)
    python -m research.cli.train_drl --algo dqn --dry-run

Options
-------
    --algo          dqn | ppo | sac. Default: dqn.
    --episodes      DQN: number of episodes (also per gate fold). Default: 500.
    --updates       PPO: number of policy updates. Default: 200.
    --steps         SAC: number of environment steps. Default: 100000.
    --seed          Random seed. Default: 42.
    --as-of         Point-in-time end date for the (stub) dataset. Default: today.
    --train-frac    Fraction of data used for training; rest is held-out OOS eval.
    --wf-folds      Walk-forward OOS folds for the DSR gate (ADR-040). Default: 5.
    --n-trials-searched  Number of agent configs/seeds actually searched —
                    deflates the DSR for selection bias. Default: 1.
    --device        cpu | cuda. Default: cpu.
    --checkpoint-dir  Output dir for checkpoints. Default: artifacts/drl/<algo>.
    --dry-run       Build env + trainer and validate; do not train.

Exit codes
----------
    0  training completed and the walk-forward DSR gate PASSED (DQN).
    1  invalid configuration / insufficient data.
    2  trained but the DSR gate FAILED — do not promote (DQN only).

Promotion gate (ADR-040)
------------------------
For DQN the old single-split heuristic ``edge = oos_reward > 0`` is replaced
by ``models.drl.dsr_gate``: the agent is re-trained per walk-forward fold
(regime GMM re-fitted on each fold's train bars only, embargo >= 60 bars),
evaluated greedily (eps=0) on concatenated OOS folds, and promoted only if
DSR_agent > threshold AND Sharpe_agent > buy-and-hold AND DSR_agent >
DSR_XGBoost (CLAUDE.md §6.10). Cost scales as ``wf_folds x episodes``.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Ensure research/ and shared/ are importable.
# File lives at <repo>/research/cli/train_drl.py, so:
#   parents[1] = <repo>/research   parents[2] = <repo>
_RESEARCH = Path(__file__).parents[1]
_REPO = _RESEARCH.parent
for _p in [str(_RESEARCH), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from envs import EnvironmentConfig, TradingEnvironment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_drl")

_VALID_ALGOS = ("dqn", "ppo", "sac")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DRL trainer — DQN/PPO/SAC ladder (ADR-006/036/039)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo", choices=_VALID_ALGOS, default="dqn")
    parser.add_argument("--episodes", type=int, default=500, help="DQN episodes")
    parser.add_argument("--updates", type=int, default=200, help="PPO policy updates")
    parser.add_argument("--steps", type=int, default=100_000, help="SAC env steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--as-of", type=str, default=date.today().isoformat())
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--wf-folds", type=int, default=5,
                        help="Walk-forward OOS folds for the DSR gate (ADR-040)")
    parser.add_argument("--n-trials-searched", type=int, default=1,
                        help="Configs/seeds actually searched (DSR deflation)")
    # Real-data source (Alpaca). If --symbol is omitted, a synthetic stub is used.
    parser.add_argument("--symbol", type=str, default=None,
                        help="Ticker for real Alpaca data, e.g. SPY (omit = stub)")
    parser.add_argument("--start", type=str, default=None, help="Real data start (ISO)")
    parser.add_argument("--end", type=str, default=None, help="Real data end (ISO)")
    parser.add_argument("--timeframe", type=str, default="1d", help="Alpaca timeframe")
    parser.add_argument("--feed", type=str, default="iex", help="Alpaca feed (iex|sip)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_stub_data(n_bars: int, as_of: date, seed: int):
    """
    Synthetic OHLCV stub: a random-walk close with consistent OHLV columns
    on a UTC daily index.

    Full OHLCV (not just close) so the stub flows through the same feature
    pipeline as real data (``data.drl_dataset.build_env_frame``) and through
    the ADR-040 walk-forward gate. Not real signal — wiring validation only.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(as_of, tz="UTC"), periods=n_bars, freq="D")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n_bars))
    high = close * (1.0 + np.abs(rng.normal(0.0, 0.005, n_bars)))
    low = close * (1.0 - np.abs(rng.normal(0.0, 0.005, n_bars)))
    open_ = np.clip(close * (1.0 + rng.normal(0.0, 0.003, n_bars)), low, high)
    volume = rng.integers(1_000_000, 5_000_000, n_bars).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def _evaluate_dqn(trainer, eval_env, n_episodes: int = 20) -> float:
    """Greedy (epsilon=0) rollout on the held-out OOS slice; mean reward."""
    import torch

    rewards: list[float] = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        state = torch.tensor(obs, dtype=torch.float32)
        ep_reward = 0.0
        while True:
            action = trainer.online_net.select_action(
                state.to(trainer.device), epsilon=0.0
            )
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            state = torch.tensor(obs, dtype=torch.float32)
            ep_reward += float(reward)
            if terminated or truncated:
                break
        rewards.append(ep_reward)
    return float(sum(rewards) / len(rewards)) if rewards else 0.0


def _load_raw_ohlcv(args: argparse.Namespace):
    """Raw OHLCV frame: real Alpaca bars when --symbol, else synthetic stub."""
    symbol = getattr(args, "symbol", None)
    if symbol:
        start, end = getattr(args, "start", None), getattr(args, "end", None)
        if not (start and end):
            raise ValueError("--symbol requires --start and --end")
        from data.drl_dataset import fetch_ohlcv_frame

        timeframe = getattr(args, "timeframe", "1d")
        feed = getattr(args, "feed", "iex")
        logger.info("Loading REAL data: %s [%s..%s] tf=%s feed=%s",
                    symbol, start, end, timeframe, feed)
        return fetch_ohlcv_frame(symbol, start, end, timeframe=timeframe, feed=feed)

    ep_len = EnvironmentConfig().episode_length
    # +90 covers the rolling-feature warmup trimmed by build_env_frame.
    min_bars = int((ep_len + 2) / min(args.train_frac, 1.0 - args.train_frac)) + 92
    logger.info("Loading STUB data (random-walk) — not real signal.")
    return _build_stub_data(
        n_bars=max(min_bars, 1200), as_of=date.fromisoformat(args.as_of), seed=args.seed
    )


def _split_envs(
    args: argparse.Namespace,
    raw=None,
) -> tuple[TradingEnvironment, TradingEnvironment]:
    """Single-split train/eval envs (checkpoint training path).

    The regime GMM is fitted on the train fraction only (anti-leakage);
    the walk-forward gate re-fits it per fold separately (ADR-040 §4.1).
    """
    from data.drl_dataset import build_env_frame, n_clean_bars

    env_cfg = EnvironmentConfig()
    ep_len = env_cfg.episode_length
    if raw is None:
        raw = _load_raw_ohlcv(args)

    split = int(n_clean_bars(raw) * args.train_frac)
    data = build_env_frame(raw, gmm_train_idx=split)
    train_data, eval_data = data.iloc[:split], data.iloc[split:]
    if len(train_data) < ep_len + 1 or len(eval_data) < ep_len + 1:
        raise ValueError(
            f"insufficient data after split: train={len(train_data)} "
            f"eval={len(eval_data)} need >= {ep_len + 1} each"
        )

    train_env = TradingEnvironment(train_data, config=env_cfg, seed=args.seed)
    eval_env = TradingEnvironment(eval_data, config=env_cfg, seed=args.seed + 1)
    return train_env, eval_env


def _run(args: argparse.Namespace) -> int:
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _RESEARCH / "artifacts" / "drl" / args.algo

    logger.info("algo=%s seed=%d device=%s train_frac=%.2f as_of=%s",
                args.algo, args.seed, args.device, args.train_frac, args.as_of)

    raw = _load_raw_ohlcv(args)
    train_env, eval_env = _split_envs(args, raw=raw)
    logger.info("Envs built: train/eval temporal split (disjoint, anti-leakage).")

    if args.dry_run:
        logger.info("DRY RUN — env + split valid, no training. ckpt_dir=%s", ckpt_dir)
        return 0

    start_t = time.monotonic()

    if args.algo == "dqn":
        from models.drl import DQNConfig, DQNTrainer, TradingDQN
        from models.drl.dsr_gate import (
            AgentSpec,
            buyhold_oos_returns,
            evaluate_drl_gate,
            make_wf_splitter,
            walk_forward_oos_returns,
            xgb_oos_returns,
        )

        env_cfg = EnvironmentConfig()

        # 1) Single-split training run — produces the checkpoint artifacts.
        net = TradingDQN(obs_dim=env_cfg.obs_dim)
        trainer = DQNTrainer(net, DQNConfig(device=args.device))
        trainer.train(train_env, n_episodes=args.episodes, checkpoint_dir=ckpt_dir)
        oos = _evaluate_dqn(trainer, eval_env, n_episodes=5)
        logger.info("Held-out OOS mean reward (greedy, diagnostic only): %.5f", oos)

        # 2) ADR-040 promotion gate: walk-forward DSR vs baselines.
        wf_folds = getattr(args, "wf_folds", 5)
        n_trials = getattr(args, "n_trials_searched", 1)
        splitter = make_wf_splitter(raw, n_folds=wf_folds, env_cfg=env_cfg)
        logger.info(
            "DSR gate: folds=%d train0=%d test=%d embargo=%d episodes/fold=%d",
            wf_folds, splitter.train_size, splitter.test_size,
            splitter.embargo, args.episodes,
        )
        spec = AgentSpec(
            algo="dqn", episodes=args.episodes, seed=args.seed, device=args.device,
        )
        agent_r = walk_forward_oos_returns(
            spec, raw, splitter, env_cfg, seed=args.seed,
        )
        buyhold_r = buyhold_oos_returns(raw, splitter, fee_bps=env_cfg.fee_bps)
        xgb_r = xgb_oos_returns(
            raw, splitter, fee_bps=env_cfg.fee_bps, seed=args.seed,
        )
        result = evaluate_drl_gate(agent_r, buyhold_r, xgb_r, n_trials=n_trials)
        elapsed = time.monotonic() - start_t
        logger.info("Training + gate done in %.1fs. Checkpoints → %s", elapsed, ckpt_dir)
        logger.info("GATE: %s", result.reason)
        logger.info("GateResult: %s", result)
        return 0 if result.passed else 2

    elif args.algo == "ppo":
        from models.drl import PPOConfig, PPOTrainer, TradingActorCritic

        ac = TradingActorCritic()
        trainer = PPOTrainer(ac, PPOConfig(device=args.device))
        trainer.train(train_env, n_updates=args.updates, checkpoint_dir=ckpt_dir)
        edge = True  # TODO(@alex 2026-06-30): extend AgentSpec/DSR gate to PPO

    else:  # sac
        from models.drl import SACConfig, SACTrainer, TradingDiscreteActor

        actor = TradingDiscreteActor()
        trainer = SACTrainer(actor, SACConfig(device=args.device))
        trainer.train(train_env, n_steps=args.steps, checkpoint_dir=ckpt_dir)
        edge = True  # TODO(@alex 2026-06-30): extend AgentSpec/DSR gate to SAC

    elapsed = time.monotonic() - start_t
    logger.info("Training done in %.1fs. Checkpoints → %s", elapsed, ckpt_dir)

    if edge:
        logger.info("SUCCESS: training completed. Exit 0.")
        return 0
    logger.warning("NO EDGE: do not promote. Exit 2.")
    return 2


def main() -> None:
    args = _parse_args()
    try:
        sys.exit(_run(args))
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
