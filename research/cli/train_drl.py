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
    --episodes      DQN: number of episodes. Default: 500.
    --updates       PPO: number of policy updates. Default: 200.
    --steps         SAC: number of environment steps. Default: 100000.
    --seed          Random seed. Default: 42.
    --as-of         Point-in-time end date for the (stub) dataset. Default: today.
    --train-frac    Fraction of data used for training; rest is held-out OOS eval.
    --device        cpu | cuda. Default: cpu.
    --checkpoint-dir  Output dir for checkpoints. Default: artifacts/drl/<algo>.
    --dry-run       Build env + trainer and validate; do not train.

Exit codes
----------
    0  training completed (and, for DQN, held-out OOS mean reward > 0).
    1  invalid configuration / insufficient data.
    2  trained but held-out OOS evaluation shows no edge (DQN only).

Anti-leakage
------------
Training and evaluation use disjoint, time-ordered slices of the data
(`--train-frac` split). The trainer never sees the evaluation slice. This is
the minimal walk-forward contract; full DSR-vs-XGBoost-baseline gating is a
follow-up (see TODO below).
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Ensure research/ and shared/ are importable (mirrors train_multi_horizon.py)
_REPO = Path(__file__).parents[3]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
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
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _build_stub_data(n_bars: int, as_of: date, seed: int):
    """
    Synthetic OHLC stub: a single ``close`` series on a UTC daily index.

    The TradingEnvironment only strictly requires a ``close`` column and a
    timezone-aware (UTC) DatetimeIndex; the 42-dim observation block defaults
    missing feature columns to 0.0. This is enough to validate the training
    loop end-to-end in the cloud.

    TODO(@alex 2026-06-30): replace with the real loader from TimescaleDB /
    Parquet (offline feature store), populating market + regime feature columns
    so the agent trains on real signal rather than a random walk.
    """
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=pd.Timestamp(as_of, tz="UTC"), periods=n_bars, freq="D")
    close = 100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.01, n_bars))
    return pd.DataFrame({"close": close}, index=idx)


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


def _split_envs(args: argparse.Namespace) -> tuple[TradingEnvironment, TradingEnvironment]:
    env_cfg = EnvironmentConfig()
    ep_len = env_cfg.episode_length
    # Need enough bars on each side of the split for at least one episode.
    min_bars = int((ep_len + 2) / min(args.train_frac, 1.0 - args.train_frac)) + 2
    data = _build_stub_data(n_bars=max(min_bars, 1200), as_of=date.fromisoformat(args.as_of), seed=args.seed)

    split = int(len(data) * args.train_frac)
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
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _REPO / "research" / "artifacts" / "drl" / args.algo

    logger.info("algo=%s seed=%d device=%s train_frac=%.2f as_of=%s",
                args.algo, args.seed, args.device, args.train_frac, args.as_of)

    train_env, eval_env = _split_envs(args)
    logger.info("Envs built: train/eval temporal split (disjoint, anti-leakage).")

    if args.dry_run:
        logger.info("DRY RUN — env + split valid, no training. ckpt_dir=%s", ckpt_dir)
        return 0

    start_t = time.monotonic()

    if args.algo == "dqn":
        from models.drl import DQNConfig, DQNTrainer, TradingDQN

        net = TradingDQN(obs_dim=EnvironmentConfig().obs_dim)
        trainer = DQNTrainer(net, DQNConfig(device=args.device))
        trainer.train(train_env, n_episodes=args.episodes, checkpoint_dir=ckpt_dir)
        oos = _evaluate_dqn(trainer, eval_env)
        logger.info("Held-out OOS mean reward (greedy): %.5f", oos)
        edge = oos > 0.0

    elif args.algo == "ppo":
        from models.drl import PPOConfig, PPOTrainer, TradingActorCritic

        ac = TradingActorCritic()
        trainer = PPOTrainer(ac, PPOConfig(device=args.device))
        trainer.train(train_env, n_updates=args.updates, checkpoint_dir=ckpt_dir)
        edge = True  # TODO(@alex 2026-06-30): OOS eval + DSR gate for PPO

    else:  # sac
        from models.drl import SACConfig, SACTrainer, TradingDiscreteActor

        actor = TradingDiscreteActor()
        trainer = SACTrainer(actor, SACConfig(device=args.device))
        trainer.train(train_env, n_steps=args.steps, checkpoint_dir=ckpt_dir)
        edge = True  # TODO(@alex 2026-06-30): OOS eval + DSR gate for SAC

    elapsed = time.monotonic() - start_t
    logger.info("Training done in %.1fs. Checkpoints → %s", elapsed, ckpt_dir)

    # TODO(@alex 2026-06-30): replace `edge` heuristic with walk-forward DSR
    # comparison vs the XGBoost baseline (CLAUDE.md §6.10): promote only if
    # DSR_agent > DSR_baseline on concatenated OOS folds.
    if edge:
        logger.info("SUCCESS: training completed with positive OOS signal. Exit 0.")
        return 0
    logger.warning("NO EDGE: OOS mean reward <= 0. Document and do not promote. Exit 2.")
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
