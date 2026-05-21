"""
CLI entry point for multi-horizon training.

Usage:
    python -m research.cli.train_multi_horizon --as-of 2026-05-01 --seed 42

Options:
    --as-of         Point-in-time date (ISO format). Default: today.
    --seed          Random seed. Default: 42.
    --horizons      Comma-separated subset: intraday,swing,daily. Default: all.
    --n-trials      Override Optuna trials per horizon.
    --no-ablate     Skip ablative analysis (faster).
    --dry-run       Build configs and validate; do not train.

Time budget:
    If run exceeds 12h wall-clock, script will log a WARNING and continue.
    To stay under budget, reduce --n-trials to 25 and the universe to 25 symbols.

Exits with code 0 if >= 2 horizons achieve DSR >= 0.4.
Exits with code 2 if < 2 horizons achieve DSR >= 0.4 (no edge).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Ensure research/ and shared/ are importable
_REPO = Path(__file__).parents[3]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.multi_horizon.horizon_config import ALL_HORIZONS, HorizonConfig
from models.multi_horizon.trainer import MultiHorizonTrainer, TrainResult
from models.multi_horizon.reports import write_ablative_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_multi_horizon")

_WALL_CLOCK_LIMIT_SECONDS = 12 * 3600


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semana 7 — Multi-Horizon Trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default=date.today().isoformat(),
        help="Point-in-time date (ISO format)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--horizons",
        type=str,
        default="intraday,swing,daily",
        help="Comma-separated horizon names to train",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=None,
        help="Override Optuna trials per horizon (default: from HorizonConfig)",
    )
    parser.add_argument(
        "--no-ablate",
        action="store_true",
        help="Skip ablative analysis",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config only; do not train",
    )
    return parser.parse_args()


def _build_stub_datasets(
    horizons: list[HorizonConfig],
    as_of: date,
) -> dict[str, tuple]:
    """
    Build synthetic stub datasets when real data is not available.

    Replace with real data loading from TimescaleDB / Parquet in production.
    """
    import numpy as np
    import pandas as pd

    datasets: dict[str, tuple] = {}
    bar_sizes = {"intraday": "5min", "swing": "4H", "daily": "B"}
    n_bars    = {"intraday": 1000,   "swing": 800,  "daily": 500}

    for cfg in horizons:
        rng = np.random.default_rng(42 + HORIZON_ORDER.index(cfg.name))
        freq = bar_sizes[cfg.name]
        n    = n_bars[cfg.name]

        from models.multi_horizon.feature_sets import get_feature_set

        feature_names = get_feature_set(cfg.feature_set)
        idx = pd.date_range(
            end=pd.Timestamp(as_of),
            periods=n,
            freq=freq,
        )
        X = pd.DataFrame(
            rng.standard_normal((n, len(feature_names))),
            index=idx,
            columns=feature_names,
        )
        # Regime probs must be positive and sum to 1
        regime_cols = [c for c in feature_names if c.startswith("regime_prob_")]
        if regime_cols:
            raw = X[regime_cols].abs()
            X[regime_cols] = raw.div(raw.sum(axis=1), axis=0)
        # Session flags
        for sc in ("session_pre", "session_rth", "session_post"):
            if sc in X.columns:
                X[sc] = 0
        if "session_rth" in X.columns:
            X["session_rth"] = 1

        y = pd.Series(rng.choice([-1, 0, 1], size=n), index=idx)
        prices = pd.Series(
            100.0 * (1 + rng.normal(0, 0.01, n)).cumprod(), index=idx
        )
        datasets[cfg.name] = (X, y, prices)

    return datasets


HORIZON_ORDER = ["intraday", "swing", "daily"]


async def _train(args: argparse.Namespace) -> int:
    as_of = date.fromisoformat(args.as_of)
    requested = {h.strip() for h in args.horizons.split(",")}
    selected_cfgs = [cfg for cfg in ALL_HORIZONS if cfg.name in requested]

    if not selected_cfgs:
        logger.error("No valid horizons selected. Choose from: intraday, swing, daily")
        return 1

    # Override n_optuna_trials if requested
    if args.n_trials is not None:
        from dataclasses import replace

        selected_cfgs = [
            replace(cfg, n_optuna_trials=args.n_trials) for cfg in selected_cfgs
        ]

    logger.info("Training horizons: %s  as_of=%s  seed=%d",
                [c.name for c in selected_cfgs], as_of, args.seed)

    if args.dry_run:
        logger.info("DRY RUN — configs valid. No training performed.")
        for cfg in selected_cfgs:
            logger.info(
                "  %s: bar=%s  embargo=%s  lookback=%s  n_trials=%d",
                cfg.name, cfg.bar_size, cfg.embargo, cfg.train_lookback, cfg.n_optuna_trials,
            )
        return 0

    # Load data — stub for now; replace with real loader
    logger.info("Loading datasets (stub)...")
    datasets = _build_stub_datasets(selected_cfgs, as_of)

    # Registry
    try:
        from quant_shared.models.registry import ModelRegistry

        registry = ModelRegistry()
        logger.info("Model registry loaded.")
    except Exception as exc:
        logger.warning("Could not load ModelRegistry: %s — results will not be registered.", exc)
        registry = None

    trainer = MultiHorizonTrainer(
        seed=args.seed,
        max_parallel=2,
        n_wf_splits=8,
        ablate=not args.no_ablate,
    )

    start_t = time.monotonic()
    results: dict[str, TrainResult] = await trainer.run_all_horizons(
        horizon_datasets=datasets,
        as_of=as_of,
        symbols=["STUB_SYMBOL"],
        registry=registry,
    )
    elapsed = time.monotonic() - start_t

    if elapsed > _WALL_CLOCK_LIMIT_SECONDS:
        logger.warning(
            "Wall-clock budget exceeded: %.1fh > 12h. "
            "Consider reducing --n-trials to 25 and universe to 25 symbols.",
            elapsed / 3600,
        )

    # Write cross-horizon ablative summary
    ablative_data: dict[str, dict[str, float]] = {
        name: r.ablative_dsrs for name, r in results.items()
    }
    write_ablative_summary(ablative_data)

    # Determine exit code
    promoted_count = sum(1 for r in results.values() if r.promoted)
    if promoted_count >= 2:
        logger.info("SUCCESS: %d/3 horizons promoted. Exit 0.", promoted_count)
        return 0
    else:
        logger.warning(
            "NO EDGE: only %d/3 horizons achieved DSR >= 0.4. "
            "Document as 'no edge in current data'. Exit 2.",
            promoted_count,
        )
        return 2


def main() -> None:
    args = _parse_args()
    exit_code = asyncio.run(_train(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
