"""
CLI entry point for the nightly retraining DAG (S11).

Usage:
    python -m research.cli.run_nightly_retrain [OPTIONS]

Options:
    --as-of DATE        Point-in-time date ISO (default: today).
    --seed INT          Random seed (default: 42).
    --horizons STR      Comma-separated: intraday,swing,daily (default: all).
    --n-trials INT      Optuna trials per horizon (default: 50).
    --dsr-min FLOAT     Absolute DSR floor gate (default: 0.4).
    --dsr-delta FLOAT   New DSR >= prod_DSR * dsr_delta (default: 0.95).
    --ece-max FLOAT     ECE ceiling gate (default: 0.05).
    --dry-run           Validate config; skip training.
    --run-log-dir PATH  Directory for JSON run logs (default: artifacts/runs).

Exit codes:
    0 — at least 1 horizon promoted to staging
    2 — no horizons promoted (no edge detected in current data)
    1 — runtime or configuration error
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).parents[3]
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipelines.nightly_retrain import NightlyRetrainConfig, NightlyRetrainDAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_nightly_retrain")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="S11 — Nightly retraining DAG",
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
        help="Comma-separated horizon names",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--dsr-min", type=float, default=0.4)
    parser.add_argument("--dsr-delta", type=float, default=0.95)
    parser.add_argument("--ece-max", type=float, default=0.05)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run-log-dir",
        type=str,
        default="artifacts/runs",
        help="Directory for JSON run logs",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> int:
    config = NightlyRetrainConfig(
        as_of=date.fromisoformat(args.as_of),
        seed=args.seed,
        horizons=[h.strip() for h in args.horizons.split(",") if h.strip()],
        n_trials=args.n_trials,
        dsr_min=args.dsr_min,
        dsr_delta=args.dsr_delta,
        ece_max=args.ece_max,
        dry_run=args.dry_run,
        run_log_dir=Path(args.run_log_dir),
    )

    if not config.horizons:
        logger.error("No valid horizons provided. Use: intraday,swing,daily")
        return 1

    try:
        from quant_shared.models.registry import ModelRegistry
        registry = ModelRegistry()
        logger.info("Model registry loaded from %s", registry._path)
    except Exception as exc:
        logger.warning("ModelRegistry unavailable: %s — continuing without registry.", exc)
        from quant_shared.models.registry import ModelRegistry
        registry = ModelRegistry()

    dag = NightlyRetrainDAG(config=config, registry=registry)
    run_log = await dag.run()

    if run_log.exit_code == 0:
        logger.info(
            "SUCCESS: %d/%d horizons promoted to staging. run_id=%s",
            run_log.n_promoted, len(config.horizons), run_log.run_id,
        )
    else:
        logger.warning(
            "NO EDGE: 0/%d horizons promoted. "
            "Document as 'no edge in current data'. run_id=%s",
            len(config.horizons), run_log.run_id,
        )
    return run_log.exit_code


def main() -> None:
    args = _parse_args()
    exit_code = asyncio.run(_run(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
