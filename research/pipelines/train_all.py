"""Master training driver — trains EVERYTHING on the chosen 4H universe.

Per asset (BTC/USD, ETH/USD, EUR/USD @ 4H) it runs, on the SAME anti-leakage
walk-forward folds:

* DRL gate for each of {dqn, ppo, sac}  → ``models.drl.multi_algo_gate.run_gate``
  (each verdict already compares the agent vs XGBoost baseline vs buy-and-hold,
  ADR-040; XGBoost is therefore always trained too).
* Supervised deep baselines {res_mlp, lstm} → ``supervised_oos_returns`` scored
  with the same gate vs buy-and-hold + XGBoost.

Then once across configured pairs (default BTC/USD–ETH/USD):

* Stat-arb (market-neutral) → ``alpha.statarb`` + ``evaluate_zero_gate`` (ADR-043).

Everything reuses the audited research libraries; this driver only sequences
them, annualizes at the correct 4H factor (crypto 2190 / FX 1560), handles
per-stage failures without sinking the run, and writes a consolidated JSON +
console report. Cross-algo target (ADR-039): DSR(SAC) > DSR(PPO) > DSR(XGBoost).

Heavy deps (torch/xgboost/gymnasium/sklearn) are imported lazily inside the
stage functions, so ``--help`` / ``--dry-run`` work on a bare interpreter.

Usage
-----
::

    cd research                         # ALPACA_API_KEY/SECRET in env for crypto
    python -m pipelines.train_all                      # full run, all stages
    python -m pipelines.train_all --smoke              # tiny wiring check
    python -m pipelines.train_all --stages drl --algos ppo,sac --n-jobs 4
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from experiments.btc_eth_eur_4h import config as exp_config
from experiments.btc_eth_eur_4h import data_sources

logger = logging.getLogger(__name__)

DEFAULT_ASSETS = ("BTC/USD", "ETH/USD", "EUR/USD")
DEFAULT_ALGOS = ("dqn", "ppo", "sac")
DEFAULT_SUPERVISED = ("res_mlp", "lstm")
DEFAULT_PAIRS = (("BTC/USD", "ETH/USD"),)
DEFAULT_STAGES = ("drl", "supervised", "statarb")


@dataclass
class TrainAllConfig:
    """Configuration for the master training run."""

    assets: tuple[str, ...] = DEFAULT_ASSETS
    algos: tuple[str, ...] = DEFAULT_ALGOS
    supervised_models: tuple[str, ...] = DEFAULT_SUPERVISED
    pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS
    stages: tuple[str, ...] = DEFAULT_STAGES
    timeframe: str = "4h"
    n_folds: int = 4
    n_seeds: int = 5
    episodes: int = 150
    episode_length: int = 180
    fee_bps: float = 5.0
    dsr_threshold: float = 0.4
    device: str = "cpu"
    n_jobs: int = 1
    seed: int = 42
    crypto_start: str = "2021-01-01"
    fx_lookback_days: int = 700
    end: Optional[str] = None
    feed: str = "iex"
    out_dir: str = "artifacts/runs"
    # stat-arb pair splitter (4H crypto bars)
    pair_train_size: int = 2000
    pair_test_size: int = 500
    pair_embargo: int = 60

    @classmethod
    def smoke(cls) -> "TrainAllConfig":
        return cls(
            n_folds=2, n_seeds=1, episodes=2, episode_length=40,
            supervised_models=("res_mlp",), fx_lookback_days=120,
            pair_train_size=300, pair_test_size=120,
        )


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def _load_asset(symbol: str, cfg: TrainAllConfig):
    """Load raw 4H OHLCV for ``symbol`` via the unified loader (crypto + FX)."""
    exp = exp_config.ExperimentConfig(
        crypto_start=cfg.crypto_start, fx_lookback_days=cfg.fx_lookback_days, end=cfg.end
    )
    start, end = exp.start_for(symbol), exp.end_date()
    return data_sources.load_raw_ohlcv(symbol, start, end, timeframe=cfg.timeframe, feed=cfg.feed)


def _env_cfg(cfg: TrainAllConfig):
    from envs import EnvironmentConfig

    return EnvironmentConfig(
        fee_bps=cfg.fee_bps, episode_length=cfg.episode_length, reward_mode="mtm"
    )


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def run_drl_for_asset(symbol: str, ohlcv, cfg: TrainAllConfig) -> dict[str, Any]:
    """Run the DRL gate for every configured algo on one asset."""
    from models.drl.multi_algo_gate import run_gate

    ppy = exp_config.periods_per_year(symbol, cfg.timeframe)
    env_cfg = _env_cfg(cfg)
    seeds = list(range(cfg.seed, cfg.seed + cfg.n_seeds))
    out: dict[str, Any] = {}
    for algo in cfg.algos:
        try:
            out[algo] = run_gate(
                ohlcv, algo,
                n_folds=cfg.n_folds, episodes=cfg.episodes, seeds=seeds,
                env_cfg=env_cfg, dsr_threshold=cfg.dsr_threshold,
                periods_per_year=ppy, device=cfg.device, n_jobs=cfg.n_jobs,
            )
        except Exception as exc:  # noqa: BLE001 — one algo must not sink the asset
            logger.exception("DRL %s/%s failed", symbol, algo)
            out[algo] = {"algo": algo, "error": f"{type(exc).__name__}: {exc}"}
    return out


def run_supervised_for_asset(symbol: str, ohlcv, cfg: TrainAllConfig) -> dict[str, Any]:
    """Score supervised deep baselines with the same gate (vs buy-and-hold + XGB)."""
    from models.drl.dsr_gate import (
        buyhold_oos_returns,
        evaluate_drl_gate,
        make_wf_splitter,
        xgb_oos_returns,
    )
    from models.drl.multi_algo_gate import _ann_sharpe, supervised_oos_returns

    ppy = exp_config.periods_per_year(symbol, cfg.timeframe)
    env_cfg = _env_cfg(cfg)
    splitter = make_wf_splitter(ohlcv, cfg.n_folds, env_cfg=env_cfg)
    xgb_r = xgb_oos_returns(ohlcv, splitter, fee_bps=cfg.fee_bps, seed=cfg.seed)
    bh_r = buyhold_oos_returns(ohlcv, splitter, fee_bps=cfg.fee_bps)

    out: dict[str, Any] = {}
    for model_name in cfg.supervised_models:
        try:
            r = supervised_oos_returns(ohlcv, splitter, model_name, fee_bps=cfg.fee_bps, seed=cfg.seed)
            gate = evaluate_drl_gate(
                r, bh_r, xgb_r, n_trials=1,
                dsr_threshold=cfg.dsr_threshold, periods_per_year=ppy,
            )
            out[model_name] = {
                "model": model_name,
                "dsr": gate.dsr_agent,
                "sharpe": _ann_sharpe(r, ppy),
                "sharpe_buyhold": gate.sharpe_buyhold,
                "dsr_xgb": gate.dsr_xgb,
                "n_oos_bars": gate.n_oos_bars,
                "passed": gate.passed,
                "reason": gate.reason,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("supervised %s/%s failed", symbol, model_name)
            out[model_name] = {"model": model_name, "error": f"{type(exc).__name__}: {exc}"}
    return out


def run_statarb(cfg: TrainAllConfig) -> dict[str, Any]:
    """Market-neutral stat-arb gate (vs ZERO, ADR-043) over configured pairs."""
    from alpha.statarb.pairs import PairStatArb, PairStatArbConfig, walk_forward_pair_returns
    from models.drl.dsr_gate import evaluate_zero_gate
    from models.validation import WalkForwardSplitter

    ppy = exp_config.CRYPTO_PPY_4H  # configured pairs are crypto 24/7
    out: dict[str, Any] = {}
    for y_sym, x_sym in cfg.pairs:
        key = f"{y_sym}~{x_sym}"
        try:
            oy = _load_asset(y_sym, cfg)
            ox = _load_asset(x_sym, cfg)
            prices = (
                oy[["close"]].rename(columns={"close": y_sym})
                .join(ox[["close"]].rename(columns={"close": x_sym}), how="inner")
                .dropna()
            )
            strat = PairStatArb(y_sym, x_sym, PairStatArbConfig(fee_bps=cfg.fee_bps))
            splitter = WalkForwardSplitter(
                train_size=cfg.pair_train_size, test_size=cfg.pair_test_size,
                expanding=True, embargo=cfg.pair_embargo,
            )
            r = walk_forward_pair_returns(prices, splitter, strat)
            gate = evaluate_zero_gate(
                r, n_trials=len(cfg.pairs),
                dsr_threshold=cfg.dsr_threshold, periods_per_year=ppy,
            )
            out[key] = {
                "pair": key, "n_bars": int(len(prices)),
                "dsr": gate.dsr_agent, "sharpe": gate.sharpe_agent,
                "n_oos_bars": gate.n_oos_bars, "passed": gate.passed, "reason": gate.reason,
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("statarb %s failed", key)
            out[key] = {"pair": key, "error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- #
# Orchestration + reporting
# --------------------------------------------------------------------------- #


def run_all(cfg: TrainAllConfig) -> dict[str, Any]:
    """Execute the configured stages and return the consolidated report."""
    results: dict[str, Any] = {"drl": {}, "supervised": {}, "statarb": {}}

    if "drl" in cfg.stages or "supervised" in cfg.stages:
        for symbol in cfg.assets:
            logger.info("=== loading %s ===", symbol)
            try:
                ohlcv = _load_asset(symbol, cfg)
            except Exception as exc:  # noqa: BLE001
                logger.exception("load %s failed", symbol)
                err = {"error": f"{type(exc).__name__}: {exc}"}
                if "drl" in cfg.stages:
                    results["drl"][symbol] = err
                if "supervised" in cfg.stages:
                    results["supervised"][symbol] = err
                continue
            if "drl" in cfg.stages:
                logger.info("=== DRL %s ===", symbol)
                results["drl"][symbol] = run_drl_for_asset(symbol, ohlcv, cfg)
            if "supervised" in cfg.stages:
                logger.info("=== supervised %s ===", symbol)
                results["supervised"][symbol] = run_supervised_for_asset(symbol, ohlcv, cfg)

    if "statarb" in cfg.stages:
        logger.info("=== stat-arb ===")
        results["statarb"] = run_statarb(cfg)

    report = {
        "experiment": "train_all",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": _config_to_jsonable(cfg),
        "results": results,
    }
    print("\n" + _format_report(results) + "\n")
    return report


def _config_to_jsonable(cfg: TrainAllConfig) -> dict[str, Any]:
    d = asdict(cfg)
    d["pairs"] = [list(p) for p in cfg.pairs]  # tuples -> lists for JSON
    return d


def _format_report(results: dict[str, Any]) -> str:
    lines: list[str] = []
    if results.get("drl"):
        lines.append("DRL gate (agent vs XGBoost vs buy&hold)")
        lines.append(f"  {'asset':<9}{'algo':<6}{'DSR_ag':>8}{'S_ag':>7}{'DSR_xgb':>8}{'S_bh':>7}  gate")
        for asset, algos in results["drl"].items():
            if "error" in algos:
                lines.append(f"  {asset:<9} ERROR: {algos['error']}")
                continue
            for algo, r in algos.items():
                if "error" in r:
                    lines.append(f"  {asset:<9}{algo:<6} ERROR: {r['error']}")
                    continue
                lines.append(
                    f"  {asset:<9}{algo:<6}{r['dsr_agent']:>8.3f}{r['sharpe_agent']:>7.2f}"
                    f"{r['dsr_xgb']:>8.3f}{r['sharpe_buyhold']:>7.2f}  "
                    f"{'PASS' if r['passed'] else 'FAIL'}"
                )
    if results.get("supervised"):
        lines.append("Supervised (vs XGBoost baseline)")
        for asset, models in results["supervised"].items():
            if isinstance(models, dict) and "error" in models:
                lines.append(f"  {asset:<9} ERROR: {models['error']}")
                continue
            for m, r in models.items():
                if "error" in r:
                    lines.append(f"  {asset:<9}{m:<8} ERROR: {r['error']}")
                else:
                    lines.append(
                        f"  {asset:<9}{m:<8} DSR={r['dsr']:.3f} S={r['sharpe']:.2f} "
                        f"{'PASS' if r['passed'] else 'FAIL'}"
                    )
    if results.get("statarb"):
        lines.append("Stat-arb (vs ZERO)")
        for key, r in results["statarb"].items():
            if "error" in r:
                lines.append(f"  {key} ERROR: {r['error']}")
            else:
                lines.append(
                    f"  {key}  DSR={r['dsr']:.3f} S={r['sharpe']:.2f} "
                    f"{'PASS' if r['passed'] else 'FAIL'}"
                )
    return "\n".join(lines) if lines else "(no results)"


def write_report(report: dict[str, Any], out_dir: str) -> Path:
    base = Path(__file__).resolve().parents[1]  # research/
    target = base / out_dir
    target.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"train_all_{ts}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("report written -> %s", path)
    return path


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--assets", type=str, default=",".join(DEFAULT_ASSETS))
    p.add_argument("--algos", type=str, default=",".join(DEFAULT_ALGOS))
    p.add_argument("--supervised", type=str, default=",".join(DEFAULT_SUPERVISED))
    p.add_argument("--stages", type=str, default=",".join(DEFAULT_STAGES))
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--episode-length", type=int, default=180)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--dsr-threshold", type=float, default=0.4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--crypto-start", type=str, default="2021-01-01")
    p.add_argument("--fx-lookback-days", type=int, default=700)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="artifacts/runs")
    p.add_argument("--smoke", action="store_true", help="tiny wiring config")
    p.add_argument("--dry-run", action="store_true", help="print config and exit")
    return p.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> TrainAllConfig:
    base = TrainAllConfig.smoke() if args.smoke else TrainAllConfig()
    base.assets = tuple(s.strip() for s in args.assets.split(",") if s.strip())
    base.algos = tuple(s.strip() for s in args.algos.split(",") if s.strip())
    base.supervised_models = tuple(s.strip() for s in args.supervised.split(",") if s.strip())
    base.stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    if not args.smoke:
        base.n_folds = args.n_folds
        base.n_seeds = args.n_seeds
        base.episodes = args.episodes
        base.episode_length = args.episode_length
    base.fee_bps = args.fee_bps
    base.dsr_threshold = args.dsr_threshold
    base.device = args.device
    base.n_jobs = args.n_jobs
    base.crypto_start = args.crypto_start
    base.fx_lookback_days = args.fx_lookback_days
    base.end = args.end
    base.out_dir = args.out_dir
    return base


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)
    cfg = _config_from_args(args)
    if args.dry_run:
        print(json.dumps(_config_to_jsonable(cfg), indent=2))
        return 0
    report = run_all(cfg)
    write_report(report, cfg.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
