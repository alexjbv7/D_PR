"""Orchestrate XGBoost-vs-DQN on BTC/USD, ETH/USD, EUR/USD (4H, ADR-040 gate).

Per asset, on the SAME walk-forward folds (embargo >= 60 bars, regime GMM
re-fitted per fold — anti-leakage), it computes concatenated OOS per-bar
returns for three strategies and applies the ADR-040 promotion gate:

* DQN          -> ``dsr_gate.walk_forward_oos_returns`` (N seeds; CI from
                  their dispersion, agent DSR deflated by ``n_seeds``).
* XGBoost      -> ``dsr_gate.xgb_oos_returns`` (supervised baseline).
* Buy-and-hold -> ``dsr_gate.buyhold_oos_returns``.

The only experiment-specific logic is (a) the unified 4H loader and (b) the 4H
annualization factor (``config.periods_per_year``) passed to
``evaluate_drl_gate`` in place of its daily default of 252. Nothing statistical
is reimplemented (CLAUDE.md §20.2).

Heavy deps (torch via the DQN trainer, xgboost, gymnasium) are imported lazily
inside ``run_one_asset`` so ``--help`` and ``--dry-run`` work without them.

Usage
-----
::

    # from research/ with ALPACA_API_KEY / ALPACA_API_SECRET in env:
    python -m experiments.btc_eth_eur_4h.run_experiment
    python -m experiments.btc_eth_eur_4h.run_experiment \\
        --symbols BTC/USD,ETH/USD --episodes 150 --n-seeds 5 --n-folds 4
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from . import data_sources
from .config import ExperimentConfig, periods_per_year

logger = logging.getLogger(__name__)


def _annualized_sharpe(returns: np.ndarray, ppy: int) -> float:
    """Annualized Sharpe at ``ppy`` bars/year; 0.0 if flat, NaN if too short."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 2:
        return float("nan")
    sigma = float(r.std(ddof=1))
    if sigma < 1e-12:
        return 0.0
    return float(r.mean() / sigma * np.sqrt(ppy))


def _median_seed(sharpes: list[float], returns: list[np.ndarray]) -> int:
    """Index of the seed with the MEDIAN Sharpe (avoids cherry-picking the max).

    Reporting the median run rather than the best is the honest summary for a
    high-variance estimator; the DSR is separately deflated by the seed count.
    """
    order = sorted(range(len(sharpes)), key=lambda i: sharpes[i])
    return order[(len(order) - 1) // 2]


def run_one_asset(
    symbol: str,
    cfg: ExperimentConfig,
    *,
    ohlcv: Optional[pd.DataFrame] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict[str, Any]:
    """Run the full XGBoost/DQN/buy-and-hold gate for one instrument.

    Parameters
    ----------
    symbol : str
        Instrument (``"BTC/USD"`` / ``"EUR/USD"``).
    cfg : ExperimentConfig
        Experiment parameters.
    ohlcv : pd.DataFrame, optional
        Pre-loaded raw OHLCV. If ``None`` it is fetched via
        ``data_sources.load_raw_ohlcv`` (module attribute access keeps it
        monkeypatchable in tests/smoke).
    start, end : str, optional
        Override the window (defaults derive from ``cfg``).

    Returns
    -------
    dict
        Metrics + the ADR-040 verdict for this asset.
    """
    # Lazy heavy imports (torch/xgboost/gymnasium) — keep CLI import-light.
    from envs import EnvironmentConfig
    from models.drl.dsr_gate import (
        AgentSpec,
        buyhold_oos_returns,
        evaluate_drl_gate,
        make_wf_splitter,
        walk_forward_oos_returns,
        xgb_oos_returns,
    )
    from models.drl.e1_baseline_comparison import sharpe_ci_from_seeds

    if ohlcv is None:
        start = start or cfg.start_for(symbol)
        end = end or cfg.end_date()
        ohlcv = data_sources.load_raw_ohlcv(
            symbol, start, end, timeframe=cfg.timeframe, feed=cfg.feed
        )

    ppy = periods_per_year(symbol, cfg.timeframe)
    env_cfg = EnvironmentConfig(
        fee_bps=cfg.fee_bps,
        episode_length=cfg.episode_length,
        reward_mode="mtm",
    )
    splitter = make_wf_splitter(ohlcv, cfg.n_folds, env_cfg=env_cfg)

    # Baselines (deterministic; same folds/embargo/return-def as the agent).
    xgb_r = xgb_oos_returns(ohlcv, splitter, fee_bps=cfg.fee_bps, seed=cfg.seed)
    bh_r = buyhold_oos_returns(ohlcv, splitter, fee_bps=cfg.fee_bps)

    # DQN across seeds (the high-variance estimator; CI from seed dispersion).
    seeds = list(range(cfg.seed, cfg.seed + cfg.n_seeds))
    dqn_returns: list[np.ndarray] = []
    dqn_sharpes: list[float] = []
    for s in seeds:
        spec = AgentSpec(
            algo="dqn", episodes=cfg.episodes, seed=s, device=cfg.device
        )
        r = walk_forward_oos_returns(
            spec, ohlcv, splitter, env_cfg, seed=s, n_jobs=cfg.n_jobs
        )
        dqn_returns.append(r)
        dqn_sharpes.append(_annualized_sharpe(r, ppy))
        logger.info("DQN %s seed=%d sharpe=%.3f oos=%d", symbol, s, dqn_sharpes[-1], len(r))

    med = _median_seed(dqn_sharpes, dqn_returns)
    agent_r = dqn_returns[med]
    mean_sh, lb95, p95 = sharpe_ci_from_seeds(dqn_sharpes)

    gate = evaluate_drl_gate(
        agent_r,
        bh_r,
        xgb_r,
        n_trials=len(seeds),
        dsr_threshold=cfg.dsr_threshold,
        periods_per_year=ppy,
    )

    return {
        "symbol": symbol,
        "timeframe": cfg.timeframe,
        "periods_per_year": ppy,
        "n_bars": int(len(ohlcv)),
        "n_oos_bars": gate.n_oos_bars,
        "n_seeds": len(seeds),
        "dqn_sharpe_median": float(dqn_sharpes[med]),
        "dqn_sharpe_mean": float(mean_sh),
        "dqn_sharpe_lb95": float(lb95),
        "dqn_sharpe_p95": float(p95),
        "dsr_agent": gate.dsr_agent,
        "psr_agent": gate.psr_agent,
        "sharpe_agent": gate.sharpe_agent,
        "sharpe_buyhold": gate.sharpe_buyhold,
        "sharpe_xgb": _annualized_sharpe(xgb_r, ppy),
        "dsr_xgb": gate.dsr_xgb,
        "passed": gate.passed,
        "reason": gate.reason,
    }


def _format_table(results: list[dict[str, Any]]) -> str:
    """Render the per-asset comparison as a fixed-width table."""
    header = (
        f"{'symbol':<9} {'oos':>5} {'S_dqn':>7} {'DSR_dqn':>8} "
        f"{'S_xgb':>7} {'DSR_xgb':>8} {'S_bh':>7} {'gate':>6}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        if "error" in r:
            lines.append(f"{r['symbol']:<9} ERROR: {r['error']}")
            continue
        lines.append(
            f"{r['symbol']:<9} {r['n_oos_bars']:>5} "
            f"{r['dqn_sharpe_median']:>7.2f} {r['dsr_agent']:>8.3f} "
            f"{r['sharpe_xgb']:>7.2f} {r['dsr_xgb']:>8.3f} "
            f"{r['sharpe_buyhold']:>7.2f} {'PASS' if r['passed'] else 'FAIL':>6}"
        )
    return "\n".join(lines)


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    """Run all configured assets and return the aggregated report dict."""
    results: list[dict[str, Any]] = []
    for symbol in cfg.symbols:
        logger.info("=== %s (%s) ===", symbol, cfg.timeframe)
        try:
            results.append(run_one_asset(symbol, cfg))
        except Exception as exc:  # one asset failing must not sink the others
            logger.exception("asset %s failed", symbol)
            results.append({"symbol": symbol, "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "experiment": "btc_eth_eur_4h",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(cfg),
        "results": results,
    }
    print("\n" + _format_table(results) + "\n")
    return report


def _write_report(report: dict[str, Any], out_dir: str) -> Path:
    """Persist the run report as timestamped JSON; return the path."""
    base = Path(__file__).resolve().parents[2]  # research/
    target = base / out_dir
    target.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"exp_btc_eth_eur_4h_{ts}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("report written -> %s", path)
    return path


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--symbols", type=str, default="BTC/USD,ETH/USD,EUR/USD")
    p.add_argument("--timeframe", type=str, default="4h")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=4)
    p.add_argument("--episode-length", type=int, default=180)
    p.add_argument("--dsr-threshold", type=float, default=0.4)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--crypto-start", type=str, default="2021-01-01")
    p.add_argument("--fx-lookback-days", type=int, default=700)
    p.add_argument("--end", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="artifacts/runs")
    p.add_argument("--smoke", action="store_true", help="tiny config for a wiring check")
    p.add_argument("--dry-run", action="store_true", help="print config and exit")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = _parse_args(argv)

    if args.smoke:
        cfg = ExperimentConfig.smoke()
        cfg.symbols = tuple(s.strip() for s in args.symbols.split(","))
    else:
        cfg = ExperimentConfig(
            symbols=tuple(s.strip() for s in args.symbols.split(",")),
            timeframe=args.timeframe,
            crypto_start=args.crypto_start,
            fx_lookback_days=args.fx_lookback_days,
            end=args.end,
            n_folds=args.n_folds,
            n_seeds=args.n_seeds,
            episodes=args.episodes,
            dsr_threshold=args.dsr_threshold,
            episode_length=args.episode_length,
            device=args.device,
            n_jobs=args.n_jobs,
            out_dir=args.out_dir,
        )

    if args.dry_run:
        print(json.dumps(asdict(cfg), indent=2))
        return 0

    report = run_experiment(cfg)
    _write_report(report, cfg.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
