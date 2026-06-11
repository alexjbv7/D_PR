"""
CLI: gate vs ZERO para un par de stat-arb (ADR-043 §6, DoD).

Usage:
    python -m research.cli.run_pair_gate --y SPY --x QQQ --start 2018-01-01

Trae las dos patas con ``data.drl_dataset.fetch_ohlcv_frame`` (Alpaca; lee
ALPACA_API_KEY / ALPACA_API_SECRET del entorno), corre el walk-forward
anti-leakage de ``alpha.statarb.pairs`` y evalúa ``evaluate_zero_gate``.

Deflación honesta: ``--n-trials`` = nº de pares/configs que has evaluado OOS
en total — si pruebas 10 pares y reportas el mejor con n_trials=1, el DSR
está inflado (ADR-043 §9, CLAUDE.md §6.10).

Exit codes: 0 = gate PASS, 2 = gate FAIL, 1 = error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

_REPO = Path(__file__).parents[2]  # cli → research → quant_bot
for _p in [str(_REPO / "research"), str(_REPO / "shared")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_pair_gate")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ADR-043 — gate vs ZERO para stat-arb de pares",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--y", type=str, default="SPY", help="Pata regresada")
    parser.add_argument("--x", type=str, default="QQQ", help="Pata de hedge")
    parser.add_argument("--start", type=str, default="2018-01-01")
    parser.add_argument("--end", type=str, default=date.today().isoformat())
    parser.add_argument("--feed", type=str, default="iex")
    parser.add_argument("--entry-z", type=float, default=2.0)
    parser.add_argument("--exit-z", type=float, default=0.5)
    parser.add_argument("--fee-bps", type=float, default=5.0)
    parser.add_argument("--max-half-life", type=float, default=30.0)
    parser.add_argument("--train-size", type=int, default=600)
    parser.add_argument("--test-size", type=int, default=150)
    parser.add_argument("--embargo", type=int, default=60)
    parser.add_argument(
        "--n-trials", type=int, default=1,
        help="Pares/configs evaluados OOS en total (deflación del DSR)",
    )
    parser.add_argument("--dsr-threshold", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    from alpha.statarb.pairs import (
        PairStatArb,
        PairStatArbConfig,
        walk_forward_pair_returns,
    )
    from data.drl_dataset import fetch_ohlcv_frame
    from models.drl.dsr_gate import evaluate_zero_gate
    from models.validation import WalkForwardSplitter

    logger.info("Fetching %s y %s — %s a %s (%s)…",
                args.y, args.x, args.start, args.end, args.feed)
    ohlcv_y = fetch_ohlcv_frame(args.y, args.start, args.end, feed=args.feed)
    ohlcv_x = fetch_ohlcv_frame(args.x, args.start, args.end, feed=args.feed)

    prices = (
        ohlcv_y[["close"]].rename(columns={"close": args.y})
        .join(ohlcv_x[["close"]].rename(columns={"close": args.x}), how="inner")
        .dropna()
    )
    logger.info("Barras alineadas: %d (%s → %s)",
                len(prices), prices.index[0].date(), prices.index[-1].date())

    strategy = PairStatArb(
        args.y, args.x,
        PairStatArbConfig(
            entry_z=args.entry_z, exit_z=args.exit_z,
            fee_bps=args.fee_bps, max_half_life=args.max_half_life,
        ),
    )
    splitter = WalkForwardSplitter(
        train_size=args.train_size, test_size=args.test_size,
        expanding=True, embargo=args.embargo,
    )
    r = walk_forward_pair_returns(prices, splitter, strategy)
    result = evaluate_zero_gate(r, n_trials=args.n_trials,
                                dsr_threshold=args.dsr_threshold)

    logger.info(
        "GateResult(vs ZERO): passed=%s dsr=%.4f psr=%.4f sharpe=%.4f "
        "n_oos=%d n_trials=%d",
        result.passed, result.dsr_agent, result.psr_agent,
        result.sharpe_agent, result.n_oos_bars, result.n_trials,
    )
    logger.info("%s", result.reason)
    sys.exit(0 if result.passed else 2)


if __name__ == "__main__":
    main()
