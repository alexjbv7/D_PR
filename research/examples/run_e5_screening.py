"""
run_e5_screening.py — Screening de pares cointegrados sobre la caché 4h (Rama B, E5).

Carga los parquet 4h del universo, corre ``screen_universe`` (anchor XRP/USD),
imprime el ranking por LB95 y el veredicto vs ZERO, y guarda un JSON.

Prerrequisitos: haber bajado los 4h con ``fetch_4h_crypto.py``; ``statsmodels``.
Uso:
  python research/examples/run_e5_screening.py
  python research/examples/run_e5_screening.py --symbols XRP/USD BTC/USD ETH/USD --anchor XRP/USD
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_RESEARCH = Path(__file__).resolve().parents[1]
for _p in (str(_RESEARCH), str(_RESEARCH.parent / "shared")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_DEFAULT = ["XRP/USD", "BTC/USD", "ETH/USD", "SOL/USD",
            "LINK/USD", "DOGE/USD", "AVAX/USD", "LTC/USD"]


def _safe(sym: str) -> str:
    return sym.replace("/", "_")


def main() -> int:
    ap = argparse.ArgumentParser(description="E5 — screening de pares cointegrados (4h)")
    ap.add_argument("--symbols", nargs="+", default=_DEFAULT)
    ap.add_argument("--anchor", default="XRP/USD")
    ap.add_argument("--timeframe", default="4h")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--out", default="e5_pair_screening.json")
    args = ap.parse_args()

    from alpha.statarb.screen import evaluate_screen, rank_pairs, screen_universe

    base = _RESEARCH / "data" / "alpaca_bars" / "bars" / args.timeframe
    prices, missing = {}, []
    for s in args.symbols:
        p = base / f"{_safe(s)}.parquet"
        if p.exists():
            prices[s] = pd.read_parquet(p).sort_index()
        else:
            missing.append(s)
    if missing:
        print(f"FALTAN parquet 4h: {missing}\nCorre primero fetch_4h_crypto.py.",
              file=sys.stderr)
    syms = [s for s in args.symbols if s in prices]
    if args.anchor not in syms or len(syms) < 2:
        print("ERROR: faltan datos (necesitas el anchor + >=1 par).", file=sys.stderr)
        return 2

    results = screen_universe(prices, syms, anchor=args.anchor, n_folds=args.n_folds)
    if not results:
        print("Sin resultados (todos los pares saltados).", file=sys.stderr)
        return 1
    verdict = evaluate_screen(results)

    pack = {
        "anchor": args.anchor, "timeframe": args.timeframe, "n_pairs": len(results),
        "verdict": {"branch": verdict.branch, "best": verdict.best.label,
                    "best_lb95": round(verdict.best_lb95, 4),
                    "dsr_deflated": round(verdict.dsr_deflated, 4),
                    "gate_passed": verdict.gate_passed, "reason": verdict.reason},
        "pairs": [
            {"pair": r.label, "sharpe": round(r.sharpe, 4), "lb95": round(r.lb95, 4),
             "p95": round(r.p95, 4), "frac_traded": round(r.frac_traded, 4),
             "n_oos": r.n_oos}
            for r in rank_pairs(results)
        ],
    }
    Path(args.out).write_text(json.dumps(pack, indent=2, ensure_ascii=False),
                              encoding="utf-8")

    print(f"\n=== E5 screening · anchor {args.anchor} · {len(results)} pares ===")
    print(f"{'par':18s} {'Sharpe':>7s} {'LB95':>7s} {'operado':>8s}")
    for r in rank_pairs(results):
        print(f"{r.label:18s} {r.sharpe:7.2f} {r.lb95:7.2f} {r.frac_traded*100:7.0f}%")
    print(f"\nVEREDICTO → {verdict.branch} | mejor={verdict.best.label} "
          f"LB95={verdict.best_lb95:.2f} DSR={verdict.dsr_deflated:.2f}")
    print(f"  {verdict.reason}")
    print(f"Pack → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
