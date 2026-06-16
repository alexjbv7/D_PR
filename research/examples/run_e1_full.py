"""
run_e1_full.py — Orquesta el sprint de evidencia (E1 + E4 + E6, E2 opcional)
sobre datos reales y aplica el gate de decisión PRE-REGISTRADO (arbitraje D).

Corre las cuatro patas de E1 en el MISMO walk-forward:
  DQN (N seeds) · XGBoost (N seeds) · momentum · mean-reversion
…reporta Sharpe con IC por modelo, evalúa E4 (potencia + deflación honesta del DSR)
y E6 (sensibilidad al benchmark + Reality Check/SPA de data-snooping), y emite el
veredicto Rama A/A'/B SOLO si el acta está firmada (``--acta-signed``).

Prerrequisitos (venv del repo):
  - torch, xgboost, gymnasium, scikit-learn, pyarrow
  - PYTHONPATH con ``research`` y ``shared``
  - datos OHLCV en la caché (``research/data/alpaca_bars/bars/<tf>/<SYMBOL>.parquet``)

Uso:
  python research/examples/run_e1_full.py --symbol SPY --n-seeds 20 --episodes 100 \
      --n-jobs 4 --materiality 0.20 --acta-signed --out e1_evidence_pack.json

Notas:
  - El gate usa posiciones→retornos (no ``p_win``); la calibración del ``p_win``
    (E3 paso 2) es para el path de serving y se diagnostica aparte
    (``alpha.agents.dqn_calibration``).
  - Degradación: ``--no-dqn`` / ``--no-xgb`` para correr sin esas patas (p.ej. en un
    entorno sin torch); las reglas siempre corren.
  - ``--n-trials`` deflacta el DSR honestamente (nº de configs/seeds buscados).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Bootstrap de sys.path: añade research/ y shared/ para poder ejecutar el script
# directamente (`python research/examples/run_e1_full.py`) sin fijar PYTHONPATH.
_RESEARCH = Path(__file__).resolve().parents[1]
_SHARED = _RESEARCH.parent / "shared"
for _p in (str(_RESEARCH), str(_SHARED)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("run_e1_full")

_DEFAULT_CACHE = Path(__file__).resolve().parents[1] / "data" / "alpaca_bars" / "bars"


def load_ohlcv(symbol: str, timeframe: str = "1d", cache: Path = _DEFAULT_CACHE) -> pd.DataFrame:
    path = cache / timeframe / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe la caché {path}. Símbolos disponibles: "
                                f"{[p.stem for p in (cache / timeframe).glob('*.parquet')]}")
    df = pd.read_parquet(path).sort_index()
    log.info("%s: %d barras (%s → %s)", symbol, len(df), df.index.min(), df.index.max())
    return df


def _mk_result(name, sharpes, n_oos):
    from models.drl.e1_baseline_comparison import ModelSharpeResult, sharpe_ci_from_seeds
    mean, lb, p95 = sharpe_ci_from_seeds(sharpes)
    return ModelSharpeResult(
        name=name, sharpe_by_seed=np.asarray(sharpes, float), mean=mean, lb95=lb,
        p95=p95, n_seeds=len(sharpes), n_oos_bars=int(n_oos), ci_method="seeds",
    )


def run(args) -> dict:
    from models.drl.dsr_gate import (
        AgentSpec, EnvironmentConfig, buyhold_oos_returns, make_wf_splitter,
        walk_forward_oos_returns, xgb_oos_returns,
    )
    from models.drl.e1_baseline_comparison import (
        ModelSharpeResult, _ann_sharpe, block_bootstrap_sharpe_ci,
        evaluate_e1_decision, rule_oos_returns, sharpe_ci_from_seeds,
    )
    from models.drl.e4_power_diagnostics import (
        deflation_report, fold_lengths_from_splitter, sharpe_power_diagnostic,
    )
    from models.drl.e6_benchmark_spa import (
        benchmark_sensitivity, reality_check_pvalue, studentized_reality_check_pvalue,
    )

    raw = load_ohlcv(args.symbol, args.timeframe)
    cfg = EnvironmentConfig()
    splitter = make_wf_splitter(raw, args.n_folds, env_cfg=cfg)
    seeds = list(range(args.n_seeds))

    results: dict[str, ModelSharpeResult] = {}
    series0: dict[str, np.ndarray] = {}     # serie OOS (seed 0 / determinista) para E4/E6

    # --- DQN ---
    if not args.no_dqn:
        sh = []
        for s in seeds:
            r = walk_forward_oos_returns(
                AgentSpec(algo="dqn", episodes=args.episodes, seed=s),
                raw, splitter, cfg, seed=s, n_jobs=args.n_jobs,
            )
            sh.append(_ann_sharpe(r))
            if s == seeds[0]:
                series0["dqn"] = r
            log.info("DQN seed=%d sharpe=%.3f", s, sh[-1])
        results["dqn"] = _mk_result("dqn", sh, len(series0["dqn"]))

    # --- XGBoost ---
    if not args.no_xgb:
        sh = []
        for s in seeds:
            r = xgb_oos_returns(raw, splitter, seed=s)
            sh.append(_ann_sharpe(r))
            if s == seeds[0]:
                series0["xgboost"] = r
        results["xgboost"] = _mk_result("xgboost", sh, len(series0["xgboost"]))

    # --- reglas (deterministas → IC por block bootstrap) ---
    for rule in ("momentum", "mean_rev"):
        r = rule_oos_returns(raw, splitter, rule)
        mean, lb, p95, _ = block_bootstrap_sharpe_ci(r)
        results[rule] = ModelSharpeResult(
            name=rule, sharpe_by_seed=np.asarray([mean]), mean=mean, lb95=lb,
            p95=p95, n_seeds=1, n_oos_bars=len(r), ci_method="block_bootstrap",
        )
        series0[rule] = r

    # --- veredicto del gate pre-registrado ---
    verdict = evaluate_e1_decision(results, materiality=args.materiality)
    best = results[verdict.best_model]
    best_ret = series0[verdict.best_model]

    # --- E4: potencia + deflación sobre la mejor serie ---
    sizes = fold_lengths_from_splitter(raw, splitter)
    power = sharpe_power_diagnostic(best.mean, sum(sizes))
    deflation = deflation_report(best_ret, n_trials=args.n_trials or args.n_seeds)

    # --- E6: sensibilidad de benchmark + data-snooping sobre las configs ---
    minlen = min(len(v) for v in series0.values())
    bh = buyhold_oos_returns(raw, splitter)[:minlen]
    benches = {"zero": np.zeros(minlen), "buy_and_hold": bh}
    sens = benchmark_sensitivity(best_ret[:minlen], benches)
    configs = np.column_stack([series0[k][:minlen] for k in series0])  # vs zero == ellas mismas
    rc_p = reality_check_pvalue(configs, n_boot=args.n_boot)
    spa = studentized_reality_check_pvalue(configs, n_boot=args.n_boot)

    # --- E2 opcional ---
    e2 = None
    if args.run_e2 and not args.no_dqn:
        from models.drl.e2_lambda_sweep import lambda_grid, run_lambda_sweep
        grid = lambda_grid([1.0, 2.0, 4.0], [0.25, 0.5, 1.0], [0.0, 0.001, 0.01])
        sweep = run_lambda_sweep(raw, grid, n_folds=args.n_folds,
                                 seeds=range(min(3, args.n_seeds)), episodes=args.episodes)
        e2 = {"reason": sweep.reason, "all_flat_collapsed": sweep.all_flat_collapsed,
              "best": (None if sweep.best is None else sweep.best.point.as_kwargs())}

    pack = {
        "symbol": args.symbol, "n_folds": args.n_folds, "n_seeds": args.n_seeds,
        "fold_oos_sizes": list(sizes), "total_oos_bars": int(sum(sizes)),
        "acta_signed": bool(args.acta_signed),
        "models": {
            n: {"sharpe_mean": round(r.mean, 4), "lb95": round(r.lb95, 4),
                "p95": round(r.p95, 4), "ci_method": r.ci_method, "n_seeds": r.n_seeds}
            for n, r in results.items()
        },
        "gate_verdict": {
            "best_model": verdict.best_model, "best_lb95": round(verdict.best_lb95, 4),
            "branch": verdict.branch, "directional_falsified": verdict.directional_falsified,
            "materiality": verdict.materiality, "reason": verdict.reason,
            "OFICIAL": bool(args.acta_signed),
        },
        "E4_power": {"verdict": power.verdict, "se_ann": round(power.se_ann, 4),
                     "ci95": [round(power.ci95_low, 3), round(power.ci95_high, 3)],
                     "reason": power.reason, "deflation": deflation},
        "E6": {"rc_pvalue": round(rc_p, 4), "spa_pvalue": round(spa["p_value"], 4),
               "benchmark_sensitivity": {k: {"sharpe_diff": round(v.sharpe_diff, 4),
                                             "model_beats": v.model_beats}
                                         for k, v in sens.items()}},
        "E2": e2,
    }
    return pack


def main():
    ap = argparse.ArgumentParser(description="Sprint de evidencia E1+E4+E6 (arbitraje D)")
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-trials", type=int, default=0, help="nº trials Optuna para deflación (0 → usa n_seeds)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--materiality", type=float, default=0.20)
    ap.add_argument("--run-e2", action="store_true")
    ap.add_argument("--no-dqn", action="store_true")
    ap.add_argument("--no-xgb", action="store_true")
    ap.add_argument("--acta-signed", action="store_true",
                    help="Confirma que ACTA_GATE_DECISION_FASE0.md está firmada. Sin esto, "
                         "el veredicto se imprime como NO OFICIAL.")
    ap.add_argument("--out", default="e1_evidence_pack.json")
    args = ap.parse_args()

    if not args.acta_signed:
        log.warning("ACTA NO FIRMADA: se ejecuta, pero el veredicto NO es oficial "
                    "(pre-registro incompleto). Firma el acta antes de decidir.")

    pack = run(args)
    # encoding utf-8 explícito: el pack contiene Δ, γ, → (Windows usa cp1252 por defecto).
    Path(args.out).write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")

    v = pack["gate_verdict"]
    print("\n================ EVIDENCE PACK (E1) ================")
    print(f"Símbolo {pack['symbol']} | folds {pack['n_folds']} | seeds {pack['n_seeds']} "
          f"| OOS/fold {pack['fold_oos_sizes']} (total {pack['total_oos_bars']})")
    print(f"{'modelo':10s} {'Sharpe':>7s} {'LB95':>7s} {'P95':>7s}  método")
    for n, m in pack["models"].items():
        print(f"{n:10s} {m['sharpe_mean']:7.3f} {m['lb95']:7.3f} {m['p95']:7.3f}  {m['ci_method']}")
    tag = "OFICIAL" if v["OFICIAL"] else "NO OFICIAL (acta sin firmar)"
    print(f"\nGATE [{tag}] → Rama {v['branch']} | mejor={v['best_model']} "
          f"LB95(S_Δ)={v['best_lb95']}")
    print(f"  {v['reason']}")
    print(f"E4 potencia: {pack['E4_power']['verdict']} | IC95 {pack['E4_power']['ci95']}")
    print(f"E6 data-snooping: RC p={pack['E6']['rc_pvalue']} SPA p={pack['E6']['spa_pvalue']}")
    print(f"Evidence pack → {args.out}")
    print("====================================================")


if __name__ == "__main__":
    main()
