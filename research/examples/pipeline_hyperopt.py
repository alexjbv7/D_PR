"""
Pipeline con Optimización Bayesiana de Hiperparámetros
=======================================================
Extiende pipeline_ml_real_data.py añadiendo una fase de búsqueda bayesiana
antes del walk-forward final.

Flujo:
  Datos EURUSD diario
       |
  Features técnicos (fijos: 16 variables)
       |
  [== HYPEROPT (solo 80% inicial del dataset) ==]
  |  Optuna TPE sampler — 50 trials
  |  Por trial:
  |    - samplea XGBoost params + upper_mult + lower_mult + horizon
  |    - recomputa labels triple-barrier con esos params
  |    - mini walk-forward de 3 folds internos -> PSR
  |  -> mejores params
       |
  Walk-forward final con mejores params (20% final del dataset)
       |
  Métricas OOS: PSR, DSR, Sharpe, coverage, win_rate

Uso:
    cd C:\\Users\\alexj\\OneDrive\\Desktop\\quant_bot
    .venv-1\\Scripts\\activate
    python examples/pipeline_hyperopt.py

Requisito adicional (además de requirements.txt):
    pip install optuna>=3.0.0
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Reusar build_features y helpers del pipeline base
# ─────────────────────────────────────────────────────────────────────
from examples.pipeline_ml_real_data import (
    build_features,
    compute_atr_series,
    triple_barrier_labels,
    print_fold_table,
    print_feature_importance,
)


# =====================================================================
# MAIN
# =====================================================================

def main():
    try:
        import optuna  # noqa: F401
    except ImportError:
        print("\nERROR: Optuna no instalado.")
        print("  pip install optuna>=3.0.0")
        sys.exit(1)

    from data.real_data import fetch_real_data
    from models.hyperopt import BayesianHyperopt, HyperoptConfig
    from models.walk_forward_runner import WalkForwardConfig, WalkForwardRunner
    from instruments import EURUSD

    CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
    SYMBOL = "EURUSD"
    START = "2018-01-01"

    print("\n" + "=" * 65)
    print(" PIPELINE ML + HYPEROPT BAYESIANO — DATOS REALES")
    print(f" {SYMBOL} diario | {START} -> hoy")
    print("=" * 65)

    # ── 1. DATOS ──────────────────────────────────────────────────────
    print("\n[1/6] Descargando datos...")
    try:
        prices_df = fetch_real_data(
            SYMBOL, interval="1d", start=START, cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)

    print(f"  {len(prices_df)} barras  |  "
          f"{prices_df.index[0].date()} -> {prices_df.index[-1].date()}")

    # ── 2. FEATURES (fijos — no dependen de barrier params) ───────────
    print("\n[2/6] Construyendo features...")
    features = build_features(prices_df)
    atr = compute_atr_series(prices_df)
    atr_aligned = atr.reindex(features.index)
    close_aligned = prices_df["close"].reindex(features.index)
    print(f"  {features.shape[1]} features  |  {len(features)} barras")

    # ── 3. label_fn (recomputa labels por trial en el hyperopt) ───────
    print("\n[3/6] Preparando label_fn para hyperopt...")

    def label_fn(upper_mult: float, lower_mult: float, horizon: int) -> pd.Series:
        """Recomputa labels triple-barrier para un conjunto de params."""
        return triple_barrier_labels(
            close_aligned, atr_aligned,
            horizon=horizon,
            upper_mult=upper_mult,
            lower_mult=lower_mult,
        )

    # ── 4. HYPEROPT BAYESIANO ─────────────────────────────────────────
    N_TRIALS = 50        # incrementar para búsqueda más exhaustiva
    N_VAL_FOLDS = 3      # folds internos por trial (velocidad vs calidad)

    print(f"\n[4/6] Optimización bayesiana ({N_TRIALS} trials, {N_VAL_FOLDS} folds internos)...")
    print("  (porcion hyperopt: 80% inicial; test final: 20% reservado)\n")

    ho_cfg = HyperoptConfig(
        # Presupuesto
        n_trials=N_TRIALS,
        timeout=None,
        n_jobs=1,

        # Espacio de búsqueda XGBoost
        n_estimators_range=(100, 500),
        max_depth_range=(3, 7),
        learning_rate_range=(0.01, 0.15),
        subsample_range=(0.60, 1.00),
        colsample_bytree_range=(0.50, 1.00),
        reg_alpha_range=(0.0, 2.0),
        reg_lambda_range=(0.5, 4.0),
        min_child_weight_range=(5, 40),
        gamma_range=(0.0, 0.5),

        # Barrera
        upper_mult_range=(0.5, 2.5),
        lower_mult_range=(0.5, 2.5),
        horizon_range=(3, 15),
        symmetric_barriers=False,

        # Protocolo de validación
        val_frac=0.80,
        n_val_folds=N_VAL_FOLDS,
        train_size=252,
        val_size=63,
        embargo=5,
        calib_frac=0.20,
        use_class_weights=True,

        # Objetivo
        objective_metric="psr",
        min_trades=20,

        # Sampler
        sampler="tpe",
        seed=42,
        use_pruner=True,
        n_warmup_steps=5,

        verbose=True,
    )

    ho = BayesianHyperopt(ho_cfg)
    ho_result = ho.run(
        X=features,
        close=close_aligned,
        atr=atr_aligned,
        label_fn=label_fn,
        prices=close_aligned,
        all_classes=[-1, 0, 1],
    )

    print("\n" + ho_result.summary())

    # ── 5. RECOMPUTAR LABELS CON MEJORES PARAMS ───────────────────────
    best_upper = ho_result.best_barrier_params.get("upper_mult", 1.0)
    best_lower = ho_result.best_barrier_params.get("lower_mult", 1.0)
    best_horizon = ho_result.best_barrier_params.get("horizon", 5)

    print(f"\n[5/6] Walk-forward final con mejores params de barrera:")
    print(f"  upper_mult={best_upper:.3f}  lower_mult={best_lower:.3f}  horizon={best_horizon}")

    raw_labels_best = label_fn(best_upper, best_lower, best_horizon)
    valid_idx = raw_labels_best.dropna().index
    X_final = features.loc[valid_idx]
    y_final = raw_labels_best.loc[valid_idx].astype(int)
    prices_final = close_aligned.reindex(valid_idx)
    atr_final = atr_aligned.reindex(valid_idx)

    dist = y_final.value_counts().sort_index()
    print(f"  {len(X_final)} muestras válidas")
    print(f"  Distribución: -1={dist.get(-1,0)} ({dist.get(-1,0)/len(y_final):.1%})  "
          f"0={dist.get(0,0)} ({dist.get(0,0)/len(y_final):.1%})  "
          f"+1={dist.get(1,0)} ({dist.get(1,0)/len(y_final):.1%})")

    # Usar solo el 20% final (test final reservado)
    n_total = len(X_final)
    test_start = int(n_total * ho_cfg.val_frac)
    X_test_final = X_final.iloc[test_start:]
    y_test_final = y_final.iloc[test_start:]
    prices_test_final = prices_final.iloc[test_start:]
    atr_test_final = atr_final.iloc[test_start:]

    if len(X_test_final) < ho_cfg.train_size + ho_cfg.val_size:
        print("  ADVERTENCIA: test final demasiado pequeno para un fold completo.")
        print("  Corriendo walk-forward sobre el dataset completo como fallback.")
        X_test_final = X_final
        y_test_final = y_final
        prices_test_final = prices_final
        atr_test_final = atr_final

    print(f"\n  Evaluando sobre {len(X_test_final)} barras del test final...")

    # ── 6. WALK-FORWARD FINAL ─────────────────────────────────────────
    print("\n[6/6] Walk-forward final...")
    best_cfg = ho_result.to_walk_forward_config()

    # Construir config final del runner
    final_cfg = WalkForwardConfig(
        train_size=252,
        test_size=63,
        embargo=5,
        expanding=False,
        calib_frac=0.20,
        calib_method="sigmoid",
        filter_symmetric=True,
        filter_min_coverage=0.05,
        filter_n_thresholds=40,
        kelly_fraction=0.25,
        max_risk_pct=0.02,
        rr_min=best_cfg.rr_min,
        rr_max=best_cfg.rr_max,
        rr_p_low=0.40,
        rr_p_high=0.70,
        rr_shape="sigmoid",
        atr_sl_mult=2.0,
        use_class_weights=True,
        track_importance=True,
        shap_sample_size=100,
        instrument=EURUSD,
        xgb_params=best_cfg.xgb_params,
    )

    runner = WalkForwardRunner(final_cfg)
    result = runner.run(
        X=X_test_final,
        y=y_test_final,
        prices=prices_test_final,
        atr=atr_test_final,
        all_classes=[-1, 0, 1],
    )

    print("\n" + result.summary())

    print("\n--- DETALLE POR FOLD ---")
    print_fold_table(result)

    print("\n--- FEATURE IMPORTANCE (cross-fold, top 10) ---")
    print_feature_importance(result, top_n=10)

    # Sizing sample
    active = result.oos_sizing[result.oos_sizing["signal"] != 0].head(5)
    if not active.empty:
        print("\n--- MUESTRA DE SIZING (primeras 5 entradas activas) ---")
        cols = ["signal", "p_win", "rr_dynamic", "kelly_raw", "risk_pct"]
        print(active[cols].to_string())

    # Veredicto final
    print("\n" + "=" * 65)
    psr = result.global_metrics.get("psr")
    dsr = result.global_metrics.get("dsr")
    sharpe = result.global_metrics.get("sharpe")
    n_trades = result.global_metrics.get("n_trades", 0)

    if psr is not None and dsr is not None and sharpe is not None:
        if psr >= 0.95 and dsr >= 0.90 and sharpe > 0:
            verdict = "PASS -- senal estadisticamente significativa"
        elif psr >= 0.80 or sharpe > 0:
            verdict = "MARGINAL -- mejoras necesarias antes de operar"
        else:
            verdict = "FAIL -- sin senal estadistica suficiente"
    else:
        verdict = "INSUFICIENTE -- pocos trades para evaluar"

    print(f" VEREDICTO: {verdict}")
    print(f" PSR={psr}  DSR={dsr}  Sharpe={sharpe}  Trades={n_trades}")
    print(f" Mejores params barrera: {ho_result.best_barrier_params}")
    print("=" * 65)
    print()
    print(" Proximos pasos:")
    print("   -> Si FAIL: ampliar espacio de búsqueda o añadir más features")
    print("   -> Si MARGINAL: incrementar n_trials (100+) y search_rr=True")
    print("   -> Si PASS: paper trading minimo 1 mes antes de capital real")
    print("   -> Siempre: verificar que PSR se mantiene en datos OOS reales")
    print("=" * 65)


if __name__ == "__main__":
    main()
