"""
Pipeline ML Completo con Datos Reales
======================================
Primer pipeline end-to-end con el stack completo de prioridad alta:

  Yahoo Finance (EURUSD diario)
       ↓
  Feature engineering (técnico: ATR, momentum, vol, RSI, MACD, z-score)
       ↓
  Triple-barrier labels {-1, 0, +1}
       ↓
  WalkForwardRunner:
    por fold -> fit XGBoost -> calibrar -> EntryFilter -> Kelly + DynamicRR
       ↓
  Métricas OOS: PSR, DSR, Sharpe, coverage, win_rate
  Feature importance cross-fold
  Calibration report por fold

Uso:
    cd C:\\Users\\alexj\\OneDrive\\Desktop\\quant_bot
    .venv-1\\Scripts\\activate
    python examples/pipeline_ml_real_data.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from features.labeling import triple_barrier_labels_atr, compute_atr_ewm

logging.basicConfig(
    level=logging.WARNING,          # silenciar INFO durante el run
    format="%(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# =====================================================================
# FEATURE ENGINEERING — self-contained, sin dependencias externas
# =====================================================================

def build_features(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """
    Construye features técnicos desde OHLCV diario.

    Features:
      - ret_1..ret_20        : log-retornos a distintos horizontes
      - vol_10, vol_20       : volatilidad rolling (std de ret_1)
      - vol_ratio            : vol_5 / vol_20  (régimen de volatilidad)
      - rsi_14               : RSI clásico
      - macd_signal          : diferencia entre EMA12 y EMA26 normalizada
      - close_vs_ma20/50     : precio relativo a MA (tendencia)
      - atr_norm             : ATR/precio (volatilidad relativa)
      - range_norm           : (high-low)/close (rango intradiario)
      - ret_z10, ret_z20     : z-score de ret_1 sobre ventanas rolling

    Returns pd.DataFrame sin NaN (filas iniciales eliminadas).
    """
    c = prices["close"].copy()
    h = prices["high"].copy()
    lo = prices["low"].copy()

    feat = pd.DataFrame(index=prices.index)

    # ── Retornos log ──────────────────────────────────────────────────
    ret = np.log(c / c.shift(1))
    for w in [1, 5, 10, 20]:
        feat[f"ret_{w}"] = np.log(c / c.shift(w))

    # ── Volatilidad rolling ───────────────────────────────────────────
    for w in [5, 10, 20]:
        feat[f"vol_{w}"] = ret.rolling(w).std()
    feat["vol_ratio"] = feat["vol_5"] / (feat["vol_20"] + 1e-10)

    # ── RSI 14 ────────────────────────────────────────────────────────
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat["rsi_14"] = 100 - (100 / (1 + rs))
    feat["rsi_14_norm"] = (feat["rsi_14"] - 50) / 50   # centrado en 0

    # ── MACD (EMA12 - EMA26) normalizado ─────────────────────────────
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    feat["macd_signal"] = macd / (c + 1e-10)            # normalizado por precio

    # ── Precio vs MAs ────────────────────────────────────────────────
    for w in [20, 50]:
        ma = c.rolling(w).mean()
        feat[f"close_vs_ma{w}"] = (c - ma) / (ma + 1e-10)

    # ── ATR normalizado ───────────────────────────────────────────────
    prev_c = c.shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, adjust=False).mean()
    feat["atr_norm"] = atr / (c + 1e-10)

    # ── Rango intradiario ─────────────────────────────────────────────
    feat["range_norm"] = (h - lo) / (c + 1e-10)

    # ── Z-score de retornos ───────────────────────────────────────────
    for w in [10, 20]:
        mu_roll = ret.rolling(w).mean()
        std_roll = ret.rolling(w).std()
        feat[f"ret_z{w}"] = (ret - mu_roll) / (std_roll + 1e-10)

    # Eliminar vol_5 (sustituida por vol_ratio, evitar colinealidad)
    feat.drop(columns=["vol_5"], inplace=True)

    return feat.dropna()


# triple_barrier_labels_atr y compute_atr_ewm importadas desde features.labeling
# (módulo canónico — ver research/features/labeling.py)


# =====================================================================
# REPORTE FINAL
# =====================================================================

def print_fold_table(result) -> None:
    """Imprime tabla por fold: threshold | trades | coverage | ECE | Sharpe."""
    print(f"\n  {'Fold':>4}  {'Thresh':>7}  {'Trades':>6}  {'Cov':>5}  "
          f"{'ECE':>6}  {'Sharpe':>7}  {'PSR':>6}  {'Calib'}")
    print("  " + "-" * 62)
    for fr in result.fold_results:
        sharpe = fr.metrics.get("sharpe")
        psr = fr.metrics.get("psr")
        ece = fr.calibration.get("ece_calibrated")
        verdict = fr.calibration.get("verdict", "?")
        cov = fr.metrics.get("coverage", 0)
        print(
            f"  {fr.fold_idx+1:>4}  "
            f"{fr.threshold_long:.3f}  "
            f"{fr.metrics.get('n_trades', 0):>6}  "
            f"{cov:>5.1%}  "
            f"{ece if ece is not None else 'N/A':>6}  "
            f"{sharpe if sharpe is not None else 'N/A':>7}  "
            f"{psr if psr is not None else 'N/A':>6}  "
            f"{verdict}"
        )


def print_feature_importance(result, top_n: int = 10) -> None:
    """Imprime las top N features por importancia mediana cross-fold."""
    if result.feature_importance_agg.empty:
        print("  (sin datos de importancia)")
        return
    df = result.feature_importance_agg.head(top_n)
    print(f"\n  {'Feature':30s} {'Median':>8} {'Std':>7} {'NonZero%':>9}")
    print("  " + "-" * 58)
    for feat, row in df.iterrows():
        flag = " <--" if feat in result.features_to_drop else ""
        print(
            f"  {str(feat):30s} "
            f"{row['median_importance']:8.4f} "
            f"{row['std_importance']:7.4f} "
            f"{row['frac_folds_nonzero']:9.0%}"
            f"{flag}"
        )
    if result.features_to_drop:
        print(f"\n  Candidatas a eliminar: {result.features_to_drop}")


# =====================================================================
# MAIN
# =====================================================================

def main():
    from data.real_data import fetch_real_data
    from models.walk_forward_runner import WalkForwardConfig, WalkForwardRunner
    from instruments import EURUSD

    CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
    SYMBOL = "EURUSD"
    START = "2018-01-01"
    HORIZON = 5          # días de horizonte para el label
    UPPER_MULT = 1.0     # barreras ATR — reducido de 1.5 para menos neutrales
    LOWER_MULT = 1.0

    print("\n" + "=" * 65)
    print(" PIPELINE ML COMPLETO — DATOS REALES")
    print(f" {SYMBOL} diario | {START} -> hoy")
    print("=" * 65)

    # ── 1. DATOS ──────────────────────────────────────────────────────
    print("\n[1/5] Descargando datos...")
    try:
        prices_df = fetch_real_data(
            SYMBOL, interval="1d", start=START, cache_dir=CACHE_DIR
        )
    except Exception as e:
        print(f"  ERROR descargando datos: {e}")
        print("  Asegurate de tener conexion a internet y yfinance instalado.")
        sys.exit(1)

    print(f"  {len(prices_df)} barras  |  "
          f"{prices_df.index[0].date()} -> {prices_df.index[-1].date()}")

    # ── 2. FEATURES ───────────────────────────────────────────────────
    print("\n[2/5] Construyendo features...")
    features = build_features(prices_df, horizon=HORIZON)
    atr = compute_atr_ewm(prices_df)
    print(f"  {features.shape[1]} features  |  {len(features)} barras tras dropna")
    print(f"  Features: {list(features.columns)}")

    # ── 3. LABELS (triple-barrier) ────────────────────────────────────
    print(f"\n[3/5] Calculando labels triple-barrier "
          f"(horizon={HORIZON}, mult=±{UPPER_MULT}×ATR)...")

    atr_aligned = atr.reindex(features.index)
    close_aligned = prices_df["close"].reindex(features.index)
    raw_labels = triple_barrier_labels_atr(
        close_aligned, atr_aligned,
        horizon=HORIZON, upper_mult=UPPER_MULT, lower_mult=LOWER_MULT
    )

    # Alinear features + labels (eliminar NaN de los últimos `horizon` barras)
    valid_idx = raw_labels.dropna().index
    X = features.loc[valid_idx]
    y = raw_labels.loc[valid_idx].astype(int)
    prices_aligned = prices_df["close"].reindex(valid_idx)
    atr_final = atr_aligned.reindex(valid_idx)

    dist = y.value_counts().sort_index()
    print(f"  {len(X)} muestras válidas")
    print(f"  Distribución: -1={dist.get(-1, 0)} ({dist.get(-1,0)/len(y):.1%})  "
          f"0={dist.get(0, 0)} ({dist.get(0,0)/len(y):.1%})  "
          f"+1={dist.get(1, 0)} ({dist.get(1,0)/len(y):.1%})")

    # ── 4. WALK-FORWARD RUNNER ────────────────────────────────────────
    print("\n[4/5] Ejecutando WalkForwardRunner...")
    print("  (calibracion + entry filter + Kelly + dynamic R:R por fold)\n")

    cfg = WalkForwardConfig(
        train_size=252,
        test_size=63,
        embargo=5,
        expanding=False,

        # Calibracion
        calib_frac=0.20,
        calib_method="sigmoid",

        # Entry filter
        filter_symmetric=True,
        filter_min_coverage=0.05,
        filter_n_thresholds=40,

        # Kelly + R:R
        kelly_fraction=0.25,
        max_risk_pct=0.02,
        rr_min=1.2,
        rr_max=2.5,
        rr_p_low=0.40,
        rr_p_high=0.70,
        rr_shape="sigmoid",
        atr_sl_mult=2.0,

        # Class weights (compensar 68% neutros)
        use_class_weights=True,

        # Regime detection — GMM con 3 componentes (quiet/trending/volatile)
        use_regime_features=True,
        regime_n_components=3,

        # PCA Denoising — retener 95% de varianza, excluir columnas regime_
        use_pca=True,
        pca_n_components=0.95,

        # Meta-labeling — segundo modelo binario que filtra señales del primario
        use_meta_labeling=True,
        meta_min_samples=20,

        # Bayesian P(win) — combina prior de régimen con likelihood del modelo
        use_bayesian_sizing=True,
        bayesian_combination="product",
        bayesian_min_samples=15,

        # Features importance
        track_importance=True,
        shap_sample_size=100,

        # Instrumento (sizing real en lotes FX)
        instrument=EURUSD,

        # XGBoost (conservador, anti-overfitting)
        xgb_params={
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.1,
            "reg_lambda": 1.5,
            "min_child_weight": 10,
            "gamma": 0.1,
        },
    )

    runner = WalkForwardRunner(cfg)
    result = runner.run(
        X=X, y=y,
        prices=prices_aligned,
        atr=atr_final,
        all_classes=[-1, 0, 1],
    )

    # ── 5. RESULTADOS ─────────────────────────────────────────────────
    print("\n[5/5] Resultados\n")
    print(result.summary())

    # Tabla por fold
    print("\n--- DETALLE POR FOLD ---")
    print_fold_table(result)

    # Feature importance
    print("\n--- FEATURE IMPORTANCE (cross-fold, top 10) ---")
    print_feature_importance(result, top_n=10)

    # Sizing sample (primeras 5 barras con señal activa)
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
            verdict = "PASS — señal estadisticamente significativa"
        elif psr >= 0.80 or sharpe > 0:
            verdict = "MARGINAL — mejoras necesarias antes de operar"
        else:
            verdict = "FAIL — sin señal estadistica suficiente"
    else:
        verdict = "INSUFICIENTE — pocos trades para evaluar"

    print(f" VEREDICTO: {verdict}")
    print(f" PSR={psr}  DSR={dsr}  Sharpe={sharpe}  Trades={n_trades}")
    print("=" * 65)
    print()
    print(" Proximos pasos:")
    print("   -> Si FAIL: revisar feature importance, ajustar horizon/mult de barriers")
    print("   -> Si MARGINAL: optimizar hyperparams con Bayesian search")
    print("   -> Si PASS: correr sobre ES + otros instrumentos")
    print("   -> Siempre: paper trading minimo 1 mes antes de capital real")
    print("=" * 65)


if __name__ == "__main__":
    main()
