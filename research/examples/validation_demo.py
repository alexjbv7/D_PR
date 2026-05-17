"""
Validation Framework Demo (end-to-end)
======================================
Demuestra el pipeline completo de validación rigurosa SIN necesidad de datos
reales ni broker:

  1. Genera datos sintéticos OHLCV (GBM con régimen y vol cluster).
  2. Ejecuta una estrategia de momentum trivial.
  3. Valida con walk-forward.
  4. Calcula métricas básicas + avanzadas (DSR, PSR, bootstrap CI).
  5. Compara IS vs OOS para detectar overfitting.
  6. Aplica la función objetivo formal y emite veredicto.

Uso:
    cd quant_bot
    python examples/validation_demo.py

Este demo ES la prueba de que tu framework funciona. Si esto pasa, sabes que
cualquier estrategia futura puede pasarse por el mismo pipeline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure we can import from the project root (this file is in examples/)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.validation import WalkForwardSplitter
from backtesting.engine import Backtester, BacktestConfig
from metrics.objective import ObjectiveSpec, HardConstraints
from reporting.report import (
    build_report,
    compare_is_oos,
    aggregate_folds,
    FoldReport,
    print_report,
)


# =====================================================================
# 1. DATOS SINTÉTICOS REALISTAS
# =====================================================================

def generate_synthetic_ohlcv(
    n_bars: int = 5000,
    base_price: float = 100.0,
    annual_drift: float = 0.05,
    annual_vol: float = 0.20,
    bars_per_year: int = 252 * 24,  # horario
    vol_cluster: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Genera OHLCV sintético con:
    - Geometric Brownian Motion como base
    - Volatility clustering opcional (GARCH-like)
    - High/Low simulados con un rango proporcional a la vol intra-bar

    Esto NO es un mercado real, pero es suficiente para validar la mecánica del
    pipeline sin riesgo de filtrar resultados de un backtest sobre datos vivos.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year

    # Vol con clustering simple (AR(1) en log-vol)
    if vol_cluster:
        log_vol = np.zeros(n_bars)
        log_vol[0] = np.log(annual_vol)
        for t in range(1, n_bars):
            log_vol[t] = 0.95 * log_vol[t - 1] + 0.05 * np.log(annual_vol) \
                + rng.normal(0, 0.05)
        sigma_t = np.exp(log_vol)
    else:
        sigma_t = np.full(n_bars, annual_vol)

    # Retornos log
    z = rng.standard_normal(n_bars)
    log_returns = (annual_drift - 0.5 * sigma_t ** 2) * dt + sigma_t * np.sqrt(dt) * z

    close = base_price * np.exp(np.cumsum(log_returns))
    # Open[t] ≈ Close[t-1] (gap pequeño)
    open_ = np.concatenate([[base_price], close[:-1]])
    # Range intra-bar proporcional a vol per-period
    intra_range = sigma_t * np.sqrt(dt) * close * 0.7
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1, n_bars)) * intra_range / 2
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1, n_bars)) * intra_range / 2
    volume = rng.lognormal(mean=8, sigma=0.5, size=n_bars)

    idx = pd.date_range("2022-01-01", periods=n_bars, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# =====================================================================
# 2. ESTRATEGIA TRIVIAL (sólo para validar el pipeline)
# =====================================================================

def momentum_signal(prices: pd.DataFrame, fast: int = 12, slow: int = 48,
                    threshold: float = 0.0) -> pd.Series:
    """
    Señal MOM trivial: cruce de medias móviles.

    Si MA_fast > MA_slow * (1 + threshold) → +1 (long)
    Si MA_fast < MA_slow * (1 - threshold) → -1 (short)
    Caso contrario → 0

    NOTA: deliberadamente simple. Esto es una HERRAMIENTA DE VALIDACIÓN del
    framework, no una estrategia. NO operes esto.
    """
    close = prices["close"]
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    spread = (ma_fast - ma_slow) / ma_slow
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[spread > threshold] = 1
    sig[spread < -threshold] = -1
    return sig


# =====================================================================
# 3. PIPELINE COMPLETO
# =====================================================================

def run_demo():
    print("\n" + "█" * 75)
    print(" DEMO: VALIDATION FRAMEWORK END-TO-END")
    print("█" * 75)

    # -----------------------------------------------------------------
    # Datos
    # -----------------------------------------------------------------
    prices = generate_synthetic_ohlcv(n_bars=5000, seed=7)
    print(f"\n[DATA] Generadas {len(prices)} barras horarias sintéticas.")
    print(f"       Período: {prices.index[0]} → {prices.index[-1]}")

    bars_per_year = 252 * 24  # asumimos 252 días hábiles, 24h crypto-like

    # -----------------------------------------------------------------
    # Walk-forward splits
    # -----------------------------------------------------------------
    splitter = WalkForwardSplitter(
        train_size=24 * 90,      # 90 días de "entrenamiento" (aquí es tuning del threshold)
        test_size=24 * 30,       # 30 días de test
        embargo=24,              # 1 día de embargo
        expanding=False,
    )
    n_splits = splitter.get_n_splits(prices)
    print(f"\n[VALIDATION] Walk-forward: {n_splits} folds")

    # -----------------------------------------------------------------
    # Backtest config
    # -----------------------------------------------------------------
    bt_config = BacktestConfig(
        initial_capital=10_000.0,
        fee_bps=5.0,             # 0.05%
        slippage_bps=2.0,
        slippage_vol_mult=0.3,
        max_position_pct=1.0,
        allow_short=True,        # demo permite short
        risk_free_rate=0.04,
    )

    # -----------------------------------------------------------------
    # Loop por fold
    # -----------------------------------------------------------------
    fold_reports: list[FoldReport] = []
    oos_returns_concat: list[pd.Series] = []
    is_returns_concat: list[pd.Series] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(prices)):
        train_slice = prices.iloc[train_idx]
        test_slice = prices.iloc[test_idx]

        # En este demo no hay tuning real (threshold fijo). En producción aquí
        # entrenarías un modelo o optimizarías hiperparámetros con train_slice.
        sig_train = momentum_signal(train_slice, fast=12, slow=48)
        sig_test = momentum_signal(
            prices.iloc[max(0, test_idx[0] - 48):test_idx[-1] + 1],
            fast=12, slow=48,
        ).loc[test_slice.index]  # warm-up para evitar NaN

        # Backtest IS y OOS
        bt = Backtester(bt_config)
        res_train = bt.run(train_slice, sig_train)
        res_test = bt.run(test_slice, sig_test)

        is_returns_concat.append(res_train.returns)
        oos_returns_concat.append(res_test.returns)

        # Sharpe anualizado del fold
        r_test = res_test.returns
        sigma = r_test.std(ddof=1)
        sharpe_test = (
            np.sqrt(bars_per_year) * r_test.mean() / sigma if sigma > 0 else 0.0
        )

        # DSR rápido del fold (n_trials=1 — fold individual)
        from metrics.advanced import (
            probabilistic_sharpe_ratio,
            returns_skewness,
            returns_kurtosis,
        )
        sr_period = r_test.mean() / sigma if sigma > 0 else 0.0
        dsr_test = probabilistic_sharpe_ratio(
            sr_period,
            len(r_test),
            returns_skewness(r_test),
            returns_kurtosis(r_test, excess=False),
            sr_benchmark=0.0,
        )

        fold_reports.append(FoldReport(
            fold_idx=fold_idx,
            train_start=str(train_slice.index[0]),
            train_end=str(train_slice.index[-1]),
            test_start=str(test_slice.index[0]),
            test_end=str(test_slice.index[-1]),
            n_test_returns=len(r_test),
            test_sharpe_annual=float(sharpe_test),
            test_dsr=float(dsr_test),
            test_max_drawdown=float(abs(res_test.metrics["max_drawdown"])),
            test_n_trades=int(res_test.metrics["n_trades"]),
        ))

        if fold_idx < 3 or fold_idx == n_splits - 1:
            print(f"  fold {fold_idx:2d}: SR_OOS={sharpe_test:+.3f}  "
                  f"PSR_OOS={dsr_test:.3f}  "
                  f"|MDD|={abs(res_test.metrics['max_drawdown']):.2%}  "
                  f"N_trades={res_test.metrics['n_trades']}")
        elif fold_idx == 3:
            print("  ...")

    # -----------------------------------------------------------------
    # Agregar
    # -----------------------------------------------------------------
    print(f"\n[AGGREGATION] {len(fold_reports)} folds completados.")
    agg = aggregate_folds(fold_reports)
    print("\n>> AGREGADO ENTRE FOLDS")
    for k, v in agg.items():
        if isinstance(v, list):
            print(f"  {k:35s} [{v[0]:.4f}, {v[1]:.4f}]")
        else:
            print(f"  {k:35s} {v}")

    # -----------------------------------------------------------------
    # Reportes IS y OOS concatenados
    # -----------------------------------------------------------------
    is_returns = pd.concat(is_returns_concat).dropna()
    oos_returns = pd.concat(oos_returns_concat).dropna()

    # n_trials = número de configuraciones probadas. Aquí "1" porque solo
    # ejecutamos un threshold. En producción con grid search, sé honesto.
    n_trials_search = 1
    spec = ObjectiveSpec(
        primary="dsr",
        constraints=HardConstraints(
            max_drawdown=0.20,
            min_skew=-0.5,
            min_n_trades=30,
            min_psr=0.95,
            min_dsr=0.95,
            min_sharpe_annual=0.0,
        ),
        n_trials=n_trials_search,
        periods_per_year=bars_per_year,
    )

    n_trades_oos = int(sum(f.test_n_trades for f in fold_reports))

    rep_is = build_report(
        is_returns, label="IS (combinado)",
        n_trades=None,
        objective_spec=None,  # no aplicamos veredicto a IS
    )
    rep_oos = build_report(
        oos_returns, label="OOS (combinado)",
        n_trades=n_trades_oos,
        objective_spec=spec,
    )

    print()
    print_report(rep_is)
    print()
    print_report(rep_oos)

    # -----------------------------------------------------------------
    # Comparación IS vs OOS
    # -----------------------------------------------------------------
    print("\n>> COMPARACIÓN IS vs OOS (overfitting check)")
    comp = compare_is_oos(rep_is, rep_oos)
    print(f"  {'metric':22s} {'IS':>10s} {'OOS':>10s} {'haircut':>10s}")
    for k, v in comp.items():
        haircut = v.get("haircut")
        haircut_str = f"{haircut:>10.2%}" if haircut is not None and not np.isnan(haircut) else f"{'n/a':>10s}"
        print(f"  {k:22s} {v['is']:>10.4f} {v['oos']:>10.4f} {haircut_str}")

    print("\n" + "█" * 75)
    print(" DEMO COMPLETADO. Pipeline operativo.")
    print("█" * 75)
    return rep_is, rep_oos, fold_reports


if __name__ == "__main__":
    run_demo()
