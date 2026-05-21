"""
Demo end-to-end: VALIDATION FRAMEWORK + MULTI-ASSET ENGINE
==========================================================
Demuestra el pipeline COMPLETO sobre FX (EURUSD) y futuros (ES) sintéticos:

  1. Genera datos sintéticos OHLCV con régimen y vol clustering.
  2. Ejecuta una estrategia momentum sobre cada instrumento.
  3. Walk-forward validation con embargo.
  4. Backtester multi-asset con spreads + comisiones + swap reales.
  5. Métricas avanzadas (DSR, PSR, bootstrap CI) + veredicto formal.

Uso:
    cd quant_bot
    python examples/validation_demo_fx_futures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from instruments import EURUSD, ES, get_instrument
from instruments.specs import InstrumentSpec
from models.validation import WalkForwardSplitter
from backtesting.multi_asset_engine import (
    MultiAssetBacktester,
    MultiAssetBacktestConfig,
)
from risk.sizing_multi_asset import ATRRiskSizer
from metrics.objective import ObjectiveSpec, HardConstraints
from reporting.report import (
    build_report,
    compare_is_oos,
    aggregate_folds,
    FoldReport,
    print_report,
)


# =====================================================================
# DATOS SINTÉTICOS: FX y FUTUROS
# =====================================================================

def synthetic_fx(
    n_bars: int = 5000,
    base: float = 1.1000,
    annual_drift: float = 0.0,
    annual_vol: float = 0.08,        # ~8% vol anual típica EURUSD
    bars_per_year: int = 252 * 24,   # horario, 24h
    seed: int = 7,
) -> pd.DataFrame:
    """
    EURUSD-like OHLCV con vol clustering simple.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year

    # log-vol AR(1)
    log_vol = np.zeros(n_bars)
    log_vol[0] = np.log(annual_vol)
    for t in range(1, n_bars):
        log_vol[t] = (
            0.97 * log_vol[t - 1]
            + 0.03 * np.log(annual_vol)
            + rng.normal(0, 0.04)
        )
    sigma_t = np.exp(log_vol)
    z = rng.standard_normal(n_bars)
    log_returns = (annual_drift - 0.5 * sigma_t ** 2) * dt \
        + sigma_t * np.sqrt(dt) * z
    close = base * np.exp(np.cumsum(log_returns))
    open_ = np.concatenate([[base], close[:-1]])
    intra_range = sigma_t * np.sqrt(dt) * close * 0.5
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1, n_bars)) * intra_range / 2
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1, n_bars)) * intra_range / 2
    volume = rng.lognormal(8, 0.5, n_bars)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


def synthetic_es_future(
    n_bars: int = 5000,
    base: float = 4500.0,
    annual_drift: float = 0.07,
    annual_vol: float = 0.18,        # ~18% vol anual típica
    bars_per_year: int = 252 * 24,
    seed: int = 13,
) -> pd.DataFrame:
    """ES-like OHLCV. Drift positivo modesto (renta variable equity premium)."""
    rng = np.random.default_rng(seed)
    dt = 1.0 / bars_per_year
    log_vol = np.zeros(n_bars)
    log_vol[0] = np.log(annual_vol)
    for t in range(1, n_bars):
        log_vol[t] = 0.97 * log_vol[t - 1] + 0.03 * np.log(annual_vol) \
            + rng.normal(0, 0.05)
    sigma_t = np.exp(log_vol)
    z = rng.standard_normal(n_bars)
    log_returns = (annual_drift - 0.5 * sigma_t ** 2) * dt \
        + sigma_t * np.sqrt(dt) * z
    close = base * np.exp(np.cumsum(log_returns))
    # Redondear a tick válido (0.25)
    close = np.round(close * 4) / 4
    open_ = np.concatenate([[base], close[:-1]])
    intra_range = sigma_t * np.sqrt(dt) * close * 0.7
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 1, n_bars)) * intra_range / 2
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 1, n_bars)) * intra_range / 2
    volume = rng.lognormal(10, 0.4, n_bars)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )


# =====================================================================
# ESTRATEGIA TRIVIAL (sólo para validar pipeline)
# =====================================================================

def momentum_signal(prices: pd.DataFrame, fast: int = 12, slow: int = 48,
                    threshold: float = 0.0005) -> pd.Series:
    close = prices["close"]
    ma_fast = close.rolling(fast).mean()
    ma_slow = close.rolling(slow).mean()
    spread = (ma_fast - ma_slow) / ma_slow
    sig = pd.Series(0, index=close.index, dtype=int)
    sig[spread > threshold] = 1
    sig[spread < -threshold] = -1
    return sig


# =====================================================================
# PIPELINE PARA UN INSTRUMENTO
# =====================================================================

def run_for_instrument(
    instrument: InstrumentSpec,
    prices: pd.DataFrame,
    risk_pct: float = 0.005,
    initial_capital: float = 25_000.0,
    bars_per_year: int = 252 * 24,
):
    print(f"\n{'█' * 75}\n DEMO: {instrument.symbol} ({instrument.display_name})\n{'█' * 75}")
    print(f"  Datos: {len(prices)} barras, {prices.index[0]} → {prices.index[-1]}")

    # Walk-forward
    splitter = WalkForwardSplitter(
        train_size=24 * 90,
        test_size=24 * 30,
        embargo=24,
        expanding=False,
    )
    n_splits = splitter.get_n_splits(prices)
    print(f"  Walk-forward folds: {n_splits}")

    bt_config = MultiAssetBacktestConfig(
        initial_capital=initial_capital,
        extra_slippage_in_price=0.0,
        slippage_vol_mult=0.3,
        apply_swap=instrument.is_forex,
        risk_free_rate=0.04,
        allow_short=True,
    )

    sizer = ATRRiskSizer(
        risk_pct=risk_pct,
        atr_stop_mult=2.0,
        instrument=instrument,
        max_units_per_trade=10 if instrument.is_future else 5.0,
        daily_loss_pct_pause=0.03,
    )

    fold_reports: list[FoldReport] = []
    is_returns_concat: list[pd.Series] = []
    oos_returns_concat: list[pd.Series] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(prices)):
        train_slice = prices.iloc[train_idx]
        test_slice = prices.iloc[test_idx]

        # Generar señales con warm-up para evitar NaN al inicio del test
        warmup_start = max(0, test_idx[0] - 48)
        warmed = prices.iloc[warmup_start:test_idx[-1] + 1]
        sig_warmed = momentum_signal(warmed, fast=12, slow=48)
        sig_test = sig_warmed.loc[test_slice.index]
        sig_train = momentum_signal(train_slice, fast=12, slow=48)

        bt = MultiAssetBacktester(instrument, bt_config)
        res_train = bt.run(train_slice, sig_train, sizer)
        # NUEVO sizer por fold (estado limpio del daily-loss tracker)
        sizer_test = ATRRiskSizer(
            risk_pct=risk_pct, atr_stop_mult=2.0, instrument=instrument,
            max_units_per_trade=10 if instrument.is_future else 5.0,
        )
        res_test = bt.run(test_slice, sig_test, sizer_test)

        is_returns_concat.append(res_train.returns)
        oos_returns_concat.append(res_test.returns)

        r_test = res_test.returns
        sigma = r_test.std(ddof=1)
        sharpe_test = (
            np.sqrt(bars_per_year) * r_test.mean() / sigma if sigma > 0 else 0.0
        )

        from metrics.advanced import (
            probabilistic_sharpe_ratio,
            returns_skewness,
            returns_kurtosis,
        )
        sr_period = r_test.mean() / sigma if sigma > 0 else 0.0
        psr_test = probabilistic_sharpe_ratio(
            sr_period, len(r_test),
            returns_skewness(r_test),
            returns_kurtosis(r_test, excess=False),
        )

        fold_reports.append(FoldReport(
            fold_idx=fold_idx,
            train_start=str(train_slice.index[0]),
            train_end=str(train_slice.index[-1]),
            test_start=str(test_slice.index[0]),
            test_end=str(test_slice.index[-1]),
            n_test_returns=len(r_test),
            test_sharpe_annual=float(sharpe_test),
            test_dsr=float(psr_test),
            test_max_drawdown=float(abs(res_test.metrics["max_drawdown"])),
            test_n_trades=int(res_test.metrics["n_trades"]),
        ))

        if fold_idx < 3 or fold_idx == n_splits - 1:
            print(f"   fold {fold_idx:2d}: SR_OOS={sharpe_test:+.3f}  "
                  f"PSR={psr_test:.3f}  "
                  f"|MDD|={abs(res_test.metrics['max_drawdown']):.2%}  "
                  f"trades={res_test.metrics['n_trades']}  "
                  f"final=${res_test.equity.iloc[-1]:.2f}")
        elif fold_idx == 3:
            print("   ...")

    # Agregación
    print(f"\n  >> AGREGADO ({len(fold_reports)} folds)")
    agg = aggregate_folds(fold_reports)
    for k, v in agg.items():
        if isinstance(v, list):
            print(f"     {k:35s} [{v[0]:.4f}, {v[1]:.4f}]")
        else:
            print(f"     {k:35s} {v}")

    # IS vs OOS
    is_returns = pd.concat(is_returns_concat).dropna()
    oos_returns = pd.concat(oos_returns_concat).dropna()

    spec = ObjectiveSpec(
        primary="dsr",
        constraints=HardConstraints(
            max_drawdown=0.20,
            min_skew=-0.5,
            min_n_trades=30,
            min_psr=0.95,
            min_dsr=0.95,
            min_sharpe_annual=0.0,
            max_tail_ratio_inverse=2.0,
        ),
        n_trials=1,  # para producción: contabiliza honestamente
        periods_per_year=bars_per_year,
    )

    n_trades_total = int(sum(f.test_n_trades for f in fold_reports))

    rep_is = build_report(is_returns, label=f"{instrument.symbol} IS")
    rep_oos = build_report(
        oos_returns, label=f"{instrument.symbol} OOS",
        n_trades=n_trades_total, objective_spec=spec,
    )

    print()
    print_report(rep_oos)

    # IS vs OOS
    print(f"\n  >> COMPARACIÓN IS vs OOS — {instrument.symbol}")
    comp = compare_is_oos(rep_is, rep_oos)
    print(f"     {'metric':22s} {'IS':>10s} {'OOS':>10s} {'haircut':>10s}")
    for k, v in comp.items():
        haircut = v.get("haircut")
        if haircut is None or np.isnan(haircut):
            haircut_str = f"{'n/a':>10s}"
        else:
            haircut_str = f"{haircut:>10.2%}"
        print(f"     {k:22s} {v['is']:>10.4f} {v['oos']:>10.4f} {haircut_str}")

    return rep_oos


def main():
    print("\n" + "█" * 75)
    print(" DEMO MULTI-ASSET (FX + FUTURES) — VALIDATION FRAMEWORK")
    print("█" * 75)

    # ----------- FX -----------
    fx_prices = synthetic_fx(n_bars=5000, seed=7)
    rep_fx = run_for_instrument(
        instrument=EURUSD,
        prices=fx_prices,
        risk_pct=0.005,
        initial_capital=10_000.0,
    )

    # ----------- FUTURES -----------
    es_prices = synthetic_es_future(n_bars=5000, seed=13)
    rep_es = run_for_instrument(
        instrument=ES,
        prices=es_prices,
        risk_pct=0.005,
        initial_capital=25_000.0,    # ES requiere margen
    )

    print("\n" + "█" * 75)
    print(" DEMO COMPLETADO. Pipeline multi-asset operativo.")
    print(" Ambos instrumentos pasaron por el mismo protocolo de validación.")
    print("█" * 75)
    return rep_fx, rep_es


if __name__ == "__main__":
    main()
