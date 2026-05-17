"""
Demo: Pipeline Completo con Datos Reales
=========================================
Valida que el framework multi-asset funciona con OHLCV real (no sintético).

Instrumentos:
  - EURUSD  (Yahoo Finance: EURUSD=X, barras diarias)
  - ES      (Yahoo Finance: ES=F, E-mini S&P 500 front-month, barras diarias)

Pipeline:
  1. Descarga datos reales vía Yahoo Finance (caché parquet).
  2. Estrategia momentum simple (MA12 / MA48 diario).
  3. Walk-forward: 1 año train / 3 meses test / 5 días embargo.
  4. Motor multi-asset con spreads + comisiones reales.
  5. Sizing ATR-based en lots/contratos reales.
  6. Métricas avanzadas (DSR, PSR, bootstrap CI) + veredicto.
  7. Comparación IS vs OOS (haircut de overfitting).

NOTA SOBRE CAPITAL:
  ES exige capital significativo para que el ATR sizer produzca ≥1 contrato.
  Con bars diarias, ATR típico ≈ 40-60 puntos → stop_usd ≈ $4k-$6k por contrato.
  El demo usa $100k y risk_pct=5% → puede generar 0-1 contrato según el período.
  Para investigar el pipeline sin preocuparte por sizing, cambia a FixedUnitsSizer.

Uso:
    cd C:\\Users\\alexj\\OneDrive\\Desktop\\quant_bot
    .venv-1\\Scripts\\activate
    python examples/validation_demo_real_data.py
"""
from __future__ import annotations

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from data.real_data import fetch_real_data, list_available_symbols
from instruments import EURUSD, ES
from instruments.specs import InstrumentSpec
from models.validation import WalkForwardSplitter
from backtesting.multi_asset_engine import MultiAssetBacktester, MultiAssetBacktestConfig
from risk.sizing_multi_asset import ATRRiskSizer
from metrics.objective import ObjectiveSpec, HardConstraints
from reporting.report import (
    build_report,
    compare_is_oos,
    aggregate_folds,
    FoldReport,
    print_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s  %(message)s",
)

# =====================================================================
# CONFIGURACIÓN GLOBAL
# =====================================================================

BARS_PER_YEAR = 252          # barras diarias
DATA_START = "2018-01-01"    # ~7 años de historia
DATA_INTERVAL = "1d"

TRAIN_BARS = 252             # 1 año de entrenamiento
TEST_BARS = 63               # 1 trimestre de test
EMBARGO_BARS = 5             # 1 semana de embargo

# Caché en el directorio estándar del proyecto
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"


# =====================================================================
# ESTRATEGIA: MOMENTUM MA CROSSOVER (diario)
# =====================================================================

def momentum_signal(
    prices: pd.DataFrame,
    fast: int = 12,
    slow: int = 48,
    threshold: float = 0.001,
) -> pd.Series:
    """
    Señal de cruce de medias móviles. Simple — solo para validar el pipeline.

    +1 cuando MA(fast) > MA(slow) + threshold (tendencia alcista)
    -1 cuando MA(fast) < MA(slow) - threshold (tendencia bajista)
     0 en zona plana

    Con barras diarias: fast=12 ≈ 2.5 semanas, slow=48 ≈ 2.5 meses.
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
# RUNNER POR INSTRUMENTO
# =====================================================================

def run_instrument(
    instrument: InstrumentSpec,
    prices: pd.DataFrame,
    initial_capital: float,
    risk_pct: float,
    max_units: float,
    usd_conversion_series: pd.Series | None = None,
) -> tuple:
    """
    Ejecuta walk-forward + backtest sobre un instrumento con datos reales.
    Retorna (rep_oos, rep_is, fold_reports).
    """
    sym = instrument.symbol
    banner = f"{'█' * 70}\n DATOS REALES: {sym} ({instrument.display_name})\n{'█' * 70}"
    print(f"\n{banner}")
    print(f"  Barras: {len(prices):,}  |  {prices.index[0].date()} → {prices.index[-1].date()}")
    print(f"  Capital inicial: ${initial_capital:,.0f}  |  risk_pct: {risk_pct:.1%}")

    # Estadísticas básicas de los retornos del activo
    log_ret = np.log(prices["close"] / prices["close"].shift(1)).dropna()
    ann_vol = log_ret.std() * np.sqrt(BARS_PER_YEAR)
    print(f"  Vol anual estimada: {ann_vol:.1%}  |  N barras válidas: {len(log_ret):,}")

    # Walk-forward splitter
    splitter = WalkForwardSplitter(
        train_size=TRAIN_BARS,
        test_size=TEST_BARS,
        embargo=EMBARGO_BARS,
        expanding=False,
    )
    n_splits = splitter.get_n_splits(prices)
    if n_splits == 0:
        print(f"  ⚠️  Datos insuficientes para walk-forward. "
              f"Necesitas ≥ {TRAIN_BARS + EMBARGO_BARS + TEST_BARS} barras.")
        return None, None, []
    print(f"  Folds walk-forward: {n_splits}")

    bt_config = MultiAssetBacktestConfig(
        initial_capital=initial_capital,
        slippage_vol_mult=0.3,
        apply_swap=instrument.is_forex,
        risk_free_rate=0.04,
        allow_short=True,
    )

    fold_reports: list[FoldReport] = []
    is_returns_list: list[pd.Series] = []
    oos_returns_list: list[pd.Series] = []
    n_zero_trade_folds = 0

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(prices)):
        train_slice = prices.iloc[train_idx]
        test_slice = prices.iloc[test_idx]

        # Señal con warm-up para evitar NaN al inicio del periodo test
        warmup_start = max(0, test_idx[0] - 48)
        warmed = prices.iloc[warmup_start: test_idx[-1] + 1]
        sig_warmed = momentum_signal(warmed)
        sig_test = sig_warmed.loc[test_slice.index]
        sig_train = momentum_signal(train_slice)

        # USD conversion series si se pasa (para crosses non-USD-quoted)
        conv_train = (
            usd_conversion_series.loc[train_slice.index]
            if usd_conversion_series is not None else None
        )
        conv_test = (
            usd_conversion_series.loc[test_slice.index]
            if usd_conversion_series is not None else None
        )

        # Sizer fresco por fold (reset del daily-loss tracker)
        sizer = ATRRiskSizer(
            risk_pct=risk_pct,
            atr_stop_mult=2.0,
            instrument=instrument,
            max_units_per_trade=max_units,
        )

        bt = MultiAssetBacktester(instrument, bt_config)
        res_train = bt.run(train_slice, sig_train, sizer, conv_train)

        sizer_test = ATRRiskSizer(
            risk_pct=risk_pct, atr_stop_mult=2.0, instrument=instrument,
            max_units_per_trade=max_units,
        )
        res_test = bt.run(test_slice, sig_test, sizer_test, conv_test)

        if res_test.metrics["n_trades"] == 0:
            n_zero_trade_folds += 1

        is_returns_list.append(res_train.returns)
        oos_returns_list.append(res_test.returns)

        # Métricas por fold
        r_test = res_test.returns.dropna()
        sigma_t = r_test.std(ddof=1)
        sharpe_test = (
            np.sqrt(BARS_PER_YEAR) * r_test.mean() / sigma_t
            if sigma_t > 0 else 0.0
        )

        from metrics.advanced import (
            probabilistic_sharpe_ratio,
            returns_skewness,
            returns_kurtosis,
        )
        sr_period = r_test.mean() / sigma_t if sigma_t > 0 else 0.0
        psr_test = probabilistic_sharpe_ratio(
            sr_period, len(r_test),
            returns_skewness(r_test),
            returns_kurtosis(r_test, excess=False),
        )

        fold_reports.append(FoldReport(
            fold_idx=fold_idx,
            train_start=str(train_slice.index[0].date()),
            train_end=str(train_slice.index[-1].date()),
            test_start=str(test_slice.index[0].date()),
            test_end=str(test_slice.index[-1].date()),
            n_test_returns=len(r_test),
            test_sharpe_annual=float(sharpe_test),
            test_dsr=float(psr_test),
            test_max_drawdown=float(abs(res_test.metrics["max_drawdown"])),
            test_n_trades=int(res_test.metrics["n_trades"]),
        ))

        # Imprimir primeros 3 folds + último
        if fold_idx < 3 or fold_idx == n_splits - 1:
            n_tr = res_test.metrics["n_trades"]
            eq_f = res_test.equity.iloc[-1]
            print(
                f"   fold {fold_idx:2d}  "
                f"{test_slice.index[0].date()} → {test_slice.index[-1].date()}  "
                f"SR={sharpe_test:+.2f}  PSR={psr_test:.2f}  "
                f"|MDD|={abs(res_test.metrics['max_drawdown']):.1%}  "
                f"trades={n_tr}  equity=${eq_f:,.0f}"
            )
        elif fold_idx == 3:
            print("   ...")

    if n_zero_trade_folds > 0:
        pct = n_zero_trade_folds / len(fold_reports) * 100
        print(
            f"\n  ⚠️  {n_zero_trade_folds}/{len(fold_reports)} folds sin trades ({pct:.0f}%). "
            f"Capital insuficiente para el sizer ATR o señal siempre flat. "
            f"Considera aumentar capital o usar FixedUnitsSizer para debugging."
        )

    # Agregación across folds
    print(f"\n  >> AGREGADO ({len(fold_reports)} folds)")
    agg = aggregate_folds(fold_reports)
    for k, v in agg.items():
        if isinstance(v, list):
            print(f"     {k:40s} [{v[0]:+.3f}, {v[1]:+.3f}]")
        elif isinstance(v, float):
            print(f"     {k:40s} {v:+.4f}")
        else:
            print(f"     {k:40s} {v}")

    # Retornos IS y OOS concatenados
    is_returns = pd.concat(is_returns_list).dropna()
    oos_returns = pd.concat(oos_returns_list).dropna()

    spec = ObjectiveSpec(
        primary="dsr",
        constraints=HardConstraints(
            max_drawdown=0.25,
            min_skew=-1.0,
            min_n_trades=10,
            min_psr=0.90,
            min_dsr=0.90,
            min_sharpe_annual=-0.5,
            max_tail_ratio_inverse=3.0,
        ),
        n_trials=1,
        periods_per_year=BARS_PER_YEAR,
    )

    n_trades_total = int(sum(f.test_n_trades for f in fold_reports))

    rep_is = build_report(is_returns, label=f"{sym} IS")
    rep_oos = build_report(
        oos_returns,
        label=f"{sym} OOS (datos reales)",
        n_trades=n_trades_total,
        objective_spec=spec,
    )

    print()
    print_report(rep_oos)

    # IS vs OOS
    print(f"\n  >> IS vs OOS — {sym}")
    comp = compare_is_oos(rep_is, rep_oos)
    print(f"     {'métrica':25s} {'IS':>8s} {'OOS':>8s} {'haircut':>10s}")
    for k, v in comp.items():
        hc = v.get("haircut")
        hc_str = f"{hc:>10.1%}" if hc is not None and not np.isnan(hc) else f"{'n/a':>10s}"
        print(f"     {k:25s} {v['is']:>8.3f} {v['oos']:>8.3f} {hc_str}")

    return rep_oos, rep_is, fold_reports


# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n" + "█" * 70)
    print(" DEMO: DATOS REALES (Yahoo Finance)")
    print(" Objetivo: validar que el pipeline multi-asset funciona con OHLCV real")
    print("█" * 70)
    print(f"\n  Símbolos disponibles: {list_available_symbols()}\n")

    # ------------------------------------------------------------------
    # 1. DESCARGAR DATOS REALES
    # ------------------------------------------------------------------
    print("━" * 70)
    print(" DESCARGANDO DATOS (primera vez puede tardar; después usa caché)")
    print("━" * 70)

    try:
        eurusd_prices = fetch_real_data(
            "EURUSD",
            interval=DATA_INTERVAL,
            start=DATA_START,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"\n  ✗ Error descargando EURUSD: {e}")
        print("  Verifica tu conexión a internet o instala yfinance: pip install yfinance")
        return

    try:
        es_prices = fetch_real_data(
            "ES",
            interval=DATA_INTERVAL,
            start=DATA_START,
            cache_dir=CACHE_DIR,
        )
    except Exception as e:
        print(f"\n  ✗ Error descargando ES: {e}")
        print("  Verifica tu conexión a internet o instala yfinance: pip install yfinance")
        return

    # ------------------------------------------------------------------
    # 2. EURUSD — FX
    # ------------------------------------------------------------------
    rep_eurusd, rep_eurusd_is, folds_eurusd = run_instrument(
        instrument=EURUSD,
        prices=eurusd_prices,
        initial_capital=10_000.0,
        risk_pct=0.01,       # 1% risk/trade → ~0.05-0.15 lots por trade
        max_units=5.0,       # máximo 5 lots estándar
    )

    # ------------------------------------------------------------------
    # 3. ES — FUTURES
    # ------------------------------------------------------------------
    # Nota de capital:
    # ES daily ATR ≈ 40-60 puntos → stop_usd ≈ $4k-$6k por contrato.
    # Con $100k y risk_pct=5%: risk_usd=$5k → ~1 contrato cuando ATR ≤ 50 ptos.
    # Folds con ATR > 50 producirán 0 trades (correcto: el sizer rechaza el trade).
    rep_es, rep_es_is, folds_es = run_instrument(
        instrument=ES,
        prices=es_prices,
        initial_capital=100_000.0,
        risk_pct=0.05,       # 5% risk/trade — agresivo, necesario para 1 contrato ES
        max_units=5.0,       # máximo 5 contratos
    )

    # ------------------------------------------------------------------
    # 4. RESUMEN FINAL
    # ------------------------------------------------------------------
    print("\n" + "█" * 70)
    print(" RESUMEN FINAL")
    print("█" * 70)

    results = [
        ("EURUSD (FX, $10k)", rep_eurusd, folds_eurusd),
        ("ES (Futures, $100k)", rep_es, folds_es),
    ]

    print(f"\n  {'instrumento':25s} {'Sharpe OOS':>12s} {'DSR':>8s} {'|MDD|':>8s} {'trades':>8s} {'veredicto'}")
    print("  " + "-" * 80)

    for label, rep, folds in results:
        if rep is None:
            print(f"  {label:25s} {'sin datos':>12s}")
            continue

        adv = rep.advanced_metrics
        obj = rep.objective_result or {}
        n_trades = int(sum(f.test_n_trades for f in folds))
        mdd = max(f.test_max_drawdown for f in folds) if folds else float("nan")
        veredicto = obj.get("verdict", "N/A")

        print(
            f"  {label:25s} "
            f"{adv.get('sharpe_annual', float('nan')):>+12.3f} "
            f"{adv.get('dsr', float('nan')):>8.3f} "
            f"{mdd:>8.1%} "
            f"{n_trades:>8d} "
            f"  {veredicto}"
        )

    print("\n" + "█" * 70)
    print(" PIPELINE REAL COMPLETADO.")
    print()
    print(" Próximos pasos (cuando los datos reales funcionen):")
    print("   → Implementar estrategia con racional económico real (no solo MA crossover)")
    print("   → Añadir features: momentum multitemporal, vol-scaling de señal")
    print("   → Ejecutar sobre 5+ instrumentos y consolidar con métricas agregadas")
    print("   → Validar DSR > 0.95 y Sharpe OOS > 1.0 con fees realistas")
    print("█" * 70)


if __name__ == "__main__":
    main()
