"""
Pipeline Orchestrator
=====================
Pone todo junto: ingesta → features → walk-forward training → backtest → métricas.

Este es el archivo que ejecutas para validar una estrategia completa.

Uso:
    python -m quant_bot.pipeline --symbol BTC/USDT --timeframe 1h --model xgboost
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# Imports relativos al proyecto
from data.ingestion import OHLCVIngestor, clean_ohlcv
from features.engineering import FeatureBuilder, triple_barrier_labels
from models.zoo import get_model, BaseModel
from models.validation import WalkForwardSplitter
from backtesting.engine import Backtester, BacktestConfig
from risk.management import IntegratedRiskManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    # Datos
    exchange: str = 'binance'
    symbol: str = 'BTC/USDT'
    timeframe: str = '1h'
    since: str = '2021-01-01'
    until: str = '2025-01-01'

    # Features / Target
    horizon: int = 24                  # 24 barras horarias = 1 día
    upper_barrier_mult: float = 2.0
    lower_barrier_mult: float = 2.0

    # Modelo
    model_name: str = 'xgboost'

    # Walk-forward
    train_size: int = 24 * 365         # 1 año de datos horarios
    test_size: int = 24 * 30           # 1 mes para validación
    embargo: int = 24                  # 24 barras = 1 día de embargo

    # Backtest
    initial_capital: float = 10_000.0
    fee_bps: float = 10.0
    slippage_bps: float = 5.0

    # Risk
    target_vol: float = 0.15
    soft_dd: float = 0.10
    hard_dd: float = 0.20


def run_pipeline(cfg: PipelineConfig):
    # ========================================================================
    # 1. INGESTA
    # ========================================================================
    logger.info("=" * 70)
    logger.info("FASE 1: INGESTA DE DATOS")
    logger.info("=" * 70)

    ingestor = OHLCVIngestor(exchange=cfg.exchange)
    raw = ingestor.fetch_historical(cfg.symbol, cfg.timeframe, cfg.since, cfg.until)
    raw = clean_ohlcv(raw)
    logger.info(f"Datos descargados: {len(raw)} velas, desde {raw.index[0]} a {raw.index[-1]}")

    # ========================================================================
    # 2. FEATURE ENGINEERING
    # ========================================================================
    logger.info("=" * 70)
    logger.info("FASE 2: FEATURE ENGINEERING")
    logger.info("=" * 70)

    fb = FeatureBuilder()
    features = fb.build(raw)

    # Target con triple-barrier
    target = triple_barrier_labels(
        raw['close'],
        horizon=cfg.horizon,
        upper_mult=cfg.upper_barrier_mult,
        lower_mult=cfg.lower_barrier_mult,
    )

    # Alinear y eliminar NaN
    df = features.join(target.rename('target')).dropna()
    X = df.drop(columns=['target'])
    y = df['target'].astype(int)

    logger.info(f"Features generadas: {X.shape[1]} columnas, {X.shape[0]} muestras")
    logger.info(f"Distribución de target:\n{y.value_counts(normalize=True).to_string()}")

    # ========================================================================
    # 3. WALK-FORWARD TRAINING
    # ========================================================================
    logger.info("=" * 70)
    logger.info("FASE 3: WALK-FORWARD VALIDATION")
    logger.info("=" * 70)

    splitter = WalkForwardSplitter(
        train_size=cfg.train_size,
        test_size=cfg.test_size,
        embargo=cfg.embargo,
        expanding=False,
    )

    n_splits = splitter.get_n_splits(X)
    logger.info(f"Número de folds walk-forward: {n_splits}")

    # Acumulamos predicciones out-of-sample
    oos_predictions = pd.Series(index=X.index, dtype=float)
    oos_signals = pd.Series(index=X.index, dtype=float)

    fold_metrics = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        logger.info(f"Train: {X.index[train_idx[0]]} → {X.index[train_idx[-1]]} ({len(train_idx)} muestras)")
        logger.info(f"Test:  {X.index[test_idx[0]]} → {X.index[test_idx[-1]]} ({len(test_idx)} muestras)")

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        model = get_model(cfg.model_name)
        model.fit(X_train, y_train)

        # Predicciones probabilísticas
        proba = model.predict_proba(X_test)
        # Asumimos clases [-1, 0, 1] mapeadas
        if hasattr(model, 'inv_label_map_') and model.inv_label_map_:
            class_order = [model.inv_label_map_[i] for i in range(proba.shape[1])]
        else:
            class_order = sorted(np.unique(y_train))

        # Señal: prob(clase 1) - prob(clase -1) > umbral → long
        proba_df = pd.DataFrame(proba, columns=class_order, index=X_test.index)
        if 1 in proba_df.columns and -1 in proba_df.columns:
            net_signal = proba_df[1] - proba_df[-1]
        else:
            net_signal = pd.Series(0, index=X_test.index)

        # Convertir a señal discreta con umbral conservador
        threshold = 0.15
        signals = pd.Series(0, index=X_test.index)
        signals[net_signal > threshold] = 1
        signals[net_signal < -threshold] = -1

        oos_predictions.loc[X_test.index] = net_signal
        oos_signals.loc[X_test.index] = signals

        # Accuracy direccional como sanity check (NO es la métrica final)
        non_zero = signals != 0
        if non_zero.sum() > 0:
            hit_rate = (signals[non_zero] == y_test[non_zero]).mean()
            logger.info(f"Hit rate (cuando se opera): {hit_rate:.3f}")
            logger.info(f"Trades generados en este fold: {non_zero.sum()}")
            fold_metrics.append({
                'fold': fold_idx,
                'hit_rate': hit_rate,
                'n_trades': non_zero.sum(),
                'avg_signal_strength': net_signal.abs().mean(),
            })

    # ========================================================================
    # 4. BACKTEST CON SEÑALES OUT-OF-SAMPLE
    # ========================================================================
    logger.info("=" * 70)
    logger.info("FASE 4: BACKTEST OUT-OF-SAMPLE")
    logger.info("=" * 70)

    # Solo backtesteamos en el período donde tenemos predicciones OOS
    valid = oos_signals.dropna()
    backtest_prices = raw.loc[valid.index]
    backtest_signals = valid

    bt_config = BacktestConfig(
        initial_capital=cfg.initial_capital,
        fee_bps=cfg.fee_bps,
        slippage_bps=cfg.slippage_bps,
        max_position_pct=0.95,
        allow_short=False,
    )

    risk_mgr = IntegratedRiskManager(
        target_vol=cfg.target_vol,
        soft_dd=cfg.soft_dd,
        hard_dd=cfg.hard_dd,
    )

    bt = Backtester(bt_config)
    result = bt.run(backtest_prices, backtest_signals, position_sizer=risk_mgr)

    # ========================================================================
    # 5. REPORTE FINAL
    # ========================================================================
    logger.info("=" * 70)
    logger.info("FASE 5: RESULTADOS FINALES")
    logger.info("=" * 70)
    print(result.summary())

    # Comparar con buy & hold
    bh_return = backtest_prices['close'].iloc[-1] / backtest_prices['close'].iloc[0] - 1
    logger.info(f"\nComparación Buy & Hold: {bh_return*100:.2f}%")
    logger.info(f"Estrategia vs B&H: {(result.metrics['total_return'] - bh_return)*100:+.2f}% de alpha")

    return result, oos_signals, fold_metrics


if __name__ == '__main__':
    cfg = PipelineConfig()
    run_pipeline(cfg)
