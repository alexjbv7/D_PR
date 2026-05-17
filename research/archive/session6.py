# session6.py - Experimento cross-asset (SPY, GLD, EURUSD)
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, balanced_accuracy_score

from data.multi_asset import fetch_asset
from data.macro import fetch_macro
from features.engineering import (
    FeatureBuilder, triple_barrier_labels, rolling_correlation
)
from models.zoo import XGBoostClassifier
from models.validation import WalkForwardSplitter
from backtesting.engine import Backtester, BacktestConfig

# ============================================================
# CONFIGURACION DEL EXPERIMENTO
# ============================================================
ASSETS_TO_TEST = ['spy_etf', 'gld', 'eurusd']

PERIOD_START = '2010-01-01'
PERIOD_END = '2025-01-31'

# Fees realistas por tipo de activo (en bps, 1 bp = 0.01%)
FEES_BY_ASSET = {
    'spy_etf': 0.5,   # IBKR moderno
    'gld': 0.5,
    'eurusd': 0.5,    # ~0.5 pip
}

# Walk-forward parametros
WF_TRAIN = 500       # ~2 anios
WF_TEST = 60         # 2 meses
WF_EMBARGO = 25

# Triple-barrier
TB_HORIZON = 20
TB_UPPER = 2.0
TB_LOWER = 2.0

# XGBoost params (mismos que sesion 3.3)
XGB_PARAMS = {
    'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 0.5, 'reg_lambda': 1.0,
    'min_child_weight': 10, 'gamma': 0.1,
    'random_state': 42, 'n_jobs': -1,
}

# ============================================================
# 1. DESCARGA TODOS LOS ACTIVOS + MACRO
# ============================================================
print("=" * 70)
print("PASO 1: Descargando activos y datos macro")
print("=" * 70)

assets_data = {}
for asset_id in ASSETS_TO_TEST:
    df = fetch_asset(asset_id, start=PERIOD_START, end=PERIOD_END)
    assets_data[asset_id] = df

# Macro: ya tenemos sp500, nasdaq, dxy, vix, gold cacheados
print("\nDescargando macro contextual...")
macro = fetch_macro(start=PERIOD_START, end=PERIOD_END)
print(f"Macro: {len(macro)} filas")

# ============================================================
# 2. CICLO POR CADA ACTIVO
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: Pipeline completo por cada activo")
print("=" * 70)

results = {}

for asset_id in ASSETS_TO_TEST:
    print(f"\n{'#' * 70}")
    print(f"# ACTIVO: {asset_id.upper()}")
    print(f"{'#' * 70}")

    df = assets_data[asset_id].copy()

    # ---- 2.1 Feature engineering ----
    print(f"\n[{asset_id}] Construyendo features...")
    fb = FeatureBuilder()
    features = fb.build(df, df_eth=None)  # sin ETH cross-asset

    # Anadir correlaciones macro
    df_returns = np.log(df['close'] / df['close'].shift(1))
    macro_aligned = macro.reindex(df.index, method='ffill')
    for asset_macro in macro_aligned.columns:
        macro_returns = np.log(macro_aligned[asset_macro] / macro_aligned[asset_macro].shift(1))
        features[f'corr_{asset_macro}_30'] = rolling_correlation(
            df_returns, macro_returns, window=30
        )

    # Eliminar la columna corr_btc_eth (no aplica)
    if 'corr_btc_eth_30' in features.columns:
        features = features.drop(columns=['corr_btc_eth_30'])

    print(f"[{asset_id}] Features: {features.shape[1]}")

    # ---- 2.2 Target con triple-barrier ----
    print(f"[{asset_id}] Triple-barrier labeling...")
    target = triple_barrier_labels(
        df['close'], horizon=TB_HORIZON,
        upper_mult=TB_UPPER, lower_mult=TB_LOWER, vol_period=20,
    )

    # ---- 2.3 Limpiar dataset ----
    full = features.copy()
    full['target'] = target
    full = full.dropna()

    X = full.drop(columns=['target'])
    y = full['target'].astype(int)

    print(f"[{asset_id}] Muestras limpias: {len(X)}")
    print(f"[{asset_id}] Distribucion target:")
    for label, prop in y.value_counts(normalize=True).sort_index().items():
        name = {-1: 'Baj', 0: 'Lat', 1: 'Alc'}[label]
        print(f"    {name}: {prop*100:.1f}%")

    # ---- 2.4 Walk-forward training ----
    splitter = WalkForwardSplitter(
        train_size=WF_TRAIN, test_size=WF_TEST, embargo=WF_EMBARGO, expanding=False
    )
    n_splits = splitter.get_n_splits(X)
    print(f"\n[{asset_id}] Folds walk-forward: {n_splits}")

    if n_splits == 0:
        print(f"  ERROR: insuficientes datos. Saltando {asset_id}.")
        continue

    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    all_indices = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Sample weights balanceados
        class_counts = y_train.value_counts()
        n_total = len(y_train)
        n_classes = len(class_counts)
        sample_weights = y_train.map(
            lambda lab: n_total / (n_classes * class_counts[lab])
        ).values

        model = XGBoostClassifier(**XGB_PARAMS)
        model.fit(X_train, y_train, sample_weight=sample_weights)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        all_y_true.extend(y_test.values)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
        all_indices.extend(X_test.index)

    # Metricas
    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)
    acc = accuracy_score(y_true_arr, y_pred_arr)
    bal_acc = balanced_accuracy_score(y_true_arr, y_pred_arr)

    print(f"\n[{asset_id}] Accuracy OOS:          {acc*100:.2f}%")
    print(f"[{asset_id}] Balanced Accuracy OOS: {bal_acc*100:.2f}%")

    # ---- 2.5 Backtest ----
    proba_arr = np.array(all_y_proba)
    proba_df = pd.DataFrame(proba_arr, index=all_indices,
                             columns=['proba_baj', 'proba_lat', 'proba_alc'])

    # Precios para el backtest (solo periodo OOS)
    backtest_idx = pd.DatetimeIndex(all_indices)
    prices_bt = df.loc[backtest_idx].copy()

    # 3 senales
    signals_bh = pd.Series(1, index=prices_bt.index)
    signals_agg = pd.Series(0, index=prices_bt.index, dtype=float)
    signals_agg[y_pred_arr == 1] = 1
    signals_cons = pd.Series(0, index=prices_bt.index, dtype=float)
    signals_cons[proba_df['proba_alc'] > 0.5] = 1

    config = BacktestConfig(
        initial_capital=10_000.0,
        fee_bps=FEES_BY_ASSET[asset_id],
        slippage_bps=2.0,
        max_position_pct=0.95,
        allow_short=False,
    )
    bt = Backtester(config)

    r_bh = bt.run(prices_bt, signals_bh)
    r_agg = bt.run(prices_bt, signals_agg)
    r_cons = bt.run(prices_bt, signals_cons)

    results[asset_id] = {
        'asset': asset_id,
        'period_oos': (backtest_idx[0].date(), backtest_idx[-1].date()),
        'n_oos': len(backtest_idx),
        'accuracy': acc,
        'balanced_acc': bal_acc,
        'bh_return': r_bh.metrics['total_return'],
        'bh_sharpe': r_bh.metrics['sharpe'],
        'bh_mdd': r_bh.metrics['max_drawdown'],
        'agg_return': r_agg.metrics['total_return'],
        'agg_sharpe': r_agg.metrics['sharpe'],
        'agg_mdd': r_agg.metrics['max_drawdown'],
        'agg_trades': r_agg.metrics['n_trades'],
        'cons_return': r_cons.metrics['total_return'],
        'cons_sharpe': r_cons.metrics['sharpe'],
        'cons_mdd': r_cons.metrics['max_drawdown'],
        'cons_trades': r_cons.metrics['n_trades'],
    }

    print(f"\n[{asset_id}] BACKTEST OOS:")
    print(f"  B&H:           ret={r_bh.metrics['total_return']*100:+7.2f}%  sharpe={r_bh.metrics['sharpe']:+.2f}  mdd={r_bh.metrics['max_drawdown']*100:.2f}%")
    print(f"  Agresivo:      ret={r_agg.metrics['total_return']*100:+7.2f}%  sharpe={r_agg.metrics['sharpe']:+.2f}  mdd={r_agg.metrics['max_drawdown']*100:.2f}%  ({int(r_agg.metrics['n_trades'])} trades)")
    print(f"  Conservador:   ret={r_cons.metrics['total_return']*100:+7.2f}%  sharpe={r_cons.metrics['sharpe']:+.2f}  mdd={r_cons.metrics['max_drawdown']*100:.2f}%  ({int(r_cons.metrics['n_trades'])} trades)")

# ============================================================
# 3. TABLA COMPARATIVA FINAL
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: TABLA COMPARATIVA FINAL CROSS-ASSET")
print("=" * 70)

if results:
    summary = pd.DataFrame(results).T
    summary_display = summary[['accuracy', 'bh_return', 'bh_sharpe',
                                'agg_return', 'agg_sharpe', 'agg_mdd',
                                'cons_return', 'cons_sharpe', 'cons_mdd']].copy()

    # Format
    for col in ['accuracy', 'bh_return', 'agg_return', 'agg_mdd', 'cons_return', 'cons_mdd']:
        summary_display[col] = summary_display[col].apply(lambda x: f"{x*100:+.2f}%")
    for col in ['bh_sharpe', 'agg_sharpe', 'cons_sharpe']:
        summary_display[col] = summary_display[col].apply(lambda x: f"{x:+.2f}")

    print("\n" + summary_display.to_string())

    # ============================================================
    # 4. VEREDICTO HONESTO vs HIPOTESIS
    # ============================================================
    print("\n" + "=" * 70)
    print("PASO 4: VEREDICTO vs hipotesis previa")
    print("=" * 70)

    print(f"""
HIPOTESIS REGISTRADA:
  - Minimo:    Sharpe OOS > 0  (en alguno de los 3 activos)
  - Bueno:     Sharpe > 0.5 con MDD < 25%
  - Excelente: vencer al B&H

RESULTADOS POR ACTIVO:
""")

    for asset_id, r in results.items():
        best_sharpe = max(r['agg_sharpe'], r['cons_sharpe'])
        best_strategy = 'Agresivo' if r['agg_sharpe'] > r['cons_sharpe'] else 'Conservador'
        bh_sharpe = r['bh_sharpe']
        mdd_acceptable = min(r['agg_mdd'], r['cons_mdd']) > -0.25

        cumple_minimo = best_sharpe > 0
        cumple_bueno = best_sharpe > 0.5 and mdd_acceptable
        cumple_excelente = best_sharpe > bh_sharpe + 0.3

        verdict = []
        if cumple_excelente:
            verdict.append("EXCELENTE (vence B&H)")
        elif cumple_bueno:
            verdict.append("BUENO (Sharpe sostenible)")
        elif cumple_minimo:
            verdict.append("MARGINAL (gana algo)")
        else:
            verdict.append("FRACASA (perdio dinero)")

        print(f"  {asset_id:10s}: mejor estrategia={best_strategy:12s} sharpe={best_sharpe:+.2f}  vs B&H={bh_sharpe:+.2f}  -> {' / '.join(verdict)}")

    # Conclusion global
    n_winners = sum(1 for r in results.values()
                    if max(r['agg_sharpe'], r['cons_sharpe']) > 0)
    n_total = len(results)

    print(f"\n--- CONCLUSION GLOBAL ---")
    print(f"Activos con sistema rentable: {n_winners}/{n_total}")

    if n_winners == 0:
        print("\nVEREDICTO: hipotesis REFUTADA. El sistema falla en TODOS los activos.")
        print("Esto confirma que el problema es la PREGUNTA (prediccion direccional con TA),")
        print("no el activo. Pivot recomendado: Camino 1 (volatilidad/regimen).")
    elif n_winners == n_total:
        print("\nVEREDICTO: hipotesis CONFIRMADA. El sistema funciona en todos los activos.")
        print("BTC era estructuralmente atipico. Tu pipeline es viable.")
    else:
        print(f"\nVEREDICTO: hipotesis PARCIAL. Funciona en algunos, no en todos.")
        print(f"Investigar QUE activo funciona y por que.")

# ============================================================
# 5. VISUALIZACION COMPARATIVA
# ============================================================
print("\n" + "=" * 70)
print("PASO 5: Generando grafico comparativo...")
print("=" * 70)

if results:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Cross-asset experiment: Sharpe Ratio comparativa',
                 fontsize=14, fontweight='bold')

    # Panel 1: Sharpe por estrategia y activo
    ax = axes[0]
    asset_ids = list(results.keys())
    x = np.arange(len(asset_ids))
    width = 0.27

    bh_sharpes = [results[a]['bh_sharpe'] for a in asset_ids]
    agg_sharpes = [results[a]['agg_sharpe'] for a in asset_ids]
    cons_sharpes = [results[a]['cons_sharpe'] for a in asset_ids]

    ax.bar(x - width, bh_sharpes, width, label='Buy & Hold', color='#1976D2', alpha=0.8)
    ax.bar(x, agg_sharpes, width, label='Modelo Agresivo', color='#F57C00', alpha=0.8)
    ax.bar(x + width, cons_sharpes, width, label='Modelo Conservador', color='#388E3C', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(asset_ids, fontsize=10)
    ax.set_ylabel('Sharpe Ratio')
    ax.set_title('Sharpe por estrategia y activo')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Retorno total por estrategia y activo
    ax = axes[1]
    bh_rets = [results[a]['bh_return']*100 for a in asset_ids]
    agg_rets = [results[a]['agg_return']*100 for a in asset_ids]
    cons_rets = [results[a]['cons_return']*100 for a in asset_ids]

    ax.bar(x - width, bh_rets, width, label='Buy & Hold', color='#1976D2', alpha=0.8)
    ax.bar(x, agg_rets, width, label='Modelo Agresivo', color='#F57C00', alpha=0.8)
    ax.bar(x + width, cons_rets, width, label='Modelo Conservador', color='#388E3C', alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(asset_ids, fontsize=10)
    ax.set_ylabel('Retorno total (%)')
    ax.set_title('Retorno total OOS')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('session6_cross_asset.png', dpi=120, bbox_inches='tight')
    print("Grafico guardado: session6_cross_asset.png")
    plt.show()

# Guardar tabla
if results:
    pd.DataFrame(results).T.to_csv('./cache/cross_asset_results.csv')
    print("\nTabla guardada: ./cache/cross_asset_results.csv")

print("\nSesion 6 completada.")