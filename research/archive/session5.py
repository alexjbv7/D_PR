# session5.py - Re-experimento con dataset 2023+ (filtrado)
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report, confusion_matrix)

from models.zoo import XGBoostClassifier
from models.validation import WalkForwardSplitter
from backtesting.engine import Backtester, BacktestConfig

# ============================================================
# 1. CARGA Y FILTRADO TEMPORAL
# ============================================================
print("=" * 70)
print("PASO 1: Carga del dataset y filtrado a partir de 2023-04-01")
print("=" * 70)

df = pd.read_parquet('./cache/dataset_with_target.parquet')

# Filtrar a partir de la fecha critica
START_DATE = pd.Timestamp('2023-04-01', tz='UTC')
df = df[df.index >= START_DATE]
print(f"Periodo filtrado: {df.index[0].date()} -> {df.index[-1].date()}")
print(f"Muestras totales tras filtrado: {len(df)}")

NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume',
                    'in_anomalous_regime', 'target']
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols]
y = df['target']

# Limpieza
mask_valid = X.notna().all(axis=1) & y.notna()
X = X[mask_valid]
y = y[mask_valid]
df_valid = df.loc[mask_valid]

# Excluir regimenes anomalos
mask_clean = ~df_valid['in_anomalous_regime']
X_clean = X[mask_clean]
y_clean = y[mask_clean]
df_clean = df_valid[mask_clean]

print(f"Muestras limpias (sin regimenes anomalos): {len(X_clean)}")

# ============================================================
# 2. SEPARAR CONJUNTO VIRGINAL
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: Separar set virginal de validacion (ultimos 90 dias)")
print("=" * 70)

VIRGIN_DAYS = 90
cutoff_date = X_clean.index[-VIRGIN_DAYS]

X_cv = X_clean[X_clean.index < cutoff_date]
y_cv = y_clean[y_clean.index < cutoff_date]
df_cv = df_clean[df_clean.index < cutoff_date]

X_virgin = X_clean[X_clean.index >= cutoff_date]
y_virgin = y_clean[y_clean.index >= cutoff_date]
df_virgin = df_clean[df_clean.index >= cutoff_date]

print(f"Set CV (walk-forward):  {X_cv.index[0].date()} -> {X_cv.index[-1].date()}")
print(f"  Muestras: {len(X_cv)}")
print(f"\nSet VIRGINAL (no se toca hasta backtest final):")
print(f"  {X_virgin.index[0].date()} -> {X_virgin.index[-1].date()}")
print(f"  Muestras: {len(X_virgin)}")

print(f"\nDistribucion target en set virginal:")
for label, prop in y_virgin.value_counts(normalize=True).sort_index().items():
    name = {-1: 'Bajista', 0: 'Lateral', 1: 'Alcista'}[label]
    print(f"  {name:10s}: {prop*100:.1f}%")

# ============================================================
# 3. WALK-FORWARD CON XGBOOST EN SET CV
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: Walk-forward training en set CV (no toca virginal)")
print("=" * 70)

splitter = WalkForwardSplitter(
    train_size=180, test_size=30, embargo=25, expanding=False
)
n_splits = splitter.get_n_splits(X_cv)
print(f"Folds posibles: {n_splits}")

XGB_PARAMS = {
    'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 0.5, 'reg_lambda': 1.0,
    'min_child_weight': 10, 'gamma': 0.1,
    'random_state': 42, 'n_jobs': -1,
}

if n_splits == 0:
    print("ERROR: Sin suficientes datos para walk-forward. Reduciendo train_size...")
    splitter = WalkForwardSplitter(
        train_size=120, test_size=30, embargo=20, expanding=False
    )
    n_splits = splitter.get_n_splits(X_cv)
    print(f"Folds con train_size=120: {n_splits}")

all_y_true_cv = []
all_y_pred_cv = []
all_y_proba_cv = []
all_indices_cv = []
fold_metrics = []
all_importances = []

for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_cv)):
    X_train = X_cv.iloc[train_idx]
    y_train = y_cv.iloc[train_idx]
    X_test = X_cv.iloc[test_idx]
    y_test = y_cv.iloc[test_idx]

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

    all_y_true_cv.extend(y_test.values)
    all_y_pred_cv.extend(y_pred)
    all_y_proba_cv.extend(y_proba)
    all_indices_cv.extend(X_test.index)
    all_importances.append(model.feature_importance())

    fold_acc = accuracy_score(y_test, y_pred)
    fold_bal = balanced_accuracy_score(y_test, y_pred)
    fold_metrics.append({
        'fold': fold_idx + 1,
        'test_start': X_test.index[0].date(),
        'test_end': X_test.index[-1].date(),
        'accuracy': fold_acc,
        'balanced_acc': fold_bal,
    })

    print(f"  Fold {fold_idx+1:2d}: "
          f"test {X_test.index[0].date()} -> {X_test.index[-1].date()} | "
          f"acc={fold_acc:.3f} | bal_acc={fold_bal:.3f}")

# ============================================================
# 4. METRICAS GLOBALES CV
# ============================================================
print("\n" + "=" * 70)
print("PASO 4: Metricas globales en set CV")
print("=" * 70)

y_true = np.array(all_y_true_cv)
y_pred = np.array(all_y_pred_cv)
acc_cv = accuracy_score(y_true, y_pred)
bal_acc_cv = balanced_accuracy_score(y_true, y_pred)

print(f"\nXGBoost Accuracy CV:          {acc_cv*100:.2f}%")
print(f"XGBoost Balanced Accuracy CV: {bal_acc_cv*100:.2f}%")

print(f"\n--- Comparacion con experimento previo (dataset completo) ---")
print(f"Sistema A (dataset 2020-2024): acc=40.00%")
print(f"Sistema B (dataset 2023-2024): acc={acc_cv*100:.2f}%")
print(f"Mejora:                       {(acc_cv-0.40)*100:+.2f} pp")

# ============================================================
# 5. PREDICCION SOBRE SET VIRGINAL (CRITICO)
# ============================================================
print("\n" + "=" * 70)
print("PASO 5: Prediccion sobre set VIRGINAL (el examen final)")
print("=" * 70)

# Entrenamos un modelo final con TODO el set CV y predecimos en virgin
class_counts_full = y_cv.value_counts()
n_total_full = len(y_cv)
n_classes_full = len(class_counts_full)
sample_weights_full = y_cv.map(
    lambda lab: n_total_full / (n_classes_full * class_counts_full[lab])
).values

final_model = XGBoostClassifier(**XGB_PARAMS)
final_model.fit(X_cv, y_cv, sample_weight=sample_weights_full)

y_virgin_pred = final_model.predict(X_virgin)
y_virgin_proba = final_model.predict_proba(X_virgin)

acc_virgin = accuracy_score(y_virgin, y_virgin_pred)
bal_acc_virgin = balanced_accuracy_score(y_virgin, y_virgin_pred)

print(f"\n*** RESULTADO CRITICO: Set virginal {X_virgin.index[0].date()} -> {X_virgin.index[-1].date()} ***")
print(f"Accuracy:          {acc_virgin*100:.2f}%")
print(f"Balanced Accuracy: {bal_acc_virgin*100:.2f}%")

print(f"\nClassification report virginal:")
print(classification_report(y_virgin, y_virgin_pred,
                             labels=[-1, 0, 1],
                             target_names=['Bajista', 'Lateral', 'Alcista'],
                             digits=3, zero_division=0))

# ============================================================
# 6. BACKTEST EN SET VIRGINAL
# ============================================================
print("=" * 70)
print("PASO 6: Backtest en set virginal")
print("=" * 70)

# Preparar datos de precios
prices_virgin = pd.DataFrame(index=X_virgin.index)
prices_virgin['close'] = df_virgin['btc_close']
prices_virgin['high'] = df_virgin['btc_high']
prices_virgin['low'] = df_virgin['btc_low']
prices_virgin['volume'] = df_virgin['btc_volume']
prices_virgin['open'] = prices_virgin['close'].shift(1).bfill()

# Construir senales
y_virgin_proba_df = pd.DataFrame(
    y_virgin_proba, index=X_virgin.index,
    columns=['proba_bajista', 'proba_lateral', 'proba_alcista']
)

# 1. Buy & Hold
signals_bh = pd.Series(1, index=prices_virgin.index)

# 2. Modelo agresivo
signals_agg = pd.Series(0, index=prices_virgin.index, dtype=float)
signals_agg[y_virgin_pred == 1] = 1

# 3. Modelo conservador (P_alcista > 0.5)
signals_cons = pd.Series(0, index=prices_virgin.index, dtype=float)
signals_cons[y_virgin_proba_df['proba_alcista'] > 0.5] = 1

config = BacktestConfig(
    initial_capital=10_000.0,
    fee_bps=10.0, slippage_bps=5.0,
    max_position_pct=0.95, allow_short=False,
    risk_free_rate=0.04,
)
bt = Backtester(config)

print("\n--- Buy & Hold ---")
r_bh = bt.run(prices_virgin, signals_bh)
print(r_bh.summary())

print("\n--- Modelo Agresivo ---")
r_agg = bt.run(prices_virgin, signals_agg)
print(r_agg.summary())

print("\n--- Modelo Conservador ---")
r_cons = bt.run(prices_virgin, signals_cons)
print(r_cons.summary())

# ============================================================
# 7. VEREDICTO HONESTO vs HIPOTESIS PREVIA
# ============================================================
print("\n" + "=" * 70)
print("PASO 7: Veredicto vs hipotesis registrada")
print("=" * 70)

print(f"""
HIPOTESIS PREVIA (registrada antes del experimento):
  - Accuracy OOS > 42% (vs 40% del sistema A)
  - Sharpe OOS > 0.3 (vs -2.86 del sistema A)
  - Max drawdown < 50% (vs 93% del sistema A)

RESULTADOS:
  - Accuracy virginal: {acc_virgin*100:.2f}%
  - Sharpe agresivo:   {r_agg.metrics['sharpe']:+.2f}
  - Sharpe conservador: {r_cons.metrics['sharpe']:+.2f}
  - MDD agresivo:      {r_agg.metrics['max_drawdown']*100:.2f}%
  - MDD conservador:   {r_cons.metrics['max_drawdown']*100:.2f}%
  - Sharpe B&H:        {r_bh.metrics['sharpe']:+.2f}

CRITERIOS:
""")

mejora_acc = acc_virgin > 0.42
mejora_sharpe = max(r_agg.metrics['sharpe'], r_cons.metrics['sharpe']) > 0.3
mejora_mdd = min(r_agg.metrics['max_drawdown'], r_cons.metrics['max_drawdown']) > -0.50

print(f"  [{'CUMPLE' if mejora_acc else 'NO'}] Accuracy > 42%")
print(f"  [{'CUMPLE' if mejora_sharpe else 'NO'}] Sharpe > 0.3")
print(f"  [{'CUMPLE' if mejora_mdd else 'NO'}] MDD > -50%")

cumplio_total = sum([mejora_acc, mejora_sharpe, mejora_mdd])

print(f"\nCriterios cumplidos: {cumplio_total}/3")

if cumplio_total >= 2:
    print("\nRESULTADO: hipotesis CONFIRMADA. El cambio de regimen explica parte del problema.")
elif cumplio_total == 1:
    print("\nRESULTADO: hipotesis PARCIALMENTE confirmada. Mejora pero insuficiente.")
else:
    print("\nRESULTADO: hipotesis REFUTADA. El problema no era el regimen.")
    print("  Esto es informacion valiosa: la dificultad es estructural del problema.")

# ============================================================
# 8. VISUALIZACION
# ============================================================
print("\n" + "=" * 70)
print("PASO 8: Generando graficos...")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))
fig.suptitle('Sistema B (2023+): Resultados en set virginal',
             fontsize=14, fontweight='bold')

# Panel 1: Equity curves del set virginal
ax = axes[0]
ax.plot(r_bh.equity.index, r_bh.equity.values,
        label=f'B&H ({r_bh.metrics["total_return"]*100:+.1f}%)',
        color='#1976D2', linewidth=2)
ax.plot(r_agg.equity.index, r_agg.equity.values,
        label=f'Agresivo ({r_agg.metrics["total_return"]*100:+.1f}%)',
        color='#F57C00', linewidth=1.5)
ax.plot(r_cons.equity.index, r_cons.equity.values,
        label=f'Conservador ({r_cons.metrics["total_return"]*100:+.1f}%)',
        color='#388E3C', linewidth=1.5)
ax.axhline(config.initial_capital, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Equity (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title(f'Set virginal: {X_virgin.index[0].date()} -> {X_virgin.index[-1].date()}')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Accuracy por fold (CV)
ax = axes[1]
fold_df = pd.DataFrame(fold_metrics)
ax.bar(fold_df['fold'], fold_df['accuracy'] * 100, color='#F57C00', alpha=0.7,
       edgecolor='black')
ax.axhline(40.4, color='red', linestyle='--', label='Baseline (40.4%)')
ax.axhline(fold_df['accuracy'].mean() * 100, color='green', linestyle=':',
           label=f'Media ({fold_df["accuracy"].mean()*100:.1f}%)')
ax.axhline(acc_virgin * 100, color='blue', linestyle='-.',
           label=f'Virginal ({acc_virgin*100:.1f}%)')
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy por fold en set CV (Sistema B)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('session5_results.png', dpi=120, bbox_inches='tight')
print("Grafico guardado: session5_results.png")
plt.show()

print("\nSesion 5 completada.")