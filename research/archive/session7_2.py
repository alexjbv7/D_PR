# session7_2.py - Entrenar XGBoost con target de volatilidad
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              classification_report, confusion_matrix)

from models.zoo import XGBoostClassifier
from models.validation import WalkForwardSplitter

# ============================================================
# 1. CARGA DEL DATASET CON TARGET DE VOL
# ============================================================
print("=" * 70)
print("PASO 1: Carga del dataset con target de volatilidad")
print("=" * 70)

df = pd.read_parquet('./cache/dataset_with_vol_target.parquet')

NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume',
                    'in_anomalous_regime', 'target', 'target_direction']
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols]
y = df['target'].astype(int)

mask_valid = X.notna().all(axis=1) & y.notna()
X = X[mask_valid]
y = y[mask_valid]
df_valid = df.loc[mask_valid]

# Excluir regimenes anomalos del entrenamiento
mask_clean = ~df_valid['in_anomalous_regime']
X_clean = X[mask_clean]
y_clean = y[mask_clean]

print(f"Features: {X_clean.shape[1]}")
print(f"Muestras: {X_clean.shape[0]}")
print(f"Distribucion target:")
class_names = {0: 'Baja vol', 1: 'Media vol', 2: 'Alta vol'}
for label, prop in y_clean.value_counts(normalize=True).sort_index().items():
    print(f"  {class_names[int(label)]:12s} ({int(label)}): {prop*100:.1f}%")

# ============================================================
# 2. BASELINE NAIVE: PERSISTENCIA
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: Baseline naive (manana = hoy)")
print("=" * 70)

# Naive: predecir que el target de manana sera igual al de hoy
y_naive = y_clean.shift(1).dropna()
y_real_for_naive = y_clean.iloc[1:]

acc_naive = accuracy_score(y_real_for_naive, y_naive)
bal_acc_naive = balanced_accuracy_score(y_real_for_naive, y_naive)

print(f"\nBaseline naive (persistencia):")
print(f"  Accuracy:          {acc_naive*100:.2f}%")
print(f"  Balanced Accuracy: {bal_acc_naive*100:.2f}%")

# Esto es nuestro PISO. XGBoost debe vencerlo.

# ============================================================
# 3. WALK-FORWARD CON XGBOOST
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: Walk-forward con XGBoost (mismo pipeline)")
print("=" * 70)

splitter = WalkForwardSplitter(
    train_size=365, test_size=60, embargo=25, expanding=False
)
n_splits = splitter.get_n_splits(X_clean)
print(f"Folds: {n_splits}")

XGB_PARAMS = {
    'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.05,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'reg_alpha': 0.5, 'reg_lambda': 1.0,
    'min_child_weight': 10, 'gamma': 0.1,
    'random_state': 42, 'n_jobs': -1,
}

all_y_true = []
all_y_pred = []
all_y_proba = []
all_indices = []
fold_metrics = []
all_importances = []

for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_clean)):
    X_train, y_train = X_clean.iloc[train_idx], y_clean.iloc[train_idx]
    X_test, y_test = X_clean.iloc[test_idx], y_clean.iloc[test_idx]

    # Sample weights balanceados
    class_counts = y_train.value_counts()
    n_total = len(y_train)
    n_classes = len(class_counts)
    sample_weights = y_train.map(
        lambda lab: n_total / (n_classes * class_counts[lab])
    ).values

    model = XGBoostClassifier(**XGB_PARAMS)
    # FIX: forzar las 3 clases siempre, aunque alguna no aparezca en este fold
    model.fit(X_train, y_train, sample_weight=sample_weights, all_classes=[0, 1, 2])

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    all_y_true.extend(y_test.values)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)
    all_indices.extend(X_test.index)
    all_importances.append(model.feature_importance())

    fold_acc = accuracy_score(y_test, y_pred)
    fold_bal = balanced_accuracy_score(y_test, y_pred)
    fold_metrics.append({
        'fold': fold_idx + 1,
        'accuracy': fold_acc,
        'balanced_acc': fold_bal,
    })

    print(f"  Fold {fold_idx+1:2d}: acc={fold_acc:.3f} | bal_acc={fold_bal:.3f}")

# ============================================================
# 4. METRICAS GLOBALES
# ============================================================
print("\n" + "=" * 70)
print("PASO 4: Metricas XGBoost vs Baseline naive")
print("=" * 70)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)
acc_xgb = accuracy_score(y_true, y_pred)
bal_acc_xgb = balanced_accuracy_score(y_true, y_pred)

print(f"\n--- COMPARACION DIRECTA ---")
print(f"Baseline naive:    accuracy = {acc_naive*100:.2f}%   bal_acc = {bal_acc_naive*100:.2f}%")
print(f"XGBoost:           accuracy = {acc_xgb*100:.2f}%   bal_acc = {bal_acc_xgb*100:.2f}%")
print(f"Diferencia:        {(acc_xgb - acc_naive)*100:+.2f} pp")

if acc_xgb > acc_naive + 0.02:
    veredicto = "XGBoost VENCE al baseline naive significativamente"
elif acc_xgb > acc_naive:
    veredicto = "XGBoost vence marginalmente"
elif acc_xgb > acc_naive - 0.02:
    veredicto = "XGBoost EMPATA con naive (no agrega valor real)"
else:
    veredicto = "XGBoost PIERDE contra naive"
print(f"\nVeredicto: {veredicto}")

# ============================================================
# 5. ANALISIS CRITICO: TRANSICIONES DE REGIMEN
# ============================================================
print("\n" + "=" * 70)
print("PASO 5: ANALISIS CRITICO - performance en transiciones")
print("=" * 70)

# Identificar muestras donde el regimen CAMBIA respecto al dia anterior
y_true_series = pd.Series(y_true, index=all_indices)
y_pred_series = pd.Series(y_pred, index=all_indices)

# Una transicion: hoy es diferente a ayer
# Para esto necesitamos el regimen de "ayer" en el periodo OOS
y_true_lag = y_true_series.shift(1)
is_transition = (y_true_series != y_true_lag) & y_true_lag.notna()

print(f"\nTotal de muestras OOS: {len(y_true_series)}")
print(f"Muestras en TRANSICION (regimen cambio): {is_transition.sum()} ({is_transition.mean()*100:.1f}%)")
print(f"Muestras en PERSISTENCIA: {(~is_transition).sum()} ({(~is_transition).mean()*100:.1f}%)")

# Accuracy en transiciones vs persistencia
acc_transition_xgb = accuracy_score(
    y_true_series[is_transition], y_pred_series[is_transition]
) if is_transition.sum() > 0 else 0
acc_persistence_xgb = accuracy_score(
    y_true_series[~is_transition], y_pred_series[~is_transition]
)

# Naive en transiciones siempre falla (predice el regimen anterior)
acc_transition_naive = 0.0  # naive ALWAYS misses transitions
acc_persistence_naive = 1.0  # naive ALWAYS gets persistence right

print(f"\n--- En TRANSICIONES (donde esta el dinero) ---")
print(f"Naive:    {acc_transition_naive*100:.2f}% (siempre falla las transiciones)")
print(f"XGBoost:  {acc_transition_xgb*100:.2f}%")

print(f"\n--- En PERSISTENCIA ---")
print(f"Naive:    {acc_persistence_naive*100:.2f}% (siempre acierta)")
print(f"XGBoost:  {acc_persistence_xgb*100:.2f}%")

print(f"""
INTERPRETACION CRITICA:
- Naive captura PERSISTENCIA pero PIERDE en transiciones (donde se gana/pierde mas)
- Si XGBoost tiene >30% acc en TRANSICIONES, hay valor real anadido
- La accuracy global puede ser similar pero el ALPHA esta en las transiciones
""")

# ============================================================
# 6. MATRIZ DE CONFUSION
# ============================================================
print("=" * 70)
print("PASO 6: Matriz de confusion XGBoost")
print("=" * 70)

cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

print("\n               Predijo Baja  Predijo Media  Predijo Alta   Total")
for i, label in enumerate([0, 1, 2]):
    name = class_names[label]
    print(f"Real {name:10s}    {cm[i,0]:5d} ({cm_pct[i,0]:4.1f}%)  "
          f"  {cm[i,1]:5d} ({cm_pct[i,1]:4.1f}%)  "
          f"  {cm[i,2]:5d} ({cm_pct[i,2]:4.1f}%)  "
          f"{cm[i].sum():5d}")

# ============================================================
# 7. FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 70)
print("PASO 7: Feature importance (top 15)")
print("=" * 70)

all_imp_df = pd.concat(all_importances, axis=1)
avg_imp = all_imp_df.mean(axis=1).sort_values(ascending=False)

print(f"\nTop 15 features para predecir volatilidad:")
for i, (feat, val) in enumerate(avg_imp.head(15).items()):
    print(f"  {i+1:2d}. {feat:30s} {val:.2f}")

# ============================================================
# 8. VISUALIZACION
# ============================================================
print("\n" + "=" * 70)
print("PASO 8: Generando graficos...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('XGBoost prediciendo regimen de volatilidad',
             fontsize=14, fontweight='bold')

# Panel 1: Comparacion XGBoost vs Naive
ax = axes[0, 0]
labels_bar = ['Naive\n(persistencia)', 'XGBoost']
accs = [acc_naive*100, acc_xgb*100]
bal_accs = [bal_acc_naive*100, bal_acc_xgb*100]
x = np.arange(2)
width = 0.35
ax.bar(x - width/2, accs, width, label='Accuracy', color='#1976D2', alpha=0.7)
ax.bar(x + width/2, bal_accs, width, label='Balanced Acc', color='#F57C00', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels_bar)
ax.set_ylabel('%')
ax.set_title('Accuracy: Naive vs XGBoost')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

# Panel 2: Matriz confusion
ax = axes[0, 1]
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Baja', 'Media', 'Alta'])
ax.set_yticklabels(['Baja', 'Media', 'Alta'])
ax.set_xlabel('Prediccion')
ax.set_ylabel('Real')
ax.set_title('Matriz de confusion (% por fila)')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                ha='center', va='center',
                color='white' if cm_pct[i,j] > 50 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 3: Top 15 features
ax = axes[1, 0]
top15 = avg_imp.head(15)
ax.barh(range(len(top15)), top15.values, color='#388E3C', alpha=0.7)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Gain importance')
ax.set_title('Top 15 features para predecir vol')
ax.grid(True, alpha=0.3, axis='x')

# Panel 4: Acc en transiciones vs persistencia
ax = axes[1, 1]
labels = ['Naive', 'XGBoost']
trans_accs = [acc_transition_naive*100, acc_transition_xgb*100]
pers_accs = [acc_persistence_naive*100, acc_persistence_xgb*100]
x = np.arange(2)
ax.bar(x - width/2, trans_accs, width, label='Transiciones', color='#C62828', alpha=0.7)
ax.bar(x + width/2, pers_accs, width, label='Persistencia', color='#2E7D32', alpha=0.7)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Accuracy (%)')
ax.set_title('Donde gana cada modelo')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 105)

plt.tight_layout()
plt.savefig('volatility_xgboost.png', dpi=120, bbox_inches='tight')
print("Grafico guardado: volatility_xgboost.png")

# Guardar resultados para sesion 7.3
results_df = pd.DataFrame({
    'index': all_indices,
    'y_true': all_y_true,
    'y_pred': all_y_pred,
    'proba_low': np.array(all_y_proba)[:, 0],
    'proba_med': np.array(all_y_proba)[:, 1],
    'proba_high': np.array(all_y_proba)[:, 2],
}).set_index('index')
results_df.to_parquet('./cache/vol_predictions.parquet')
print(f"Predicciones guardadas: ./cache/vol_predictions.parquet")

plt.show()
print("\nSesion 7.2 completada.")