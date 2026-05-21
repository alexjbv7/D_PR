# session3_1.py - Baselines triviales y setup walk-forward
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CARGA Y PREPARACION DEL DATASET
# ============================================================
print("=" * 70)
print("PASO 1: Carga y preparacion del dataset")
print("=" * 70)

df = pd.read_parquet('./cache/dataset_with_target.parquet')

# Columnas que NO son features
NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume',
                    'in_anomalous_regime', 'target']

# Separar X (features) e y (target)
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols]
y = df['target']

# Quitar filas con NaN en features o target
mask_valid = X.notna().all(axis=1) & y.notna()
X = X[mask_valid]
y = y[mask_valid]
df_valid = df.loc[mask_valid]

print(f"Features: {X.shape[1]}")
print(f"Muestras totales validas: {X.shape[0]}")
print(f"Periodo: {X.index[0].date()} -> {X.index[-1].date()}")

# Excluir regimenes anomalos del entrenamiento/test
mask_clean = ~df_valid['in_anomalous_regime']
X_clean = X[mask_clean]
y_clean = y[mask_clean]

print(f"\nDespues de excluir regimenes anomalos:")
print(f"  Muestras: {X_clean.shape[0]}")
print(f"  Distribucion target:")
for label, prop in y_clean.value_counts(normalize=True).sort_index().items():
    name = {-1: 'Bajista', 0: 'Lateral', 1: 'Alcista'}[label]
    print(f"    {name:10s} ({int(label):+d}): {prop*100:.1f}%")

# ============================================================
# 2. WALK-FORWARD SPLITS
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: Walk-forward splits")
print("=" * 70)

from models.validation import WalkForwardSplitter

# Para datos diarios:
# - train_size = 365 dias (1 ano de historia)
# - test_size = 60 dias (2 meses para validar)
# - embargo = 25 dias (HORIZON + buffer, evita leakage por target)
splitter = WalkForwardSplitter(
    train_size=365,
    test_size=60,
    embargo=25,
    expanding=False,  # rolling window
)

n_splits = splitter.get_n_splits(X_clean)
print(f"Numero de folds walk-forward: {n_splits}")
print(f"Train por fold:    365 dias (~1 ano)")
print(f"Test por fold:     60 dias  (~2 meses)")
print(f"Embargo:           25 dias  (>horizonte=20 + buffer)")

# Validar visualmente los splits
print(f"\nPrimeros 3 folds:")
for i, (train_idx, test_idx) in enumerate(splitter.split(X_clean)):
    if i >= 3:
        break
    train_dates = (X_clean.index[train_idx[0]].date(), X_clean.index[train_idx[-1]].date())
    test_dates = (X_clean.index[test_idx[0]].date(), X_clean.index[test_idx[-1]].date())
    print(f"  Fold {i+1}: train {train_dates[0]} -> {train_dates[1]} | test {test_dates[0]} -> {test_dates[1]}")

# ============================================================
# 3. BASELINES TRIVIALES
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: Calculando 3 baselines triviales")
print("=" * 70)

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              precision_score, recall_score, confusion_matrix)

class_names = {-1: 'Bajista', 0: 'Lateral', 1: 'Alcista'}

# Acumuladores OOS para cada baseline
baselines_oos = {
    'random':     {'preds': [], 'y_true': []},
    'majority':   {'preds': [], 'y_true': []},
}

np.random.seed(42)

for train_idx, test_idx in splitter.split(X_clean):
    y_train = y_clean.iloc[train_idx]
    y_test = y_clean.iloc[test_idx]

    # Baseline 1: aleatorio con la distribucion del train
    train_dist = y_train.value_counts(normalize=True)
    classes = train_dist.index.values
    probs = train_dist.values
    random_preds = np.random.choice(classes, size=len(y_test), p=probs)

    # Baseline 2: clase mayoritaria
    majority_class = y_train.value_counts().index[0]
    majority_preds = np.full(len(y_test), majority_class)

    baselines_oos['random']['preds'].extend(random_preds)
    baselines_oos['random']['y_true'].extend(y_test.values)
    baselines_oos['majority']['preds'].extend(majority_preds)
    baselines_oos['majority']['y_true'].extend(y_test.values)

# ============================================================
# 4. METRICAS COMPARATIVAS
# ============================================================
print("\n--- Metricas OOS (out-of-sample, todos los folds) ---\n")

results = []
for name, data in baselines_oos.items():
    y_true = np.array(data['y_true'])
    y_pred = np.array(data['preds'])

    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec_per_class = precision_score(y_true, y_pred, labels=[-1, 0, 1],
                                      average=None, zero_division=0)
    rec_per_class = recall_score(y_true, y_pred, labels=[-1, 0, 1],
                                  average=None, zero_division=0)

    results.append({
        'modelo': name,
        'accuracy': acc,
        'balanced_accuracy': bal_acc,
        'precision_bajista': prec_per_class[0],
        'precision_lateral': prec_per_class[1],
        'precision_alcista': prec_per_class[2],
        'recall_bajista': rec_per_class[0],
        'recall_lateral': rec_per_class[1],
        'recall_alcista': rec_per_class[2],
    })

results_df = pd.DataFrame(results).set_index('modelo')
print(results_df.round(3).to_string())

# ============================================================
# 5. MATRIZ DE CONFUSION DEL BASELINE MAJORITARIO
# ============================================================
print("\n" + "=" * 70)
print("PASO 4: Matriz de confusion - baseline 'majority class'")
print("=" * 70)

cm = confusion_matrix(baselines_oos['majority']['y_true'],
                      baselines_oos['majority']['preds'],
                      labels=[-1, 0, 1])
print("\n           Predijo Bajista  Predijo Lateral  Predijo Alcista")
for i, label in enumerate([-1, 0, 1]):
    name = class_names[label]
    print(f"Real {name:8s}      {cm[i,0]:5d}            {cm[i,1]:5d}            {cm[i,2]:5d}")

# ============================================================
# 6. INTERPRETACION
# ============================================================
print("\n" + "=" * 70)
print("INTERPRETACION")
print("=" * 70)

acc_random = results_df.loc['random', 'accuracy']
acc_majority = results_df.loc['majority', 'accuracy']

print(f"""
Baseline 'predecir aleatorio':       accuracy = {acc_random*100:.1f}%
Baseline 'predecir clase mayoritaria': accuracy = {acc_majority*100:.1f}%

PISO MINIMO para cualquier modelo serio: > {max(acc_random, acc_majority)*100:.1f}% accuracy OOS.

Si Logistic Regression / XGBoost no superan {max(acc_random, acc_majority)*100:.1f}%, NO HAY SENAL.
Cualquier numero entre {min(acc_random, acc_majority)*100:.1f}% y {max(acc_random, acc_majority)*100:.1f}% es ruido.

OBJETIVO realista:
  - 'OK' = > {max(acc_random, acc_majority)*100 + 3:.0f}% accuracy OOS
  - 'Bueno' = > {max(acc_random, acc_majority)*100 + 6:.0f}% accuracy OOS
  - 'Excelente' = > {max(acc_random, acc_majority)*100 + 10:.0f}% accuracy OOS

Recuerda: cada 1% adicional de accuracy real OOS es DIFICILISIMO.
Hedge funds top funcionan con 52-55%% en problemas binarios.
""")

# Guardar resultados para comparar despues
results_df.to_csv('./cache/baselines_results.csv')
print("Resultados guardados: ./cache/baselines_results.csv")
print("\nSesion 3.1 completada. Listos para entrenar Logistic Regression.")