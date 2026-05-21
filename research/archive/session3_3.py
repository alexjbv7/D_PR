# session3_3.py - XGBoost con walk-forward validation
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              precision_score, recall_score, confusion_matrix,
                              classification_report)

from models.zoo import XGBoostClassifier
from models.validation import WalkForwardSplitter

# ============================================================
# 1. CARGA Y PREPARACION
# ============================================================
print("=" * 70)
print("PASO 1: Carga del dataset")
print("=" * 70)

df = pd.read_parquet('./cache/dataset_with_target.parquet')

NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume',
                    'in_anomalous_regime', 'target']
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols]
y = df['target']

mask_valid = X.notna().all(axis=1) & y.notna()
X = X[mask_valid]
y = y[mask_valid]
df_valid = df.loc[mask_valid]

mask_clean = ~df_valid['in_anomalous_regime']
X_clean = X[mask_clean]
y_clean = y[mask_clean]

print(f"Features: {X_clean.shape[1]}")
print(f"Muestras: {X_clean.shape[0]}")

# ============================================================
# 2. PARAMETROS XGBOOST CONSERVADORES
# ============================================================
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'min_child_weight': 10,
    'gamma': 0.1,
    'random_state': 42,
    'n_jobs': -1,
}

print("\n" + "=" * 70)
print("PASO 2: Hiperparametros XGBoost (conservadores anti-overfitting)")
print("=" * 70)
for k, v in XGB_PARAMS.items():
    print(f"  {k:25s}: {v}")

# ============================================================
# 3. WALK-FORWARD ENTRENAMIENTO
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: Walk-forward training")
print("=" * 70)

splitter = WalkForwardSplitter(
    train_size=365, test_size=60, embargo=25, expanding=False
)
n_splits = splitter.get_n_splits(X_clean)
print(f"Folds: {n_splits}\n")

all_y_true = []
all_y_pred = []
all_y_proba = []
all_indices = []
fold_metrics = []
all_importances = []

# Sample weights: ponderar inversamente por frecuencia para balanceo
class_weights = {-1: 1.0, 0: 1.0, 1: 1.0}  # se calculan dinamicamente per fold

for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_clean)):
    X_train = X_clean.iloc[train_idx]
    y_train = y_clean.iloc[train_idx]
    X_test = X_clean.iloc[test_idx]
    y_test = y_clean.iloc[test_idx]

    # Calcular sample_weights para balancear clases (similar a class_weight='balanced')
    class_counts = y_train.value_counts()
    n_total = len(y_train)
    n_classes = len(class_counts)
    sample_weights = y_train.map(
        lambda lab: n_total / (n_classes * class_counts[lab])
    ).values

    # Entrenar
    model = XGBoostClassifier(**XGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights)

    # Predecir
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Guardar
    all_y_true.extend(y_test.values)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)
    all_indices.extend(X_test.index)
    all_importances.append(model.feature_importance())

    # Metricas por fold
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
# 4. METRICAS GLOBALES OOS
# ============================================================
print("\n" + "=" * 70)
print("PASO 4: Metricas globales OOS")
print("=" * 70)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)

acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)

print(f"\nXGBoost Accuracy OOS:          {acc*100:.2f}%")
print(f"XGBoost Balanced Accuracy OOS: {bal_acc*100:.2f}%")

# Comparacion con Logistic
print(f"\n--- Comparacion con Logistic Regression ---")
print(f"Logistic acc:    36.67%   (sesion 3.2)")
print(f"XGBoost acc:     {acc*100:.2f}%")
print(f"Diferencia:      {(acc - 0.3667)*100:+.2f} pp")

print(f"\nBaseline trivial (sesion 3.1): 40.4%")
print(f"XGBoost vs baseline: {(acc - 0.404)*100:+.2f} pp")

if acc > 0.50:
    veredicto = "EXCELENTE - hay senal fuerte y no-lineal"
elif acc > 0.45:
    veredicto = "BUENO - senal real con XGBoost"
elif acc > 0.42:
    veredicto = "MARGINAL - alguna senal pero dificil"
elif acc > 0.38:
    veredicto = "SIN SENAL clara - mismo problema que Logistic"
else:
    veredicto = "Modelo no aprende - revisar features"
print(f"\nVeredicto: {veredicto}")

# ============================================================
# 5. CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("PASO 5: Classification Report")
print("=" * 70)

print("\n" + classification_report(
    y_true, y_pred,
    labels=[-1, 0, 1],
    target_names=['Bajista', 'Lateral', 'Alcista'],
    digits=3,
    zero_division=0
))

# ============================================================
# 6. MATRIZ DE CONFUSION
# ============================================================
print("=" * 70)
print("PASO 6: Matriz de confusion")
print("=" * 70)

cm = confusion_matrix(y_true, y_pred, labels=[-1, 0, 1])
n_per_class = cm.sum(axis=1, keepdims=True)
cm_pct = cm / n_per_class * 100

print("\n           Predijo Bajista  Predijo Lateral  Predijo Alcista   Total")
for i, label in enumerate([-1, 0, 1]):
    name = {-1: 'Bajista', 0: 'Lateral', 1: 'Alcista'}[label]
    print(f"Real {name:8s}      {cm[i,0]:5d} ({cm_pct[i,0]:4.1f}%)  "
          f"  {cm[i,1]:5d} ({cm_pct[i,1]:4.1f}%)  "
          f"  {cm[i,2]:5d} ({cm_pct[i,2]:4.1f}%)  "
          f"{cm[i].sum():5d}")

# ============================================================
# 7. ESTABILIDAD ENTRE FOLDS
# ============================================================
print("\n" + "=" * 70)
print("PASO 7: Estabilidad entre folds (XGBoost vs Logistic)")
print("=" * 70)

fold_df = pd.DataFrame(fold_metrics)
print(f"\nXGBoost - Accuracy por fold:")
print(f"  Media:   {fold_df['accuracy'].mean()*100:.2f}%")
print(f"  Std:     {fold_df['accuracy'].std()*100:.2f}%")
print(f"  Min:     {fold_df['accuracy'].min()*100:.2f}%")
print(f"  Max:     {fold_df['accuracy'].max()*100:.2f}%")

print(f"\nLogistic (de sesion 3.2):")
print(f"  Media:   36.67%")
print(f"  Std:     16.87%")

# ============================================================
# 8. FEATURE IMPORTANCE PROMEDIADA
# ============================================================
print("\n" + "=" * 70)
print("PASO 8: Feature importance XGBoost (top 15)")
print("=" * 70)

# Promediar importancias entre folds
all_imp_df = pd.concat(all_importances, axis=1)
avg_imp = all_imp_df.mean(axis=1).sort_values(ascending=False)

print(f"\nTop 15 features (gain importance, promedio entre folds):")
for i, (feat, val) in enumerate(avg_imp.head(15).items()):
    print(f"  {i+1:2d}. {feat:30s} {val:.2f}")

print(f"\nBottom 5 features:")
for feat, val in avg_imp.tail(5).items():
    print(f"      {feat:30s} {val:.2f}")

# ============================================================
# 9. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 70)
print("PASO 9: Generando graficos...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('XGBoost - Resultados OOS Walk-forward',
             fontsize=14, fontweight='bold')

# Panel 1: Accuracy por fold (XGBoost vs Logistic)
ax = axes[0, 0]
folds_x = np.arange(1, len(fold_df) + 1)
width = 0.35

# Cargar logistic results para comparar
import os
logistic_accs_per_fold = []
if os.path.exists('./cache/logistic_results.parquet'):
    log_res = pd.read_parquet('./cache/logistic_results.parquet')
    # Calcular accuracy por fold para logistic
    for _, fold_row in fold_df.iterrows():
        fold_dates = (pd.Timestamp(fold_row['test_start'], tz='UTC'),
                      pd.Timestamp(fold_row['test_end'], tz='UTC'))
        mask_fold = (log_res.index >= fold_dates[0]) & (log_res.index <= fold_dates[1])
        if mask_fold.sum() > 0:
            log_acc = (log_res.loc[mask_fold, 'y_true'] ==
                       log_res.loc[mask_fold, 'y_pred']).mean()
            logistic_accs_per_fold.append(log_acc)
        else:
            logistic_accs_per_fold.append(np.nan)

if logistic_accs_per_fold:
    ax.bar(folds_x - width/2, np.array(logistic_accs_per_fold)*100, width,
           color='#1976D2', alpha=0.7, label='Logistic', edgecolor='black')
    ax.bar(folds_x + width/2, fold_df['accuracy']*100, width,
           color='#F57C00', alpha=0.7, label='XGBoost', edgecolor='black')
else:
    ax.bar(folds_x, fold_df['accuracy']*100, color='#F57C00', alpha=0.7,
           label='XGBoost', edgecolor='black')

ax.axhline(40.4, color='red', linestyle='--', linewidth=2, label='Baseline (40.4%)')
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy por fold')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Matriz de confusion
ax = axes[0, 1]
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Bajista', 'Lateral', 'Alcista'])
ax.set_yticklabels(['Bajista', 'Lateral', 'Alcista'])
ax.set_xlabel('Prediccion')
ax.set_ylabel('Real')
ax.set_title('Matriz de confusion XGBoost (% por fila)')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                ha='center', va='center',
                color='white' if cm_pct[i,j] > 50 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 3: Top 15 feature importance
ax = axes[1, 0]
top15 = avg_imp.head(15)
ax.barh(range(len(top15)), top15.values, color='#388E3C', alpha=0.7,
        edgecolor='black')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Gain importance')
ax.set_title('Top 15 features (XGBoost)')
ax.grid(True, alpha=0.3, axis='x')

# Panel 4: Distribucion de probabilidades
ax = axes[1, 1]
all_proba_arr = np.array(all_y_proba)
ax.hist(all_proba_arr[:, 0], bins=30, alpha=0.5, label='P(Bajista)', color='#C62828')
if all_proba_arr.shape[1] >= 3:
    ax.hist(all_proba_arr[:, 1], bins=30, alpha=0.5, label='P(Lateral)', color='#9E9E9E')
    ax.hist(all_proba_arr[:, 2], bins=30, alpha=0.5, label='P(Alcista)', color='#2E7D32')
ax.set_xlabel('Probabilidad predicha')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribucion de probabilidades XGBoost')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('xgboost_results.png', dpi=120, bbox_inches='tight')
print("Grafico guardado: xgboost_results.png")

# ============================================================
# 10. GUARDAR RESULTADOS
# ============================================================
results_df = pd.DataFrame({
    'index': all_indices,
    'y_true': all_y_true,
    'y_pred': all_y_pred,
    'proba_bajista': all_proba_arr[:, 0] if all_proba_arr.shape[1] >= 1 else np.nan,
    'proba_lateral': all_proba_arr[:, 1] if all_proba_arr.shape[1] >= 2 else np.nan,
    'proba_alcista': all_proba_arr[:, 2] if all_proba_arr.shape[1] >= 3 else np.nan,
}).set_index('index')

results_df.to_parquet('./cache/xgboost_results.parquet')
print(f"Predicciones guardadas: ./cache/xgboost_results.parquet")

plt.show()
print("\nSesion 3.3 completada.")
