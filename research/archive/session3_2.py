# session3_2.py - Logistic Regression con walk-forward validation
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

from models.zoo import LogisticBaseline
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

# Solo regimen limpio
mask_clean = ~df_valid['in_anomalous_regime']
X_clean = X[mask_clean]
y_clean = y[mask_clean]

print(f"Features: {X_clean.shape[1]}")
print(f"Muestras: {X_clean.shape[0]}")
print(f"Periodo:  {X_clean.index[0].date()} -> {X_clean.index[-1].date()}")

# ============================================================
# 2. WALK-FORWARD ENTRENAMIENTO + PREDICCION
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: Entrenamiento walk-forward con Logistic Regression")
print("=" * 70)

splitter = WalkForwardSplitter(
    train_size=365, test_size=60, embargo=25, expanding=False
)
n_splits = splitter.get_n_splits(X_clean)
print(f"Folds: {n_splits}\n")

# Acumuladores OOS
all_y_true = []
all_y_pred = []
all_y_proba = []
all_indices = []

# Per-fold metrics para detectar inestabilidad
fold_metrics = []

# Para feature importance promediada
all_coefs = []

for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_clean)):
    X_train = X_clean.iloc[train_idx]
    y_train = y_clean.iloc[train_idx]
    X_test = X_clean.iloc[test_idx]
    y_test = y_clean.iloc[test_idx]

    # Entrenar
    model = LogisticBaseline(C=1.0, class_weight='balanced')
    model.fit(X_train, y_train)

    # Predecir
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Guardar
    all_y_true.extend(y_test.values)
    all_y_pred.extend(y_pred)
    all_y_proba.extend(y_proba)
    all_indices.extend(X_test.index)
    all_coefs.append(model.feature_importance().values)

    # Metricas por fold
    fold_acc = accuracy_score(y_test, y_pred)
    fold_bal = balanced_accuracy_score(y_test, y_pred)
    fold_metrics.append({
        'fold': fold_idx + 1,
        'train_start': X_train.index[0].date(),
        'test_start': X_test.index[0].date(),
        'test_end': X_test.index[-1].date(),
        'n_test': len(y_test),
        'accuracy': fold_acc,
        'balanced_acc': fold_bal,
    })

    print(f"  Fold {fold_idx+1:2d}: "
          f"test {X_test.index[0].date()} -> {X_test.index[-1].date()} | "
          f"acc={fold_acc:.3f} | bal_acc={fold_bal:.3f}")

# ============================================================
# 3. METRICAS GLOBALES OOS
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: Metricas globales OOS")
print("=" * 70)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)

acc = accuracy_score(y_true, y_pred)
bal_acc = balanced_accuracy_score(y_true, y_pred)

print(f"\nAccuracy OOS:          {acc*100:.2f}%")
print(f"Balanced Accuracy OOS: {bal_acc*100:.2f}%")
print(f"\nBaseline trivial (de sesion 3.1): 40.4%")
print(f"Mejora vs baseline:    {(acc - 0.404)*100:+.2f} puntos porcentuales")

if acc > 0.50:
    veredicto = "EXCELENTE - hay senal fuerte"
elif acc > 0.45:
    veredicto = "BUENO - senal real, sistema viable"
elif acc > 0.42:
    veredicto = "MARGINAL - hay algo pero dificil con fees"
elif acc > 0.38:
    veredicto = "SIN SENAL clara - revisar features"
else:
    veredicto = "BUG sospechoso - investigar"
print(f"Veredicto: {veredicto}")

# ============================================================
# 4. CLASSIFICATION REPORT
# ============================================================
print("\n" + "=" * 70)
print("PASO 4: Classification Report (precision/recall por clase)")
print("=" * 70)

print("\n" + classification_report(
    y_true, y_pred,
    labels=[-1, 0, 1],
    target_names=['Bajista', 'Lateral', 'Alcista'],
    digits=3,
    zero_division=0
))

# ============================================================
# 5. MATRIZ DE CONFUSION
# ============================================================
print("=" * 70)
print("PASO 5: Matriz de confusion")
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

print("""
Como leer:
- Diagonal = aciertos (mejor cuanto mas alto)
- Errores graves: confundir Alcista con Bajista (esquinas opuestas)
- Errores leves: confundir cualquier cosa con Lateral
""")

# ============================================================
# 6. ESTABILIDAD ENTRE FOLDS
# ============================================================
print("=" * 70)
print("PASO 6: Estabilidad entre folds")
print("=" * 70)

fold_df = pd.DataFrame(fold_metrics)
print(f"\nAccuracy por fold:")
print(f"  Media:   {fold_df['accuracy'].mean()*100:.2f}%")
print(f"  Std:     {fold_df['accuracy'].std()*100:.2f}%")
print(f"  Min:     {fold_df['accuracy'].min()*100:.2f}%")
print(f"  Max:     {fold_df['accuracy'].max()*100:.2f}%")

if fold_df['accuracy'].std() > 0.10:
    print(f"\n  ALERTA: alta varianza entre folds (std > 10pp)")
    print(f"  El modelo es inestable a regimenes de mercado.")
elif fold_df['accuracy'].std() > 0.05:
    print(f"\n  Varianza moderada entre folds. Aceptable.")
else:
    print(f"\n  Varianza baja entre folds. Modelo estable.")

# ============================================================
# 7. FEATURE IMPORTANCE PROMEDIADA
# ============================================================
print("\n" + "=" * 70)
print("PASO 7: Feature importance promediada (top 15)")
print("=" * 70)

avg_coefs = np.abs(np.array(all_coefs)).mean(axis=0)
fi = pd.Series(avg_coefs, index=feature_cols).sort_values(ascending=False)

print(f"\nTop 15 features mas importantes (coef abs promediado):")
for i, (feat, val) in enumerate(fi.head(15).items()):
    print(f"  {i+1:2d}. {feat:30s} {val:.4f}")

print(f"\nBottom 5 features (menos importantes):")
for feat, val in fi.tail(5).items():
    print(f"      {feat:30s} {val:.4f}")

# ============================================================
# 8. VISUALIZACIONES
# ============================================================
print("\n" + "=" * 70)
print("PASO 8: Generando graficos...")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Logistic Regression - Resultados OOS Walk-forward',
             fontsize=14, fontweight='bold')

# Panel 1: Accuracy por fold
ax = axes[0, 0]
ax.bar(fold_df['fold'], fold_df['accuracy'] * 100, color='#1976D2', alpha=0.7,
       edgecolor='black')
ax.axhline(40.4, color='red', linestyle='--', label='Baseline (40.4%)')
ax.axhline(fold_df['accuracy'].mean() * 100, color='green', linestyle=':',
           label=f'Media ({fold_df["accuracy"].mean()*100:.1f}%)')
ax.set_xlabel('Fold')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy por fold (estabilidad)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: Matriz de confusion (heatmap)
ax = axes[0, 1]
im = ax.imshow(cm_pct, cmap='Blues', vmin=0, vmax=100, aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(['Bajista', 'Lateral', 'Alcista'])
ax.set_yticklabels(['Bajista', 'Lateral', 'Alcista'])
ax.set_xlabel('Prediccion')
ax.set_ylabel('Real')
ax.set_title('Matriz de confusion (% por fila)')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{cm[i,j]}\n({cm_pct[i,j]:.1f}%)',
                ha='center', va='center',
                color='white' if cm_pct[i,j] > 50 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046)

# Panel 3: Top 15 feature importance
ax = axes[1, 0]
top15 = fi.head(15)
ax.barh(range(len(top15)), top15.values, color='#388E3C', alpha=0.7,
        edgecolor='black')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15.index, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('Coeficiente abs promedio')
ax.set_title('Top 15 features mas importantes')
ax.grid(True, alpha=0.3, axis='x')

# Panel 4: Distribucion de probabilidades predichas
ax = axes[1, 1]
all_proba_arr = np.array(all_y_proba)
ax.hist(all_proba_arr[:, 0], bins=30, alpha=0.5, label='P(Bajista)', color='#C62828')
ax.hist(all_proba_arr[:, 1], bins=30, alpha=0.5, label='P(Lateral)', color='#9E9E9E')
ax.hist(all_proba_arr[:, 2], bins=30, alpha=0.5, label='P(Alcista)', color='#2E7D32')
ax.set_xlabel('Probabilidad predicha')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribucion de probabilidades predichas')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('logistic_results.png', dpi=120, bbox_inches='tight')
print("Grafico guardado: logistic_results.png")

# ============================================================
# 9. GUARDAR RESULTADOS
# ============================================================
results_df = pd.DataFrame({
    'index': all_indices,
    'y_true': all_y_true,
    'y_pred': all_y_pred,
    'proba_bajista': all_proba_arr[:, 0],
    'proba_lateral': all_proba_arr[:, 1],
    'proba_alcista': all_proba_arr[:, 2],
}).set_index('index')

results_df.to_parquet('./cache/logistic_results.parquet')
print(f"Predicciones guardadas: ./cache/logistic_results.parquet")

plt.show()
print("\nSesion 3.2 completada.")