# session7_2_diag.py - Diagnostico del fracaso de XGBoost en vol prediction
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# ============================================================
# TEST 1: Verificar alineamiento del target
# ============================================================
print("=" * 70)
print("TEST 1: Verificar que target[t] mira el futuro correcto")
print("=" * 70)

df = pd.read_parquet('./cache/dataset_with_vol_target.parquet')
print(f"Columnas en dataset: {list(df.columns)}")
print(f"Forma: {df.shape}")

# El target deberia ser el regimen de vol de los proximos 20 dias
# Verificacion: para una fila random, comprobar que el target coincide
# con la vol futura de los proximos 20 dias

print("\nVerificando alineacion del target...")
# Calcular vol futura manualmente para confirmar
import numpy as np
log_ret = np.log(df['btc_close'] / df['btc_close'].shift(1))

# Para una fila random
import random
sample_indices = random.sample(range(252, len(df)-30), 5)
print(f"\nPara 5 muestras aleatorias:")
print(f"{'Fecha':12s} {'Target':>8s} {'Vol futura 20d':>15s} {'Quantile pasado':>20s}")

for idx in sample_indices:
    fecha = df.index[idx].date()
    target_val = df['target'].iloc[idx]
    
    # Vol futura: dias t+1 a t+20
    future_returns = log_ret.iloc[idx+1:idx+21]
    future_vol = future_returns.std() * np.sqrt(365)  # anualizada
    
    # Vol pasada para thresholds
    past_252 = log_ret.iloc[idx-252:idx].rolling(20).std().dropna()
    q33 = past_252.quantile(0.33) * np.sqrt(365)
    q67 = past_252.quantile(0.67) * np.sqrt(365)
    
    expected_class = 0 if future_vol < q33 else (2 if future_vol > q67 else 1)
    
    print(f"{str(fecha):12s} {int(target_val):>8d} "
          f"{future_vol:>14.3f}  thresholds=[{q33:.3f}, {q67:.3f}] -> esperado={expected_class}")

print("\nSi 'Target' coincide con 'esperado', el target esta bien alineado.")
print("Si NO coinciden, hay un bug en la construccion del target.")

# ============================================================
# TEST 2: Verificar que features no contienen el target
# ============================================================
print("\n" + "=" * 70)
print("TEST 2: Verificar leakage del target en features")
print("=" * 70)

NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume',
                    'in_anomalous_regime', 'target', 'target_direction']
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
features = df[feature_cols]

# Calcular correlacion de cada feature con el target
target_for_corr = df['target'].astype(float)
correlations = {}
for col in feature_cols:
    if features[col].notna().sum() > 100:
        corr = features[col].corr(target_for_corr)
        correlations[col] = corr

corr_series = pd.Series(correlations).abs().sort_values(ascending=False)
print("\nTop 10 features con mayor correlacion (abs) con target:")
for feat, val in corr_series.head(10).items():
    print(f"  {feat:30s} corr_abs = {val:.3f}")

print("\n  Si alguna feature tiene corr > 0.7, hay LEAKAGE potencial.")
print("  Si la feature 'vol_regime' tiene corr alta, es leakage CONFIRMADO.")

# ============================================================
# TEST 3: Test trivial - entrenar con TODO el dataset (in-sample)
# ============================================================
print("\n" + "=" * 70)
print("TEST 3: XGBoost in-sample (overfitting test)")
print("=" * 70)

from sklearn.metrics import accuracy_score
from models.zoo import XGBoostClassifier

X = df[feature_cols]
y = df['target'].astype(int)
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# Entrenar in-sample (con los mismos datos que evaluamos)
# Si esto no da >80% accuracy, hay un problema fundamental
model = XGBoostClassifier(
    n_estimators=100, max_depth=4, learning_rate=0.1,
    subsample=1.0, colsample_bytree=1.0,
    random_state=42, n_jobs=-1
)
model.fit(X, y)
y_pred_insample = model.predict(X)
acc_insample = accuracy_score(y, y_pred_insample)

print(f"\nXGBoost in-sample accuracy: {acc_insample*100:.2f}%")
print(f"\nInterpretacion:")
if acc_insample > 0.85:
    print(f"  > 85%: el modelo PUEDE aprender el target. El bug esta en walk-forward.")
elif acc_insample > 0.50:
    print(f"  Modelo aprende parcialmente. Hay senial pero ruidosa.")
else:
    print(f"  < 50% in-sample: HAY UN PROBLEMA TECNICO. Las features no permiten")
    print(f"  separar las clases ni con todos los datos.")

# ============================================================
# TEST 4: Predecir con regla naive en walk-forward (comparacion)
# ============================================================
print("\n" + "=" * 70)
print("TEST 4: Predecir naive en EXACTAMENTE las mismas particiones del walk-forward")
print("=" * 70)

from models.validation import WalkForwardSplitter

# Mismo split que sesion 7.2
splitter = WalkForwardSplitter(
    train_size=365, test_size=60, embargo=25, expanding=False
)

# Datos limpios
df_valid = df.loc[mask]
df_clean_mask = ~df_valid['in_anomalous_regime']
X_clean = X[df_clean_mask]
y_clean = y[df_clean_mask]

all_y_true_naive = []
all_y_pred_naive = []

# Necesitamos el regimen de "ayer" para naive
y_yesterday = y_clean.shift(1)

for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X_clean)):
    y_test = y_clean.iloc[test_idx].values
    y_naive_test = y_yesterday.iloc[test_idx].values  # naive = ayer
    
    valid_mask = ~np.isnan(y_naive_test)
    if valid_mask.sum() == 0:
        continue
    
    all_y_true_naive.extend(y_test[valid_mask])
    all_y_pred_naive.extend(y_naive_test[valid_mask].astype(int))

acc_naive_wf = accuracy_score(all_y_true_naive, all_y_pred_naive)
print(f"\nNaive en MISMAS particiones walk-forward: {acc_naive_wf*100:.2f}%")
print(f"\nSi naive da ~90% en las mismas particiones, confirma que el bug")
print(f"esta en COMO se entrena/predice con XGBoost, no en los datos.")

print("\n" + "=" * 70)
print("CONCLUSION DEL DIAGNOSTICO")
print("=" * 70)
print(f"""
RESUMEN:
  - In-sample XGBoost: {acc_insample*100:.2f}%
  - Naive (mismas particiones): {acc_naive_wf*100:.2f}%
  - XGBoost OOS (sesion 7.2): 28.33%

DIAGNOSIS:
""")

if acc_insample > 0.85 and acc_naive_wf > 0.85:
    print("  XGBoost SI puede aprender el target.")
    print("  El naive SI logra 90% en walk-forward.")
    print("  -> El BUG esta en como XGBoost maneja walk-forward (mapeo de clases?)")
elif acc_insample < 0.50:
    print("  XGBoost NO puede aprender el target ni con los mismos datos.")
    print("  -> Bug en construccion del target o features")
else:
    print("  Resultados mixtos. Investigar mas.")