# verify_run2.py - test minimalista
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

print("Step 1: cargando dataset...")
df = pd.read_parquet('./cache/dataset_with_vol_target.parquet')
print(f"  OK shape={df.shape}")

NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume',
                    'in_anomalous_regime', 'target', 'target_direction']
feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
X = df[feature_cols]
y = df['target'].astype(int)

mask_valid = X.notna().all(axis=1) & y.notna()
X = X[mask_valid]
y = y[mask_valid]
df_valid = df.loc[mask_valid]
mask_clean = ~df_valid['in_anomalous_regime']
X_clean = X[mask_clean]
y_clean = y[mask_clean]
print(f"  X_clean shape: {X_clean.shape}")

print("\nStep 2: importando WalkForwardSplitter...")
from models.validation import WalkForwardSplitter
print("  OK")

print("\nStep 3: creando splitter...")
splitter = WalkForwardSplitter(train_size=365, test_size=60, embargo=25, expanding=False)
print("  OK")

print("\nStep 4: obteniendo primer fold...")
gen = splitter.split(X_clean)
train_idx, test_idx = next(gen)
print(f"  OK train_idx[:5]={train_idx[:5]}, test_idx[:5]={test_idx[:5]}")

X_train = X_clean.iloc[train_idx]
y_train = y_clean.iloc[train_idx]
X_test = X_clean.iloc[test_idx]
y_test = y_clean.iloc[test_idx]

print(f"\nStep 5: train classes: {sorted(y_train.unique())}")
print(f"Step 5: test classes: {sorted(y_test.unique())}")

print("\nStep 6: importando XGBoostClassifier...")
from models.zoo import XGBoostClassifier
print("  OK")

print("\nStep 7: creando modelo...")
model = XGBoostClassifier(
    n_estimators=50, max_depth=3, learning_rate=0.1,
    random_state=42, n_jobs=-1
)
print("  OK")

print("\nStep 8: entrenando...")
model.fit(X_train, y_train, all_classes=[0, 1, 2])
print(f"  OK inv_label_map_={model.inv_label_map_}")

print("\nStep 9: prediciendo...")
y_pred = model.predict(X_test)
print(f"  OK pred shape={y_pred.shape}, unique={sorted(np.unique(y_pred))}")

from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f"\nACCURACY FOLD 1: {acc*100:.2f}%")
print(f"\nDistribucion predicha: {pd.Series(y_pred).value_counts().sort_index().to_dict()}")
print(f"Distribucion real:    {y_test.value_counts().sort_index().to_dict()}")