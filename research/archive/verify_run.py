# verify_run.py - verificar que XGBoost se entrena CORRECTAMENTE en walk-forward
import sys
sys.path.insert(0, '.')

import importlib
import models.zoo
importlib.reload(models.zoo)
from models.zoo import XGBoostClassifier

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from models.validation import WalkForwardSplitter

# Carga
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
mask_clean = ~df_valid['in_anomalous_regime']
X_clean = X[mask_clean]
y_clean = y[mask_clean]

print(f"X_clean shape: {X_clean.shape}")
print(f"y_clean classes: {sorted(y_clean.unique())}")

# Test directo: tomar el PRIMER fold y entrenar
splitter = WalkForwardSplitter(train_size=365, test_size=60, embargo=25, expanding=False)