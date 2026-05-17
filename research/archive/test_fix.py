# test_fix.py - verificar que el fix funciona en aislamiento
import sys
sys.path.insert(0, '.')

# Limpieza forzada de cache
import importlib
import models.zoo
importlib.reload(models.zoo)
from models.zoo import XGBoostClassifier

import numpy as np
import pandas as pd

# Dataset sintetico simple: 3 features, 3 clases
np.random.seed(42)
n = 300
X = pd.DataFrame({
    'f1': np.random.randn(n),
    'f2': np.random.randn(n),
    'f3': np.random.randn(n),
})

# Train: solo clases 0 y 1 (NO incluye clase 2)
# Test: las 3 clases
y_train = pd.Series(np.random.choice([0, 1], size=200))
y_test = pd.Series(np.random.choice([0, 1, 2], size=100))

X_train = X.iloc[:200]
X_test = X.iloc[200:]

# Test 1: SIN all_classes (deberia fallar al predecir clase 2)
print("Test 1: SIN all_classes parameter")
try:
    m1 = XGBoostClassifier(n_estimators=10, max_depth=3)
    m1.fit(X_train, y_train)
    preds1 = m1.predict(X_test)
    print(f"  Predicciones unicas: {sorted(np.unique(preds1))}")
    print(f"  inv_label_map_: {m1.inv_label_map_}")
    print("  Esto deberia mapear a [0, 1] solamente")
except Exception as e:
    print(f"  ERROR: {e}")

# Test 2: CON all_classes (deberia funcionar)
print("\nTest 2: CON all_classes=[0, 1, 2]")
try:
    m2 = XGBoostClassifier(n_estimators=10, max_depth=3)
    m2.fit(X_train, y_train, all_classes=[0, 1, 2])
    preds2 = m2.predict(X_test)
    print(f"  Predicciones unicas: {sorted(np.unique(preds2))}")
    print(f"  inv_label_map_: {m2.inv_label_map_}")
    print("  Esto deberia mapear a [0, 1, 2]")
    print("  FIX FUNCIONA!")
except Exception as e:
    print(f"  ERROR: {e}")
    print("  El fix NO esta correctamente aplicado en models/zoo.py")