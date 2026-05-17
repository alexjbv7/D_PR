# session2_4.py - Filtrado manual con criterio economico
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

print("=" * 60)
print("PASO 1: Cargando dataset original (sin filtrar)...")
print("=" * 60)

# Cargamos el dataset SIN filtrar (sesion 2.2)
df = pd.read_parquet('./cache/features_session2.parquet')

NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume', 'in_anomalous_regime']
features = df.drop(columns=NON_FEATURE_COLS)
print(f"Features originales: {features.shape[1]}")

# ============================================================
# 2. ELIMINACION MANUAL CON CRITERIO ECONOMICO
# ============================================================
print("\n" + "=" * 60)
print("PASO 2: Eliminacion con criterio economico")
print("=" * 60)

# Features que ELIMINAMOS (confirmadas por criterio economico)
TO_DROP_MANUAL = [
    'atr_20',       # Redundante con parkinson y GK; menos eficiente
    'gk_10',        # Redundante con gk_20; mantenemos solo periodo medio
    'gk_50',        # Redundante con gk_20; mantenemos solo periodo medio
    'log_ret_10',   # Redundante con log_ret_5; mantenemos 1, 5, 20
    'macd_line',    # Redundante con macd_hist (ya contiene la info)
    'zscore_20',    # Redundante con bb_pctb_20; BB es mas interpretable
]

# Features que MANTENEMOS pese a alta correlacion
KEPT_BY_DECISION = [
    'log_ret_1',    # Fundamental: retorno de 1 dia, base de toda finanza cuantitativa
    'rsi_14',       # Estandar de industria desde 1978 (Wilder)
    'gk_20',        # Estimador de vol mas eficiente, periodo intermedio
]

print("\nFeatures eliminadas con justificacion:")
for feat in TO_DROP_MANUAL:
    print(f"  - {feat}")

print("\nFeatures mantenidas pese a alta correlacion (criterio economico):")
for feat in KEPT_BY_DECISION:
    print(f"  + {feat}")

# Aplicar filtrado
features_filtered = features.drop(columns=TO_DROP_MANUAL)

print(f"\n--- Resumen ---")
print(f"Features originales:       {features.shape[1]}")
print(f"Features eliminadas:       {len(TO_DROP_MANUAL)}")
print(f"Features finales:          {features_filtered.shape[1]}")

# ============================================================
# 3. GUARDAR DATASET FINAL FILTRADO
# ============================================================
print("\n" + "=" * 60)
print("PASO 3: Guardando dataset filtrado manualmente...")
print("=" * 60)

output = features_filtered.copy()
for col in NON_FEATURE_COLS:
    if col in df.columns:
        output[col] = df[col]

output.to_parquet('./cache/features_session2_manual.parquet')
print(f"\nDataset guardado: ./cache/features_session2_manual.parquet")
print(f"Dimensiones finales: {output.shape[0]} filas x {output.shape[1]} columnas")

# ============================================================
# 4. LISTADO FINAL DE FEATURES POR CATEGORIA
# ============================================================
print("\n" + "=" * 60)
print("PASO 4: Features finales agrupadas por categoria")
print("=" * 60)

groups = {
    'Retornos': [c for c in features_filtered.columns if c.startswith('log_ret')],
    'Momentum': [c for c in features_filtered.columns if c.startswith('rsi') or c.startswith('macd') or c.startswith('roc')],
    'Mean-reversion': [c for c in features_filtered.columns if 'bb_' in c or 'zscore' in c],
    'Volatilidad': [c for c in features_filtered.columns if any(x in c for x in ['parkinson', 'gk_', 'atr'])],
    'Volumen/Flujo': [c for c in features_filtered.columns if any(x in c for x in ['obv', 'vwap', 'mfi', 'volume'])],
    'Microestructura': [c for c in features_filtered.columns if any(x in c for x in ['hl_range', 'oc_range', 'shadow'])],
    'Calendario': [c for c in features_filtered.columns if any(x in c for x in ['hour', 'dow', 'weekend'])],
    'Avanzadas': [c for c in features_filtered.columns if any(x in c for x in ['frac_diff', 'vol_regime', 'corr_btc_eth'])],
}

for group, cols in groups.items():
    if cols:
        print(f"\n  {group} ({len(cols)}):")
        for c in cols:
            print(f"    - {c}")

print("\nSesion 2.4 completada.")