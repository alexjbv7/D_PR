# session2_3.py - Analisis de correlacion entre features
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. CARGA DEL DATASET DE SESION 2.2
# ============================================================
print("=" * 60)
print("PASO 1: Cargando dataset de sesion 2.2...")
print("=" * 60)

df = pd.read_parquet('./cache/features_session2.parquet')

# Separamos features (excluimos columnas de precio crudo y bandera)
NON_FEATURE_COLS = ['btc_close', 'btc_high', 'btc_low', 'btc_volume', 'in_anomalous_regime']
features = df.drop(columns=NON_FEATURE_COLS)

# Eliminamos filas con NaN para correlacion limpia
features_clean = features.dropna()

print(f"\nFeatures totales: {features.shape[1]}")
print(f"Filas limpias (sin NaN): {features_clean.shape[0]}")

# ============================================================
# 2. MATRICES DE CORRELACION
# ============================================================
print("\n" + "=" * 60)
print("PASO 2: Calculando matrices de correlacion...")
print("=" * 60)

corr_pearson = features_clean.corr(method='pearson')
corr_spearman = features_clean.corr(method='spearman')

# ============================================================
# 3. HEATMAP DE CORRELACION
# ============================================================
print("\nPASO 3: Generando heatmap...")

fig, axes = plt.subplots(1, 2, figsize=(22, 9))
fig.suptitle('Matrices de correlacion entre features', fontsize=14, fontweight='bold')

for ax, corr, name in [(axes[0], corr_pearson, 'Pearson (lineal)'),
                        (axes[1], corr_spearman, 'Spearman (rango)')]:
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    ax.set_title(name)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('features_correlation_heatmap.png', dpi=120, bbox_inches='tight')
print("  Guardado: features_correlation_heatmap.png")

# ============================================================
# 4. DETECTAR PARES ALTAMENTE CORRELACIONADOS
# ============================================================
print("\n" + "=" * 60)
print("PASO 4: Pares de features con alta correlacion")
print("=" * 60)

# Tomar el triangulo superior (sin diagonal) para evitar duplicados
upper = corr_pearson.where(
    np.triu(np.ones(corr_pearson.shape), k=1).astype(bool)
)

# Convertir a serie y ordenar por correlacion absoluta
pairs = (
    upper.stack()
    .reset_index()
    .rename(columns={'level_0': 'feature_a', 'level_1': 'feature_b', 0: 'corr'})
)
pairs['abs_corr'] = pairs['corr'].abs()
pairs = pairs.sort_values('abs_corr', ascending=False)

# Mostrar top 20
print("\nTop 20 pares con mayor correlacion absoluta:")
print(pairs.head(20).to_string(index=False))

# ============================================================
# 5. RECOMENDACIONES AUTOMATICAS DE ELIMINACION
# ============================================================
print("\n" + "=" * 60)
print("PASO 5: Recomendaciones de eliminacion (umbral > 0.95)")
print("=" * 60)

# Algoritmo greedy: si dos features tienen corr > 0.95,
# eliminar la que tenga MAS correlacion promedio con el resto
THRESHOLD = 0.95

to_drop = set()
critical_pairs = pairs[pairs['abs_corr'] > THRESHOLD]

print(f"\nPares con |correlacion| > {THRESHOLD}: {len(critical_pairs)}")

if len(critical_pairs) > 0:
    print("\nPares criticos (recomendamos eliminar uno de cada par):")
    for _, row in critical_pairs.iterrows():
        a, b = row['feature_a'], row['feature_b']
        if a in to_drop or b in to_drop:
            continue
        # Decision: eliminar el que tenga mayor correlacion media con el resto
        avg_a = corr_pearson[a].abs().mean()
        avg_b = corr_pearson[b].abs().mean()
        drop_candidate = a if avg_a > avg_b else b
        keep_candidate = b if drop_candidate == a else a
        to_drop.add(drop_candidate)
        print(f"  {a:25s} <-> {b:25s} (corr={row['corr']:+.3f})")
        print(f"      mantener: {keep_candidate}, eliminar: {drop_candidate}")

print(f"\n--- Resumen ---")
print(f"Features originales:    {features.shape[1]}")
print(f"Features a eliminar:    {len(to_drop)}")
print(f"Features finales:       {features.shape[1] - len(to_drop)}")

if to_drop:
    print(f"\nFeatures a eliminar: {sorted(to_drop)}")

# ============================================================
# 6. ZONA DE PRECAUCION (0.85 - 0.95)
# ============================================================
print("\n" + "=" * 60)
print("PASO 6: Zona de precaucion (0.85 - 0.95)")
print("=" * 60)

caution = pairs[(pairs['abs_corr'] >= 0.85) & (pairs['abs_corr'] <= 0.95)]
print(f"\nPares en zona de precaucion: {len(caution)}")
if len(caution) > 0:
    print("\nEstos NO se eliminan automaticamente, pero conviene revisar:")
    print(caution.head(15).to_string(index=False))

# ============================================================
# 7. GUARDAR DATASET LIMPIO
# ============================================================
print("\n" + "=" * 60)
print("PASO 7: Guardando dataset con features filtradas...")
print("=" * 60)

features_filtered = features.drop(columns=list(to_drop))

# Reanadir columnas no-feature
output = features_filtered.copy()
for col in NON_FEATURE_COLS:
    if col in df.columns:
        output[col] = df[col]

output.to_parquet('./cache/features_session2_filtered.parquet')
print(f"\nDataset filtrado guardado: ./cache/features_session2_filtered.parquet")
print(f"Dimensiones: {output.shape[0]} filas x {output.shape[1]} columnas")

plt.show()
print("\nSesion 2.3 completada.")