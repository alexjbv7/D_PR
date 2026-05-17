# session2_5.py - Anadir correlaciones macro (SP500, NASDAQ, DXY, VIX, Oro)
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data.macro import fetch_macro
from features.engineering import rolling_correlation

# ============================================================
# 1. CARGAR DATASET BASE Y DESCARGAR MACRO
# ============================================================
print("=" * 60)
print("PASO 1: Cargando dataset y descargando datos macro...")
print("=" * 60)

df = pd.read_parquet('./cache/features_session2_manual.parquet')
print(f"Dataset base: {df.shape[0]} filas x {df.shape[1]} columnas")

# Descargar macro
macro = fetch_macro(start='2020-01-01', end='2024-12-31')
print(f"\nMacro descargado: {macro.shape[0]} filas")
print(f"Activos: {list(macro.columns)}")
print(f"Periodo: {macro.index[0].date()} -> {macro.index[-1].date()}")

# ============================================================
# 2. ALINEAR INDICES
# ============================================================
print("\n" + "=" * 60)
print("PASO 2: Alineando indices BTC vs Macro...")
print("=" * 60)

# Aseguramos que ambos esten en UTC y daily
btc_idx = df.index
macro = macro.reindex(btc_idx, method='ffill')  # forward-fill weekends

print(f"BTC dias: {len(btc_idx)}")
print(f"Macro despues de alinear: {len(macro)} filas")
print(f"NaN por activo macro:")
print(macro.isna().sum())

# ============================================================
# 3. CALCULAR CORRELACIONES ROLLING (30 dias)
# ============================================================
print("\n" + "=" * 60)
print("PASO 3: Calculando correlaciones rolling (30 dias)...")
print("=" * 60)

# Retornos log de BTC y de cada activo macro
btc_returns = np.log(df['btc_close'] / df['btc_close'].shift(1))

corr_features = pd.DataFrame(index=df.index)

for asset in macro.columns:
    asset_returns = np.log(macro[asset] / macro[asset].shift(1))
    corr = rolling_correlation(btc_returns, asset_returns, window=30)
    corr_features[f'corr_btc_{asset}_30'] = corr

print(f"\nNuevas features creadas: {corr_features.shape[1]}")
print(corr_features.describe().round(3).to_string())

# ============================================================
# 4. ANADIR AL DATASET
# ============================================================
print("\n" + "=" * 60)
print("PASO 4: Anadiendo features macro al dataset...")
print("=" * 60)

df_extended = pd.concat([df, corr_features], axis=1)
print(f"Dataset final: {df_extended.shape[0]} filas x {df_extended.shape[1]} columnas")

# ============================================================
# 5. VISUALIZACION
# ============================================================
print("\n" + "=" * 60)
print("PASO 5: Generando graficos...")
print("=" * 60)

fig, axes = plt.subplots(6, 1, figsize=(14, 14), sharex=True)
fig.suptitle('Correlaciones rolling(30d) BTC vs Mercados tradicionales',
             fontsize=14, fontweight='bold')

# Panel 1: precio BTC referencia
axes[0].plot(df.index, df['btc_close'], color='#F7931A', linewidth=1)
axes[0].set_title('Precio BTC (referencia)')
axes[0].set_ylabel('USD')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
axes[0].grid(True, alpha=0.3)

# Panels 2-6: correlaciones
colors_macro = {
    'sp500': '#1976D2',
    'nasdaq': '#388E3C',
    'dxy': '#D32F2F',
    'vix': '#7B1FA2',
    'gold': '#FFB300',
}

panel_titles = {
    'sp500': 'BTC vs S&P 500 (esperamos correlacion positiva ~0.3-0.6)',
    'nasdaq': 'BTC vs NASDAQ (esperamos correlacion positiva ~0.4-0.7)',
    'dxy': 'BTC vs DXY (esperamos correlacion NEGATIVA, dolar fuerte = BTC debil)',
    'vix': 'BTC vs VIX (esperamos correlacion negativa, miedo = BTC cae)',
    'gold': 'BTC vs Oro (correlacion empiricamente debil, util verificar)',
}

for i, asset in enumerate(['sp500', 'nasdaq', 'dxy', 'vix', 'gold'], start=1):
    col = f'corr_btc_{asset}_30'
    axes[i].plot(corr_features.index, corr_features[col],
                 color=colors_macro[asset], linewidth=0.8)
    axes[i].axhline(0, color='black', linewidth=0.5)
    axes[i].axhline(0.5, color='gray', linestyle='--', alpha=0.4)
    axes[i].axhline(-0.5, color='gray', linestyle='--', alpha=0.4)
    axes[i].set_ylim(-0.8, 1.0)
    axes[i].set_title(panel_titles[asset])
    axes[i].set_ylabel('Corr')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('features_macro_correlations.png', dpi=120, bbox_inches='tight')
print("  Guardado: features_macro_correlations.png")

# ============================================================
# 6. GUARDAR DATASET FINAL
# ============================================================
print("\n" + "=" * 60)
print("PASO 6: Guardando dataset con macro...")
print("=" * 60)

df_extended.to_parquet('./cache/features_session2_final.parquet')
print(f"\nDataset final guardado: ./cache/features_session2_final.parquet")
print(f"Dimensiones: {df_extended.shape[0]} filas x {df_extended.shape[1]} columnas")

# Estadisticas de las nuevas correlaciones macro
print("\n--- Resumen de correlaciones medias ---")
for col in corr_features.columns:
    mean_corr = corr_features[col].mean()
    std_corr = corr_features[col].std()
    print(f"  {col:30s} media={mean_corr:+.3f}  std={std_corr:.3f}")

plt.show()
print("\nSesion 2.5 completada.")