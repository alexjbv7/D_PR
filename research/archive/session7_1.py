# session7_1.py - Construccion y validacion del target de volatilidad
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data.ingestion import OHLCVIngestor, clean_ohlcv
from features.volatility_target import (
    realized_volatility, future_volatility,
    vol_regime_target, vol_regime_descriptive_stats
)

# ============================================================
# 1. CARGA DE DATOS
# ============================================================
print("=" * 70)
print("PASO 1: Carga de datos BTC (de cache)")
print("=" * 70)

ingestor = OHLCVIngestor(exchange='binance', cache_dir='./cache')
btc = ingestor.fetch_historical(
    symbol='BTC/USDT', timeframe='1d',
    since='2020-01-01', until='2024-12-31',
)
btc = clean_ohlcv(btc)
print(f"BTC: {len(btc)} filas")

# ============================================================
# 2. CONSTRUIR TARGET DE VOLATILIDAD
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: Construyendo target de regimen de volatilidad")
print("=" * 70)

HORIZON = 20
QUANTILE_WINDOW = 252  # 1 ano de datos pasados para thresholds

print(f"Parametros:")
print(f"  Horizon (futuro): {HORIZON} dias")
print(f"  Ventana de quantiles (pasado): {QUANTILE_WINDOW} dias")
print(f"  N regimenes: 3 (baja/media/alta)")

vol_target = vol_regime_target(
    btc['close'], horizon=HORIZON,
    quantile_window=QUANTILE_WINDOW, n_regimes=3,
)

# Distribucion del target
print(f"\nDistribucion del target:")
counts = vol_target.dropna().value_counts(normalize=True).sort_index()
class_names = {0: 'Baja vol', 1: 'Media vol', 2: 'Alta vol'}
for label, prop in counts.items():
    name = class_names[int(label)]
    print(f"  {name:12s} ({int(label)}): {prop*100:.1f}%")

# El target deberia ser aproximadamente 33/33/33 si los thresholds son correctos
balance_ok = all(0.25 < p < 0.42 for p in counts.values)
print(f"\nBalance OK (cada clase 25-42%): {'SI' if balance_ok else 'NO - revisar'}")

# ============================================================
# 3. ESTADISTICAS DESCRIPTIVAS
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: Volatilidad futura realizada por regimen")
print("=" * 70)

stats = vol_regime_descriptive_stats(vol_target, btc['close'], horizon=HORIZON)
print(f"\nEstadisticas (% volatilidad anualizada):")
# Convertir vol diaria a anualizada multiplicando por sqrt(365)
stats_annual = stats.copy()
for col in ['mean', 'median', 'std', 'min', 'max']:
    stats_annual[col] = stats_annual[col] * np.sqrt(365) / 100  # back to fraction, then annual
print(stats_annual.round(3).to_string())

print(f"""
Interpretacion:
- Si las medias de cada clase estan claramente separadas, el target esta bien construido
- Por ejemplo, baja~30%, media~50%, alta~80%+ es ideal en crypto
""")

# ============================================================
# 4. PERSISTENCIA DEL REGIMEN (volatility clustering)
# ============================================================
print("=" * 70)
print("PASO 4: Volatility clustering - el hecho estilizado clave")
print("=" * 70)

# Si vol clustering es real, regimen de hoy deberia predecir el de manana
# con mejor que random
df_check = pd.DataFrame({
    'today': vol_target,
    'tomorrow': vol_target.shift(-1),
}).dropna()

print(f"\nMatriz de transicion (vol HOY -> vol MANANA):")
transition = pd.crosstab(df_check['today'], df_check['tomorrow'], normalize='index') * 100
transition.index = ['Hoy: Baja', 'Hoy: Media', 'Hoy: Alta']
transition.columns = ['Mng: Baja', 'Mng: Media', 'Mng: Alta']
print(transition.round(1).to_string())

# Si la diagonal domina, hay vol clustering fuerte
diagonal = np.diag(transition.values).mean()
off_diagonal = (transition.values.sum() - np.diag(transition.values).sum()) / 6
print(f"\nDiagonal promedio: {diagonal:.1f}%")
print(f"Off-diagonal promedio: {off_diagonal:.1f}%")
print(f"Ratio: {diagonal/off_diagonal:.2f}x")

if diagonal > 50:
    print("\nVOLATILITY CLUSTERING FUERTE detectado.")
    print("El regimen de hoy es buen predictor del de manana.")
elif diagonal > 40:
    print("\nVolatility clustering MODERADO.")
else:
    print("\nVolatility clustering DEBIL. Sospechoso.")

# ============================================================
# 5. VISUALIZACION
# ============================================================
print("\n" + "=" * 70)
print("PASO 5: Generando graficos...")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Target de regimen de volatilidad futura',
             fontsize=14, fontweight='bold')

# Panel 1: precio BTC + target coloreado
ax = axes[0]
ax.plot(btc.index, btc['close'], color='black', linewidth=0.5, alpha=0.5)

regime_colors = {0: '#4CAF50', 1: '#FFC107', 2: '#F44336'}
for label, color in regime_colors.items():
    mask = vol_target == label
    if mask.sum() > 0:
        ax.scatter(vol_target.index[mask], btc['close'][mask],
                   c=color, s=4, alpha=0.6,
                   label=f'{class_names[label]} ({mask.sum()})')

ax.set_ylabel('BTC ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Precio BTC + regimen de volatilidad futura predicho')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 2: volatilidad realizada (futura)
ax = axes[1]
rv_future = future_volatility(btc['close'], horizon=HORIZON)
rv_annual = rv_future * np.sqrt(365) * 100  # anualizada en %
ax.plot(rv_annual.index, rv_annual, color='#1976D2', linewidth=0.6)
ax.axhline(rv_annual.median(), color='green', linestyle='--', alpha=0.5, label=f'Mediana ({rv_annual.median():.0f}%)')
ax.set_ylabel('Vol anualizada (%)')
ax.set_title(f'Volatilidad realizada en proximos {HORIZON} dias (anualizada)')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: target en el tiempo
ax = axes[2]
for label, color in regime_colors.items():
    mask = vol_target == label
    ax.scatter(vol_target.index[mask], [label] * mask.sum(),
               c=color, s=8, alpha=0.7, label=class_names[label])
ax.set_ylabel('Regimen')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(['Baja', 'Media', 'Alta'])
ax.set_title('Distribucion temporal del target (clusters de regimen visible)')
ax.set_xlabel('Fecha')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('volatility_target.png', dpi=120, bbox_inches='tight')
print("Grafico guardado: volatility_target.png")

# ============================================================
# 6. GUARDAR DATASET
# ============================================================
print("\n" + "=" * 70)
print("PASO 6: Guardando dataset con nuevo target...")
print("=" * 70)

# Cargar el dataset existente con features
df_existing = pd.read_parquet('./cache/dataset_with_target.parquet')

# Reemplazar el target por el de volatilidad
df_new = df_existing.copy()
df_new['target_direction'] = df_existing['target']  # backup del anterior
df_new['target'] = vol_target  # nuevo target

# Eliminar filas con NaN en el nuevo target
df_new = df_new.dropna(subset=['target'])
df_new['target'] = df_new['target'].astype(int)

df_new.to_parquet('./cache/dataset_with_vol_target.parquet')
print(f"Dataset guardado: ./cache/dataset_with_vol_target.parquet")
print(f"Dimensiones: {df_new.shape[0]} filas x {df_new.shape[1]} columnas")

plt.show()
print("\nSesion 7.1 completada.")