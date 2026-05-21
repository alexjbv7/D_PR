# session2.py - Sesion 2: Feature Engineering completo
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data.ingestion import OHLCVIngestor, clean_ohlcv
from data.regimes import add_regime_flags, regime_summary, KNOWN_REGIMES
from features.engineering import FeatureBuilder

# ============================================================
# 1. CARGA DE DATOS (BTC ya cacheado, descarga ETH)
# ============================================================
print("=" * 60)
print("PASO 1: Cargando BTC y descargando ETH...")
print("=" * 60)

ingestor = OHLCVIngestor(exchange='binance', cache_dir='./cache')

# BTC (deberia venir de cache, instantaneo)
btc = ingestor.fetch_historical(
    symbol='BTC/USDT', timeframe='1d',
    since='2020-01-01', until='2024-12-31',
)
btc = clean_ohlcv(btc)
btc = add_regime_flags(btc)

# ETH (primera vez tarda ~30 segundos)
eth = ingestor.fetch_historical(
    symbol='ETH/USDT', timeframe='1d',
    since='2020-01-01', until='2024-12-31',
)
eth = clean_ohlcv(eth)

print(f"\nBTC: {len(btc)} filas")
print(f"ETH: {len(eth)} filas")

# Alinear ambos al mismo indice (importante)
common_idx = btc.index.intersection(eth.index)
btc = btc.loc[common_idx]
eth = eth.loc[common_idx]
print(f"Despues de alinear: {len(btc)} filas comunes")

# ============================================================
# 2. CONSTRUCCION DE FEATURES
# ============================================================
print("\n" + "=" * 60)
print("PASO 2: Construyendo features...")
print("=" * 60)

fb = FeatureBuilder()
features = fb.build(btc, df_eth=eth)

print(f"\nFeatures construidas: {features.shape[1]} columnas")
print(f"Filas totales: {features.shape[0]}")
print(f"Filas con todos los valores no-NaN: {features.dropna().shape[0]}")

# Listado de features por categoria
print("\n--- Listado de features ---")
for col in features.columns:
    n_valid = features[col].notna().sum()
    pct_valid = n_valid / len(features) * 100
    print(f"  {col:30s}  {n_valid:5d} valores ({pct_valid:5.1f}%)")

# ============================================================
# 3. ESTADISTICAS DESCRIPTIVAS
# ============================================================
print("\n" + "=" * 60)
print("PASO 3: Estadisticas descriptivas")
print("=" * 60)

stats = features.describe().T[['mean', 'std', 'min', '50%', 'max']]
stats['skew'] = features.skew()
stats['kurtosis'] = features.kurtosis()
print(stats.round(3).to_string())

# ============================================================
# 4. VISUALIZACION
# ============================================================
print("\n" + "=" * 60)
print("PASO 4: Generando graficos...")
print("=" * 60)

# Funcion auxiliar para sombrear regimenes
def shade_regimes(ax, df_index):
    colors_regime = {
        'binance_zero_fee_btc': '#FFC107',
        'ftx_collapse': '#E53935',
        'covid_crash': '#8E24AA',
    }
    for regime in KNOWN_REGIMES:
        start = pd.Timestamp(regime.start, tz='UTC')
        end = pd.Timestamp(regime.end, tz='UTC')
        if start >= df_index[0] and end <= df_index[-1]:
            ax.axvspan(start, end, alpha=0.15,
                       color=colors_regime.get(regime.name, 'gray'))

# ---------- GRAFICO 1: Features de momentum y mean-reversion ----------
fig1, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig1.suptitle('Grupo 1: Momentum y Mean-Reversion', fontsize=14, fontweight='bold')

axes[0].plot(btc.index, btc['close'], color='#F7931A', linewidth=1)
axes[0].set_title('Precio BTC (referencia)')
axes[0].set_ylabel('USD')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

axes[1].plot(features.index, features['rsi_14'], color='#1976D2', linewidth=0.8)
axes[1].axhline(70, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(30, color='green', linestyle='--', alpha=0.5)
axes[1].set_title('RSI(14) - bounded [0,100], lineas rojas=sobrecompra/sobreventa')
axes[1].set_ylabel('RSI')

axes[2].plot(features.index, features['bb_pctb_20'], color='#7B1FA2', linewidth=0.8)
axes[2].axhline(1.0, color='red', linestyle='--', alpha=0.5)
axes[2].axhline(0.0, color='green', linestyle='--', alpha=0.5)
axes[2].set_title('Bollinger %B(20) - >1 sobre banda sup, <0 bajo banda inf')
axes[2].set_ylabel('%B')

axes[3].plot(features.index, features['zscore_20'], color='#388E3C', linewidth=0.8)
axes[3].axhline(2, color='red', linestyle='--', alpha=0.5)
axes[3].axhline(-2, color='red', linestyle='--', alpha=0.5)
axes[3].axhline(0, color='black', linewidth=0.5)
axes[3].set_title('Z-score(20) - desviacion del precio respecto a su media')
axes[3].set_ylabel('Z-score')

for ax in axes:
    shade_regimes(ax, features.index)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('features_grupo1_momentum.png', dpi=120, bbox_inches='tight')
print("  Guardado: features_grupo1_momentum.png")

# ---------- GRAFICO 2: Volatilidad ----------
fig2, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig2.suptitle('Grupo 2: Volatilidad', fontsize=14, fontweight='bold')

axes[0].plot(features.index, features['atr_20'] * 100, color='#D32F2F', linewidth=0.8)
axes[0].set_title('ATR(20) normalizado por precio (%)')
axes[0].set_ylabel('%')

axes[1].plot(features.index, features['parkinson_20'] * 100, color='#F57C00', linewidth=0.8)
axes[1].set_title('Parkinson Vol(20) - estimador eficiente usando high-low')
axes[1].set_ylabel('%')

axes[2].plot(features.index, features['gk_20'] * 100, color='#5D4037', linewidth=0.8)
axes[2].set_title('Garman-Klass Vol(20) - usa OHLC completo, mas eficiente aun')
axes[2].set_ylabel('%')

for ax in axes:
    shade_regimes(ax, features.index)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('features_grupo2_volatilidad.png', dpi=120, bbox_inches='tight')
print("  Guardado: features_grupo2_volatilidad.png")

# ---------- GRAFICO 3: Volumen y flujo ----------
fig3, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
fig3.suptitle('Grupo 3: Volumen y Flujo', fontsize=14, fontweight='bold')

axes[0].plot(features.index, features['obv_zscore'], color='#1565C0', linewidth=0.8)
axes[0].axhline(0, color='black', linewidth=0.5)
axes[0].set_title('OBV z-score(50) - presion compradora/vendedora estandarizada')
axes[0].set_ylabel('z-score')

axes[1].plot(features.index, features['vwap_dev_20'] * 100, color='#00695C', linewidth=0.8)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_title('VWAP Deviation(20) - precio vs VWAP rolling (%)')
axes[1].set_ylabel('%')

axes[2].plot(features.index, features['mfi_14'], color='#C2185B', linewidth=0.8)
axes[2].axhline(80, color='red', linestyle='--', alpha=0.5)
axes[2].axhline(20, color='green', linestyle='--', alpha=0.5)
axes[2].set_title('MFI(14) - RSI ponderado por volumen')
axes[2].set_ylabel('MFI')

for ax in axes:
    shade_regimes(ax, features.index)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('features_grupo3_volumen.png', dpi=120, bbox_inches='tight')
print("  Guardado: features_grupo3_volumen.png")

# ---------- GRAFICO 4: Features avanzadas (las 3 nuevas) ----------
fig4, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig4.suptitle('Grupo 4: Features Avanzadas (las 3 nuevas)', fontsize=14, fontweight='bold')

axes[0].plot(btc.index, btc['close'], color='#F7931A', linewidth=1)
axes[0].set_title('Precio BTC (referencia)')
axes[0].set_ylabel('USD')
axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))

axes[1].plot(features.index, features['frac_diff_log_price'], color='#4527A0', linewidth=0.8)
axes[1].axhline(0, color='black', linewidth=0.5)
axes[1].set_title('Fractional Diff (d=0.4) del log-precio - estacionario PERO con memoria')
axes[1].set_ylabel('frac diff')

# Vol regime con coloreo discreto
regime_colors = {0: '#4CAF50', 1: '#FFC107', 2: '#F44336'}
for r, color in regime_colors.items():
    mask = features['vol_regime'] == r
    axes[2].scatter(features.index[mask], [r] * mask.sum(),
                    c=color, s=4, label=f'Regimen {int(r)}')
axes[2].set_title('Regimen de volatilidad: 0=baja (verde), 1=media (amarillo), 2=alta (rojo)')
axes[2].set_ylabel('Regimen')
axes[2].set_yticks([0, 1, 2])
axes[2].legend(loc='upper right', fontsize=8)

axes[3].plot(features.index, features['corr_btc_eth_30'], color='#00838F', linewidth=0.8)
axes[3].axhline(0.7, color='blue', linestyle='--', alpha=0.5, label='Alta corr (normal)')
axes[3].axhline(0.3, color='red', linestyle='--', alpha=0.5, label='Decoupling (atipico)')
axes[3].set_title('Correlacion rolling(30) BTC-ETH - decoupling suele anticipar cambios')
axes[3].set_ylabel('Correlacion')
axes[3].legend(loc='lower right', fontsize=8)
axes[3].set_ylim(-0.3, 1.05)

for ax in axes:
    shade_regimes(ax, features.index)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('features_grupo4_avanzadas.png', dpi=120, bbox_inches='tight')
print("  Guardado: features_grupo4_avanzadas.png")

plt.show()

# ============================================================
# 5. GUARDAR FEATURES PARA SESIONES POSTERIORES
# ============================================================
print("\n" + "=" * 60)
print("PASO 5: Guardando dataset...")
print("=" * 60)

# Guardamos features + precio para sesiones siguientes
output = features.copy()
output['btc_close'] = btc['close']
output['btc_high'] = btc['high']
output['btc_low'] = btc['low']
output['btc_volume'] = btc['volume']
output['in_anomalous_regime'] = btc['in_anomalous_regime']

output.to_parquet('./cache/features_session2.parquet')
print(f"\nDataset guardado: ./cache/features_session2.parquet")
print(f"Dimensiones: {output.shape[0]} filas x {output.shape[1]} columnas")

print("\nSesion 2.2 completada. Inspecciona los 4 graficos.")