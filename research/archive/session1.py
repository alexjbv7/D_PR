# session1.py — Sesión 1: Primera ingesta de datos
import sys
sys.path.insert(0, '.')  # Permite importar desde la carpeta actual

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data.ingestion import OHLCVIngestor, clean_ohlcv

print("=" * 60)
print("PASO 1: Conectando a Binance...")
print("=" * 60)

ingestor = OHLCVIngestor(exchange='binance', cache_dir='./cache')

print("\nPASO 2: Descargando BTC/USDT diario desde 2020...")
print("(Primera vez tarda ~30 segundos, luego usa cache local)\n")

df = ingestor.fetch_historical(
    symbol='BTC/USDT',
    timeframe='1d',
    since='2020-01-01',
    until='2024-12-31',
)

print("\nPASO 3: Limpiando datos...")
df = clean_ohlcv(df)

# ============================================================
# INSPECCIÓN — Entender qué descargamos
# ============================================================
print("\n" + "=" * 60)
print("ESTRUCTURA DEL DATAFRAME")
print("=" * 60)
print(f"\nDimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Período:     {df.index[0].date()} → {df.index[-1].date()}")
print(f"Timeframe:   1 día por fila\n")
print("Primeras 5 filas:")
print(df.head())
print("\nÚltimas 5 filas:")
print(df.tail())
print("\nEstadísticas descriptivas:")
print(df.describe().round(2))

# ============================================================
# MARCADO DE REGIMENES ANOMALOS
# ============================================================
from data.regimes import add_regime_flags, regime_summary, filter_clean_periods

print("\n" + "=" * 60)
print("DETECCION DE REGIMENES ANOMALOS")
print("=" * 60)

df = add_regime_flags(df)
print(regime_summary(df))

# Datos limpios (sin regimenes anomalos) para entrenar
df_clean = filter_clean_periods(df)
print(f"\nDataset original:        {len(df)} filas")
print(f"Dataset limpio (entrenar): {len(df_clean)} filas")
print(f"Datos descartados:       {len(df) - len(df_clean)} filas")

# ============================================================
# VALIDACIONES DE CALIDAD
# ============================================================
print("\n" + "=" * 60)
print("VALIDACIONES DE CALIDAD")
print("=" * 60)

# NaN
nans = df.isna().sum()
print(f"\nValores NaN por columna:\n{nans}")

# Gaps temporales
gaps = df.index.to_series().diff().dropna()
expected = pd.Timedelta('1 day')
anomalous = gaps[gaps > expected * 1.5]
print(f"\nGaps temporales anómalos (>1.5 días): {len(anomalous)}")
if len(anomalous) > 0:
    print(anomalous)

# Retornos diarios
df['log_ret'] = (df['close'] / df['close'].shift(1)).apply('log')
print(f"\nRetorno diario promedio: {df['log_ret'].mean()*100:.3f}%")
print(f"Volatilidad diaria:      {df['log_ret'].std()*100:.3f}%")
print(f"Volatilidad anualizada:  {df['log_ret'].std()*100*(252**0.5):.1f}%")
print(f"Retorno total del período: {(df['close'].iloc[-1]/df['close'].iloc[0]-1)*100:.1f}%")

# ============================================================
# VISUALIZACION CON REGIMENES MARCADOS
# ============================================================
print("\n" + "=" * 60)
print("GENERANDO GRAFICOS...")
print("=" * 60)

from data.regimes import KNOWN_REGIMES

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('BTC/USDT Diario - Inspeccion con regimenes marcados',
             fontsize=14, fontweight='bold')

# Funcion para sombrear regimenes
def shade_regimes(ax):
    colors_regime = {
        'binance_zero_fee_btc': '#FFC107',  # amarillo
        'ftx_collapse': '#E53935',           # rojo
        'covid_crash': '#8E24AA',            # purpura
    }
    for regime in KNOWN_REGIMES:
        start = pd.Timestamp(regime.start, tz='UTC')
        end = pd.Timestamp(regime.end, tz='UTC')
        if start >= df.index[0] and end <= df.index[-1]:
            ax.axvspan(start, end, alpha=0.2,
                       color=colors_regime.get(regime.name, 'gray'),
                       label=regime.name)

# Panel 1: Precio
ax1 = axes[0]
ax1.plot(df.index, df['close'], color='#F7931A', linewidth=1.5, zorder=10)
shade_regimes(ax1)
ax1.set_title('Precio de cierre (USD) - zonas sombreadas: regimenes anomalos')
ax1.set_ylabel('USD')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel 2: Volumen
ax2 = axes[1]
ax2.bar(df.index, df['volume'], color='#1E88E5', alpha=0.7, width=0.8, zorder=10)
shade_regimes(ax2)
ax2.set_title('Volumen diario - zona amarilla: zero-fee era (volumen inflado)')
ax2.set_ylabel('Volumen')
ax2.grid(True, alpha=0.3)

# Panel 3: Retornos
ax3 = axes[2]
colors = ['#2E7D32' if r >= 0 else '#C62828' for r in df['log_ret'].fillna(0)]
ax3.bar(df.index, df['log_ret'] * 100, color=colors, alpha=0.8, width=0.8, zorder=10)
shade_regimes(ax3)
ax3.axhline(y=0, color='black', linewidth=0.8)
ax3.set_title('Retornos logaritmicos diarios (%)')
ax3.set_ylabel('Retorno (%)')
ax3.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('btc_inspeccion_regimes.png', dpi=150, bbox_inches='tight')
print("\nGrafico guardado como 'btc_inspeccion_regimes.png'")
plt.show()