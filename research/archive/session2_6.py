# session2_6.py - Construccion del target con triple-barrier labeling
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from features.engineering import triple_barrier_labels

# ============================================================
# 1. CARGAR DATASET CON FEATURES
# ============================================================
print("=" * 60)
print("PASO 1: Cargando dataset final de sesion 2.5...")
print("=" * 60)

df = pd.read_parquet('./cache/features_session2_final.parquet')
print(f"Dataset: {df.shape[0]} filas x {df.shape[1]} columnas")

# ============================================================
# 2. CONSTRUIR TARGET CON TRIPLE-BARRIER
# ============================================================
print("\n" + "=" * 60)
print("PASO 2: Aplicando Triple-Barrier Labeling...")
print("=" * 60)

HORIZON = 20
UPPER_MULT = 2.0
LOWER_MULT = 2.0
VOL_PERIOD = 20

print(f"Parametros:")
print(f"  Horizonte:        {HORIZON} dias")
print(f"  Barrera superior: +{UPPER_MULT} x volatilidad")
print(f"  Barrera inferior: -{LOWER_MULT} x volatilidad")
print(f"  Periodo de vol:   {VOL_PERIOD} dias")

# IMPORTANTE: triple-barrier necesita los precios crudos, no las features
target = triple_barrier_labels(
    df['btc_close'],
    horizon=HORIZON,
    upper_mult=UPPER_MULT,
    lower_mult=LOWER_MULT,
    vol_period=VOL_PERIOD,
)

df['target'] = target

# ============================================================
# 3. ANALISIS DE LA DISTRIBUCION DE LABELS
# ============================================================
print("\n" + "=" * 60)
print("PASO 3: Distribucion del target")
print("=" * 60)

valid = df['target'].dropna()
counts = valid.value_counts().sort_index()
proportions = valid.value_counts(normalize=True).sort_index()

print(f"\nMuestras totales con target valido: {len(valid)}")
print(f"\nDistribucion absoluta:")
for label, count in counts.items():
    label_name = {-1: 'Bajista (-1)', 0: 'Lateral (0)', 1: 'Alcista (+1)'}[label]
    print(f"  {label_name:18s}: {int(count):5d} ({proportions[label]*100:5.1f}%)")

# Lo IDEAL en triple-barrier bien calibrado:
# - +1 y -1 deben ser similares (~30-40% cada uno)
# - 0 debe ser ~20-40%
# Si una clase domina demasiado, hay que ajustar barreras
print("\n--- Diagnostico de balance ---")
balance_ratio = max(proportions) / min(proportions)
if balance_ratio > 5:
    print(f"  ATENCION: ratio de desbalance = {balance_ratio:.1f}x")
    print(f"  Considera ajustar upper_mult/lower_mult para mejor balance")
else:
    print(f"  Balance OK: ratio = {balance_ratio:.1f}x")

# ============================================================
# 4. EXCLUIR REGIMENES ANOMALOS DEL TRAINING SET
# ============================================================
print("\n" + "=" * 60)
print("PASO 4: Excluyendo regimenes anomalos del training set")
print("=" * 60)

# Recordemos: 'in_anomalous_regime' incluye Binance zero-fee era, FTX, COVID
df_clean = df[~df['in_anomalous_regime']].copy()
df_anomalous = df[df['in_anomalous_regime']].copy()

print(f"\nDataset COMPLETO:           {len(df)} filas")
print(f"Dataset LIMPIO (entrenar):  {len(df_clean)} filas")
print(f"Dataset ANOMALO (excluido): {len(df_anomalous)} filas")

# Distribucion del target SOLO en datos limpios
clean_valid = df_clean['target'].dropna()
clean_counts = clean_valid.value_counts(normalize=True).sort_index()
print(f"\nDistribucion del target en datos LIMPIOS:")
for label, prop in clean_counts.items():
    label_name = {-1: 'Bajista (-1)', 0: 'Lateral (0)', 1: 'Alcista (+1)'}[label]
    print(f"  {label_name:18s}: {prop*100:5.1f}%")

# ============================================================
# 5. VISUALIZACION DEL TARGET
# ============================================================
print("\n" + "=" * 60)
print("PASO 5: Generando graficos...")
print("=" * 60)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Triple-Barrier Labeling sobre BTC',
             fontsize=14, fontweight='bold')

# Panel 1: Precio coloreado por label
ax1 = axes[0]
for label, color, name in [(1, '#2E7D32', 'Alcista (+1)'),
                            (0, '#9E9E9E', 'Lateral (0)'),
                            (-1, '#C62828', 'Bajista (-1)')]:
    mask = df['target'] == label
    ax1.scatter(df.index[mask], df['btc_close'][mask],
                c=color, s=3, label=name, alpha=0.7)
ax1.plot(df.index, df['btc_close'], color='black', linewidth=0.3, alpha=0.4)
ax1.set_title('Precio BTC coloreado por label del target')
ax1.set_ylabel('USD')
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax1.legend(loc='upper left', fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Distribucion temporal del target (rolling 90 dias)
ax2 = axes[1]
rolling_alcista = (df['target'] == 1).rolling(90).mean()
rolling_bajista = (df['target'] == -1).rolling(90).mean()
rolling_lateral = (df['target'] == 0).rolling(90).mean()
ax2.fill_between(df.index, 0, rolling_alcista, color='#2E7D32',
                  alpha=0.6, label='% Alcista')
ax2.fill_between(df.index, rolling_alcista, rolling_alcista + rolling_lateral,
                  color='#9E9E9E', alpha=0.6, label='% Lateral')
ax2.fill_between(df.index, rolling_alcista + rolling_lateral, 1,
                  color='#C62828', alpha=0.6, label='% Bajista')
ax2.set_title('Distribucion del target en ventana rolling (90 dias)')
ax2.set_ylabel('Proporcion')
ax2.set_ylim(0, 1)
ax2.legend(loc='upper left', fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Histograma de retornos forward por label
ax3 = axes[2]
fwd_returns = np.log(df['btc_close'].shift(-HORIZON) / df['btc_close']) * 100

for label, color, name in [(1, '#2E7D32', 'Alcista (+1)'),
                            (0, '#9E9E9E', 'Lateral (0)'),
                            (-1, '#C62828', 'Bajista (-1)')]:
    mask = df['target'] == label
    ax3.hist(fwd_returns[mask].dropna(), bins=50, alpha=0.5,
             color=color, label=f'{name} (n={mask.sum()})')
ax3.axvline(0, color='black', linewidth=0.5)
ax3.set_title(f'Retornos forward (horizon={HORIZON}d) segregados por label')
ax3.set_xlabel('Retorno % (log)')
ax3.set_ylabel('Frecuencia')
ax3.legend(loc='upper right', fontsize=9)
ax3.grid(True, alpha=0.3)

for ax in axes[:2]:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('triple_barrier_target.png', dpi=120, bbox_inches='tight')
print("  Guardado: triple_barrier_target.png")

# ============================================================
# 6. GUARDAR DATASET FINAL CON TARGET
# ============================================================
print("\n" + "=" * 60)
print("PASO 6: Guardando dataset con target...")
print("=" * 60)

df.to_parquet('./cache/dataset_with_target.parquet')
print(f"\nDataset guardado: ./cache/dataset_with_target.parquet")
print(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")

plt.show()
print("\nSesion 2.6 completada. Target listo para entrenar modelos.")