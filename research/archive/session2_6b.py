# session2_6b.py - Diagnostico visual mejorado del triple-barrier
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Cargar dataset con target
df = pd.read_parquet('./cache/dataset_with_target.parquet')

HORIZON = 20
fwd_returns = (np.log(df['btc_close'].shift(-HORIZON) / df['btc_close']) * 100).dropna()
target = df['target'].dropna()

# Alinear
common = fwd_returns.index.intersection(target.index)
fwd_returns = fwd_returns.loc[common]
target = target.loc[common]

# Estadisticas por label
print("=" * 60)
print("DIAGNOSTICO: Retornos forward por label")
print("=" * 60)

for label in [-1, 0, 1]:
    mask = target == label
    rets = fwd_returns[mask]
    name = {-1: 'Bajista', 0: 'Lateral', 1: 'Alcista'}[label]
    print(f"\n{name} (label={label}):")
    print(f"  N muestras:  {len(rets)}")
    print(f"  Media:       {rets.mean():+.2f}%")
    print(f"  Mediana:     {rets.median():+.2f}%")
    print(f"  Std:         {rets.std():.2f}%")
    print(f"  Min:         {rets.min():+.2f}%")
    print(f"  P5:          {rets.quantile(0.05):+.2f}%")
    print(f"  P95:         {rets.quantile(0.95):+.2f}%")
    print(f"  Max:         {rets.max():+.2f}%")

# ============================================================
# VISUALIZACION CON 3 ESCALAS DIFERENTES
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Retornos forward (20d) por label — 3 vistas',
             fontsize=14, fontweight='bold')

colors = {-1: '#C62828', 0: '#9E9E9E', 1: '#2E7D32'}
names = {-1: 'Bajista (-1)', 0: 'Lateral (0)', 1: 'Alcista (+1)'}

# ---- Panel 1: Histograma normal con zoom (recortando colas) ----
ax = axes[0]
clip_min, clip_max = -40, 40  # Recortamos colas extremas para mejor visualizacion
for label in [-1, 0, 1]:
    mask = target == label
    rets = fwd_returns[mask].clip(clip_min, clip_max)
    ax.hist(rets, bins=60, alpha=0.55, color=colors[label],
            label=f'{names[label]} (n={mask.sum()})', edgecolor='black', linewidth=0.3)
ax.axvline(0, color='black', linewidth=1)
ax.set_title(f'Histograma con zoom [{clip_min}%, {clip_max}%]')
ax.set_xlabel('Retorno forward (%)')
ax.set_ylabel('Frecuencia')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Panel 2: Densidad (KDE) — mejor para comparar formas ----
ax = axes[1]
from scipy.stats import gaussian_kde
x_eval = np.linspace(-30, 30, 500)
for label in [-1, 0, 1]:
    mask = target == label
    rets = fwd_returns[mask].dropna()
    if len(rets) > 5:
        kde = gaussian_kde(rets, bw_method=0.3)
        density = kde(x_eval)
        ax.fill_between(x_eval, density, alpha=0.4, color=colors[label],
                        label=f'{names[label]}')
        ax.plot(x_eval, density, color=colors[label], linewidth=1.5)
ax.axvline(0, color='black', linewidth=1)
ax.set_title('Densidad estimada (KDE) — forma de la distribucion')
ax.set_xlabel('Retorno forward (%)')
ax.set_ylabel('Densidad')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# ---- Panel 3: Boxplot — separacion por percentiles ----
ax = axes[2]
data_box = [fwd_returns[target == lab].values for lab in [-1, 0, 1]]
labels_box = [names[lab] for lab in [-1, 0, 1]]
bp = ax.boxplot(data_box, labels=labels_box, patch_artist=True,
                widths=0.6, showfliers=False)
for patch, lab in zip(bp['boxes'], [-1, 0, 1]):
    patch.set_facecolor(colors[lab])
    patch.set_alpha(0.6)
ax.axhline(0, color='black', linewidth=1, linestyle='--')
ax.set_title('Boxplot por label (sin outliers extremos)')
ax.set_ylabel('Retorno forward (%)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('target_diagnostico.png', dpi=120, bbox_inches='tight')
print("\nGrafico guardado: target_diagnostico.png")
plt.show()