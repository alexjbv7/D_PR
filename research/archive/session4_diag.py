# session4_diag.py - Diagnostico del backtest
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ============================================================
# 1. CARGA
# ============================================================
preds = pd.read_parquet('./cache/xgboost_results.parquet')
df = pd.read_parquet('./cache/dataset_with_target.parquet')

if preds.index.tz is None:
    preds.index = preds.index.tz_localize('UTC')

common_idx = df.index.intersection(preds.index)
preds = preds.loc[common_idx]

prices = pd.DataFrame(index=common_idx)
prices['close'] = df.loc[common_idx, 'btc_close']
prices['fwd_ret_1d'] = np.log(prices['close'].shift(-1) / prices['close'])

# ============================================================
# 2. ANALISIS DE ACIERTOS POR DECISION
# ============================================================
print("=" * 70)
print("DIAGNOSTICO 1: Que pasa cuando el modelo predice cada cosa?")
print("=" * 70)

for label in [-1, 0, 1]:
    name = {-1: 'Bajista', 0: 'Lateral', 1: 'Alcista'}[label]
    mask = preds['y_pred'] == label
    if mask.sum() == 0:
        continue

    # Que fue el retorno del DIA SIGUIENTE en esos casos?
    next_day_ret = prices.loc[mask, 'fwd_ret_1d'].dropna()

    print(f"\nCuando modelo dice '{name}' ({label}):")
    print(f"  N predicciones:    {mask.sum()}")
    print(f"  Avg ret t+1:       {next_day_ret.mean()*100:+.3f}%")
    print(f"  Mediana ret t+1:   {next_day_ret.median()*100:+.3f}%")
    print(f"  Win rate t+1:      {(next_day_ret > 0).mean()*100:.1f}%")

print("""
Interpretacion:
- Si 'Alcista' tiene avg t+1 NEGATIVO -> modelo predice MAL la direccion proxima
- Si 'Bajista' tiene avg t+1 POSITIVO -> modelo predice MAL al reves
- 'Lateral' deberia tener avg t+1 cercano a 0
""")

# ============================================================
# 3. ANALISIS POR PROBABILIDAD
# ============================================================
print("=" * 70)
print("DIAGNOSTICO 2: La probabilidad alcista es informativa?")
print("=" * 70)

# Bucketizar P(alcista) en 5 quintiles
preds_with_ret = preds.copy()
preds_with_ret['fwd_ret_1d'] = prices['fwd_ret_1d']
preds_with_ret = preds_with_ret.dropna()

preds_with_ret['proba_bucket'] = pd.qcut(preds_with_ret['proba_alcista'],
                                           q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

print("\nQuintil de P(alcista) -> Retorno del dia siguiente:")
print(f"{'Bucket':10s} {'P(alcista) range':25s} {'Avg t+1':>12s} {'Win rate':>12s}")
for bucket in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
    mask = preds_with_ret['proba_bucket'] == bucket
    p_range = (preds_with_ret.loc[mask, 'proba_alcista'].min(),
               preds_with_ret.loc[mask, 'proba_alcista'].max())
    avg_ret = preds_with_ret.loc[mask, 'fwd_ret_1d'].mean()
    win_rate = (preds_with_ret.loc[mask, 'fwd_ret_1d'] > 0).mean()
    print(f"{bucket:10s} [{p_range[0]:.2f}, {p_range[1]:.2f}]            "
          f"{avg_ret*100:+8.3f}%   {win_rate*100:6.1f}%")

print("""
Interpretacion:
- Si los retornos son CRECIENTES Q1->Q5: la probabilidad SI es informativa
  (alta prob = mejores retornos siguientes)
- Si los retornos son ALEATORIOS o DECRECIENTES: el modelo no genera senal usable
""")

# ============================================================
# 4. ANALISIS POR FOLD
# ============================================================
print("\n" + "=" * 70)
print("DIAGNOSTICO 3: Performance por periodo (fold-like)")
print("=" * 70)

# Dividir el periodo OOS en bloques de 60 dias (similar a folds)
preds_with_ret['block'] = (np.arange(len(preds_with_ret)) // 60).astype(int)

for block in preds_with_ret['block'].unique():
    block_data = preds_with_ret[preds_with_ret['block'] == block]
    if len(block_data) == 0:
        continue
    start_date = block_data.index[0].date()
    end_date = block_data.index[-1].date()

    # Acc en este bloque
    acc = (block_data['y_pred'] == block_data['y_true']).mean()

    # Retorno simulado simple: long cuando P(alcista) > 0.5
    long_signal = block_data['proba_alcista'] > 0.5
    if long_signal.sum() > 0:
        simulated_return = block_data.loc[long_signal, 'fwd_ret_1d'].sum()
    else:
        simulated_return = 0
    bh_return = block_data['fwd_ret_1d'].sum()

    print(f"Block {block:2d}: {start_date} -> {end_date} | "
          f"acc={acc*100:5.1f}% | "
          f"#long_signals={long_signal.sum():3d}/{len(block_data):3d} | "
          f"sim_ret={simulated_return*100:+6.2f}% | "
          f"bh={bh_return*100:+6.2f}%")

# ============================================================
# 5. VISUALIZACION: cuando el modelo dice long vs precio
# ============================================================
print("\n" + "=" * 70)
print("Generando grafico diagnostico...")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Diagnostico: por que falla el modelo?',
             fontsize=14, fontweight='bold')

# Panel 1: Precio + senales del modelo
ax = axes[0]
ax.plot(prices.index, prices['close'], color='black', linewidth=0.8, alpha=0.6)
long_dates = preds_with_ret[preds_with_ret['proba_alcista'] > 0.5].index
short_dates = preds_with_ret[preds_with_ret['proba_bajista'] > 0.5].index
ax.scatter(long_dates, prices.loc[long_dates.intersection(prices.index), 'close'],
           color='green', s=8, alpha=0.5, label=f'P(alcista)>0.5 ({len(long_dates)})')
ax.scatter(short_dates, prices.loc[short_dates.intersection(prices.index), 'close'],
           color='red', s=8, alpha=0.5, label=f'P(bajista)>0.5 ({len(short_dates)})')
ax.set_ylabel('BTC ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Senales del modelo vs precio')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 2: P(alcista) en el tiempo
ax = axes[1]
ax.plot(preds_with_ret.index, preds_with_ret['proba_alcista'],
        color='#388E3C', linewidth=0.8)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
ax.axhline(0.33, color='gray', linestyle=':', alpha=0.5, label='Equiprobable')
ax.set_ylabel('P(alcista)')
ax.set_title('Probabilidad alcista del modelo en el tiempo')
ax.legend()
ax.grid(True, alpha=0.3)

# Panel 3: Equity simulada simple (sin fees, solo para diagnosticar)
ax = axes[2]
# Estrategia simple: 1 si P(alcista)>0.5, 0 si no
position = (preds_with_ret['proba_alcista'] > 0.5).astype(float)
strategy_returns = position.shift(1).fillna(0) * preds_with_ret['fwd_ret_1d']
equity_strategy = (1 + strategy_returns).cumprod()
equity_bh = (1 + preds_with_ret['fwd_ret_1d']).cumprod()

ax.plot(equity_strategy.index, equity_strategy * 10000, color='#F57C00',
        linewidth=2, label='Estrategia (sin fees)')
ax.plot(equity_bh.index, equity_bh * 10000, color='#1976D2',
        linewidth=2, label='Buy & Hold')
ax.axhline(10000, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Equity ($)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Equity simulada SIN fees ni slippage (puro modelo)')
ax.legend()
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('backtest_diagnostico.png', dpi=120, bbox_inches='tight')
print("Grafico guardado: backtest_diagnostico.png")
plt.show()

print("\nDiagnostico completado.")