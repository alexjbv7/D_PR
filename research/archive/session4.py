# session4.py - Backtest realista de las predicciones XGBoost
import sys
sys.path.insert(0, '.')

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from backtesting.engine import Backtester, BacktestConfig
from risk.management import IntegratedRiskManager

# ============================================================
# 1. CARGA DATOS Y PREDICCIONES
# ============================================================
print("=" * 70)
print("PASO 1: Cargando datos y predicciones XGBoost")
print("=" * 70)

# Predicciones OOS de XGBoost (sesion 3.3)
preds = pd.read_parquet('./cache/xgboost_results.parquet')

# Dataset original con precios
df = pd.read_parquet('./cache/dataset_with_target.parquet')

# Asegurar indice UTC en preds
if preds.index.tz is None:
    preds.index = preds.index.tz_localize('UTC')

# Filtrar df solo a las fechas con predicciones (rango OOS)
common_idx = df.index.intersection(preds.index)
# Tu df tiene columnas: btc_close, btc_high, btc_low, btc_volume (sin btc_open)
prices = pd.DataFrame(index=common_idx)
prices['close'] = df.loc[common_idx, 'btc_close']
prices['high'] = df.loc[common_idx, 'btc_high']
prices['low'] = df.loc[common_idx, 'btc_low']
prices['volume'] = df.loc[common_idx, 'btc_volume']
# Aproximacion: open[t] ≈ close[t-1] (en daily, error muy pequeno)
prices['open'] = prices['close'].shift(1).bfill()

preds_aligned = preds.loc[common_idx]

print(f"Periodo OOS: {common_idx[0].date()} -> {common_idx[-1].date()}")
print(f"Muestras: {len(common_idx)}")
print(f"Buy & Hold simple: BTC pasa de ${prices['close'].iloc[0]:,.0f} "
      f"a ${prices['close'].iloc[-1]:,.0f}")
bh_return = prices['close'].iloc[-1] / prices['close'].iloc[0] - 1
print(f"Retorno B&H: {bh_return*100:+.2f}%")

# ============================================================
# 2. CONFIGURACION DEL BACKTEST
# ============================================================
print("\n" + "=" * 70)
print("PASO 2: Configuracion realista del backtest")
print("=" * 70)

config = BacktestConfig(
    initial_capital=10_000.0,
    fee_bps=10.0,            # 0.10% taker en Binance
    slippage_bps=5.0,         # 5 bps slippage base
    slippage_vol_mult=0.5,    # Slippage adicional con volatilidad
    max_position_pct=0.95,    # Max 95% del equity en una posicion
    allow_short=False,        # No shorts (spot only)
    risk_free_rate=0.04,      # 4% anual (treasury)
)

print(f"Capital inicial:    ${config.initial_capital:,.2f}")
print(f"Fee por trade:      {config.fee_bps} bps ({config.fee_bps/100:.2f}%)")
print(f"Slippage base:      {config.slippage_bps} bps")
print(f"Max posicion:       {config.max_position_pct*100:.0f}% del equity")
print(f"Permite shorts:     {'Si' if config.allow_short else 'No (spot)'}")

# ============================================================
# 3. CONSTRUCCION DE 3 ESTRATEGIAS
# ============================================================
print("\n" + "=" * 70)
print("PASO 3: Construyendo 3 estrategias para comparar")
print("=" * 70)

# --- ESTRATEGIA 1: BUY & HOLD ---
signals_bh = pd.Series(1, index=prices.index)  # siempre long
print("Estrategia 1 (B&H): siempre long")

# --- ESTRATEGIA 2: MODO AGRESIVO (sigue todas las senales del modelo) ---
signals_aggressive = pd.Series(0, index=prices.index, dtype=float)
signals_aggressive[preds_aligned['y_pred'] == 1] = 1   # long si predice alcista
signals_aggressive[preds_aligned['y_pred'] == -1] = 0  # sin posicion (no short)
signals_aggressive[preds_aligned['y_pred'] == 0] = 0   # sin posicion lateral
print(f"Estrategia 2 (Agresiva): long cuando predice alcista, "
      f"flat resto. {(signals_aggressive == 1).sum()} dias long.")

# --- ESTRATEGIA 3: MODO CONSERVADOR (filtro de confianza) ---
PROBA_THRESHOLD = 0.50
signals_conservative = pd.Series(0, index=prices.index, dtype=float)
high_confidence = preds_aligned['proba_alcista'] > PROBA_THRESHOLD
signals_conservative[high_confidence] = 1
print(f"Estrategia 3 (Conservadora): long solo cuando "
      f"P(alcista) > {PROBA_THRESHOLD}. {high_confidence.sum()} dias long.")

# ============================================================
# 4. RUN BACKTESTS
# ============================================================
print("\n" + "=" * 70)
print("PASO 4: Ejecutando los 3 backtests...")
print("=" * 70)

bt = Backtester(config)

print("\n--- Estrategia 1: Buy & Hold ---")
result_bh = bt.run(prices, signals_bh)
print(result_bh.summary())

print("\n--- Estrategia 2: Modelo Agresivo ---")
result_agg = bt.run(prices, signals_aggressive)
print(result_agg.summary())

print("\n--- Estrategia 3: Modelo Conservador (filtro confianza) ---")
result_cons = bt.run(prices, signals_conservative)
print(result_cons.summary())

# ============================================================
# 5. TABLA COMPARATIVA
# ============================================================
print("\n" + "=" * 70)
print("PASO 5: Comparacion lado a lado")
print("=" * 70)

comparison = pd.DataFrame({
    'Buy & Hold': result_bh.metrics,
    'Modelo Agresivo': result_agg.metrics,
    'Modelo Conservador': result_cons.metrics,
}).T

# Seleccionar metricas clave
key_metrics = ['total_return', 'cagr', 'sharpe', 'sortino', 'calmar',
               'max_drawdown', 'volatility', 'n_trades', 'win_rate',
               'profit_factor', 'total_fees_paid']

# Formatear para imprimir
display = pd.DataFrame()
for m in key_metrics:
    if m in comparison.columns:
        if m in ['total_return', 'cagr', 'max_drawdown', 'volatility', 'win_rate']:
            display[m] = comparison[m].apply(lambda x: f"{x*100:+.2f}%")
        elif m in ['sharpe', 'sortino', 'calmar', 'profit_factor']:
            display[m] = comparison[m].apply(lambda x: f"{x:.2f}")
        elif m == 'n_trades':
            display[m] = comparison[m].apply(lambda x: f"{int(x)}")
        elif m == 'total_fees_paid':
            display[m] = comparison[m].apply(lambda x: f"${x:.2f}")

print("\n" + display.to_string())

# ============================================================
# 6. VEREDICTO HONESTO
# ============================================================
print("\n" + "=" * 70)
print("PASO 6: Veredicto honesto")
print("=" * 70)

bh_ret = result_bh.metrics['total_return']
agg_ret = result_agg.metrics['total_return']
cons_ret = result_cons.metrics['total_return']

bh_sharpe = result_bh.metrics['sharpe']
agg_sharpe = result_agg.metrics['sharpe']
cons_sharpe = result_cons.metrics['sharpe']

bh_mdd = result_bh.metrics['max_drawdown']
agg_mdd = result_agg.metrics['max_drawdown']
cons_mdd = result_cons.metrics['max_drawdown']

print(f"\n--- Retorno Total OOS ---")
print(f"Buy & Hold:           {bh_ret*100:+.2f}%")
print(f"Modelo Agresivo:      {agg_ret*100:+.2f}%   (vs B&H: {(agg_ret-bh_ret)*100:+.2f}pp)")
print(f"Modelo Conservador:   {cons_ret*100:+.2f}%   (vs B&H: {(cons_ret-bh_ret)*100:+.2f}pp)")

print(f"\n--- Sharpe Ratio (mejor = mas alto) ---")
print(f"Buy & Hold:           {bh_sharpe:+.2f}")
print(f"Modelo Agresivo:      {agg_sharpe:+.2f}")
print(f"Modelo Conservador:   {cons_sharpe:+.2f}")

print(f"\n--- Max Drawdown (mejor = mas cerca de 0) ---")
print(f"Buy & Hold:           {bh_mdd*100:.2f}%")
print(f"Modelo Agresivo:      {agg_mdd*100:.2f}%")
print(f"Modelo Conservador:   {cons_mdd*100:.2f}%")

# Veredicto
print("\n--- Veredicto ---")
best_strategy = max([
    ('Buy & Hold', bh_sharpe, bh_ret),
    ('Modelo Agresivo', agg_sharpe, agg_ret),
    ('Modelo Conservador', cons_sharpe, cons_ret),
], key=lambda x: x[1])

print(f"Mejor estrategia por Sharpe: {best_strategy[0]} ({best_strategy[1]:.2f})")

if max(agg_sharpe, cons_sharpe) > bh_sharpe + 0.3:
    print("EL MODELO LE GANA AL B&H POR MARGEN SIGNIFICATIVO. Sistema viable.")
elif max(agg_sharpe, cons_sharpe) > bh_sharpe:
    print("Modelo le gana marginalmente al B&H. Necesita mas optimizacion.")
else:
    print("Modelo NO le gana al B&H. Hay que iterar (mas features o modelo).")

# ============================================================
# 7. VISUALIZACION
# ============================================================
print("\n" + "=" * 70)
print("PASO 7: Generando graficos comparativos...")
print("=" * 70)

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
fig.suptitle('Backtest XGBoost - 3 estrategias comparadas',
             fontsize=14, fontweight='bold')

# Panel 1: Equity curves
ax = axes[0]
ax.plot(result_bh.equity.index, result_bh.equity.values,
        label=f'Buy & Hold ({bh_ret*100:+.1f}%)',
        color='#1976D2', linewidth=2)
ax.plot(result_agg.equity.index, result_agg.equity.values,
        label=f'Agresivo ({agg_ret*100:+.1f}%)',
        color='#F57C00', linewidth=1.5)
ax.plot(result_cons.equity.index, result_cons.equity.values,
        label=f'Conservador ({cons_ret*100:+.1f}%)',
        color='#388E3C', linewidth=1.5)
ax.axhline(config.initial_capital, color='gray', linestyle='--', alpha=0.5,
           label='Capital inicial')
ax.set_ylabel('Equity (USD)')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Curva de equity (las 3 estrategias)')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Panel 2: Drawdowns
ax = axes[1]
def drawdown_series(equity):
    cummax = equity.cummax()
    return (equity - cummax) / cummax * 100

ax.fill_between(result_bh.equity.index, drawdown_series(result_bh.equity), 0,
                color='#1976D2', alpha=0.3, label='B&H DD')
ax.fill_between(result_agg.equity.index, drawdown_series(result_agg.equity), 0,
                color='#F57C00', alpha=0.3, label='Agresivo DD')
ax.fill_between(result_cons.equity.index, drawdown_series(result_cons.equity), 0,
                color='#388E3C', alpha=0.3, label='Conservador DD')
ax.set_ylabel('Drawdown (%)')
ax.set_title('Drawdowns (cuanto pierde desde su peak)')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

# Panel 3: Posiciones del modelo conservador (cuando opera)
ax = axes[2]
ax.plot(prices.index, prices['close'], color='black', linewidth=0.5, alpha=0.5,
        label='BTC')
long_dates = signals_conservative[signals_conservative == 1].index
ax.scatter(long_dates, prices.loc[long_dates, 'close'],
           color='#388E3C', s=8, alpha=0.6, label='Long (conservador)')
ax.set_ylabel('Precio BTC')
ax.set_xlabel('Fecha')
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'${x:,.0f}'))
ax.set_title('Cuando el modelo conservador esta long')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

for ax in axes:
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

plt.tight_layout()
plt.savefig('backtest_comparison.png', dpi=120, bbox_inches='tight')
print("Grafico guardado: backtest_comparison.png")
plt.show()

# Guardar resultados
result_bh.equity.to_frame('equity').to_parquet('./cache/equity_bh.parquet')
result_agg.equity.to_frame('equity').to_parquet('./cache/equity_aggressive.parquet')
result_cons.equity.to_frame('equity').to_parquet('./cache/equity_conservative.parquet')

print("\nSesion 4 completada. Revisa la tabla comparativa y los graficos.")