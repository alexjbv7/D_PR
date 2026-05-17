"""
Backtesting Engine
==================
Backtester realista que evita los errores comunes:
- Ejecuta señales en barra t+1 (open) — no t (close)
- Aplica slippage proporcional a la volatilidad
- Aplica fees diferenciadas (maker vs taker)
- Soporta position sizing dinámico
- Calcula métricas financieras profesionales
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Callable
import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURACIÓN DEL BACKTEST
# ============================================================================

@dataclass
class BacktestConfig:
    initial_capital: float = 10_000.0
    fee_bps: float = 10.0           # 10 bps = 0.1% por trade (taker en Binance)
    slippage_bps: float = 5.0       # 5 bps de slippage base
    slippage_vol_mult: float = 0.5  # Slippage adicional escalado con volatilidad
    max_position_pct: float = 1.0   # 1.0 = 100% del equity por posición
    allow_short: bool = False       # Crypto spot típicamente no permite short
    risk_free_rate: float = 0.04    # Anualizado, para Sharpe


# ============================================================================
# RESULTADOS
# ============================================================================

@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    metrics: dict = field(default_factory=dict)

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "═══════════════════ RESULTADO DEL BACKTEST ═══════════════════",
            f"  Período:           {self.equity.index[0].date()} → {self.equity.index[-1].date()}",
            f"  Capital inicial:   ${m.get('initial_capital', 0):,.2f}",
            f"  Capital final:     ${m.get('final_equity', 0):,.2f}",
            f"  Retorno total:     {m.get('total_return', 0)*100:>7.2f}%",
            f"  CAGR:              {m.get('cagr', 0)*100:>7.2f}%",
            "  ─────────────────────────────────────────────────────────",
            f"  Sharpe Ratio:      {m.get('sharpe', 0):>7.2f}",
            f"  Sortino Ratio:     {m.get('sortino', 0):>7.2f}",
            f"  Calmar Ratio:      {m.get('calmar', 0):>7.2f}",
            f"  Max Drawdown:      {m.get('max_drawdown', 0)*100:>7.2f}%",
            f"  Volatilidad anual: {m.get('volatility', 0)*100:>7.2f}%",
            "  ─────────────────────────────────────────────────────────",
            f"  Total trades:      {m.get('n_trades', 0):>7}",
            f"  Win rate:          {m.get('win_rate', 0)*100:>7.2f}%",
            f"  Profit factor:     {m.get('profit_factor', 0):>7.2f}",
            f"  Avg trade return:  {m.get('avg_trade_return', 0)*100:>7.3f}%",
            f"  Avg win/Avg loss:  {m.get('avg_win_loss_ratio', 0):>7.2f}",
            "═════════════════════════════════════════════════════════════",
        ]
        return "\n".join(lines)


# ============================================================================
# MÉTRICAS FINANCIERAS
# ============================================================================

def annualization_factor(index: pd.DatetimeIndex) -> float:
    """Calcula factor de anualización a partir de la frecuencia inferida del índice."""
    if len(index) < 2:
        return 252.0
    freq_seconds = (index[1] - index[0]).total_seconds()
    seconds_per_year = 365.25 * 24 * 3600
    return seconds_per_year / freq_seconds


def sharpe_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: float = 252) -> float:
    """Sharpe ratio anualizado."""
    if returns.std() == 0 or len(returns) == 0:
        return 0.0
    excess = returns - rf / periods_per_year
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def sortino_ratio(returns: pd.Series, rf: float = 0.0, periods_per_year: float = 252) -> float:
    """Sortino — solo penaliza downside volatility."""
    excess = returns - rf / periods_per_year
    downside = excess[excess < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / downside.std()


def max_drawdown(equity: pd.Series) -> tuple[float, pd.Timestamp, pd.Timestamp]:
    """Returns (max_dd_pct, peak_date, trough_date)."""
    cummax = equity.cummax()
    dd = (equity - cummax) / cummax
    trough = dd.idxmin()
    peak = equity.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def cagr(equity: pd.Series) -> float:
    """Compound Annual Growth Rate."""
    n_years = (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600)
    if n_years <= 0:
        return 0.0
    return (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1


def calmar_ratio(equity: pd.Series) -> float:
    cagr_val = cagr(equity)
    mdd, _, _ = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return cagr_val / abs(mdd)


# ============================================================================
# BACKTESTER PRINCIPAL
# ============================================================================

class Backtester:
    """
    Backtester vectorizado pero con execution-aware delays.

    CRÍTICO: La señal generada en la barra `t` (con close[t]) se ejecuta al
    open[t+1] (próxima barra). Esto evita look-ahead bias.

    Modelo de slippage:
        slippage = (slippage_bps + slippage_vol_mult * recent_vol) en BPS

    Modelo de fees:
        fee aplicado en cada cambio de posición (entrada y salida).
    """

    def __init__(self, config: BacktestConfig):
        self.config = config

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        position_sizer: Optional[Callable] = None,
    ) -> BacktestResult:
        """
        Parameters
        ----------
        prices : DataFrame con columnas [open, high, low, close, volume]
        signals : Series alineada con prices.index. Valores en {-1, 0, 1}.
                  La señal en t se ejecuta al open de t+1.
        position_sizer : función opcional (signal, equity, price, vol) -> position_pct
                        Si None, usa max_position_pct fija.
        """
        cfg = self.config

        # Alineamos: señal en t se vuelve target_position desde t+1
        target_position = signals.shift(1).fillna(0)

        if not cfg.allow_short:
            target_position = target_position.clip(lower=0)

        # Volatilidad reciente para slippage dinámico
        log_ret = np.log(prices['close'] / prices['close'].shift(1))
        recent_vol = log_ret.rolling(20).std().fillna(log_ret.std())

        # Position sizing: por defecto fija
        if position_sizer is None:
            position_size = pd.Series(cfg.max_position_pct, index=prices.index)
        else:
            position_size = pd.Series(index=prices.index, dtype=float)
            for t in prices.index:
                position_size.loc[t] = position_sizer(
                    target_position.loc[t],
                    1.0,  # placeholder; en práctica se actualiza on-the-fly
                    prices.loc[t, 'close'],
                    recent_vol.loc[t],
                )

        # Posiciones objetivo en términos de fracción de equity
        target_pct = target_position * position_size

        # ============================================================
        # Bucle de simulación
        # ============================================================
        equity = pd.Series(index=prices.index, dtype=float)
        equity.iloc[0] = cfg.initial_capital
        position_shares = 0.0  # número de unidades del activo
        cash = cfg.initial_capital

        trades_records = []
        last_position_pct = 0.0

        for i in range(1, len(prices)):
            t = prices.index[i]
            t_prev = prices.index[i - 1]
            open_t = prices['open'].iloc[i]
            close_t = prices['close'].iloc[i]

            # 1. Calcular equity ANTES de operar (mark-to-market al open)
            current_equity = cash + position_shares * open_t

            # 2. Decidir nuevo target_pct
            new_target_pct = target_pct.iloc[i]

            # 3. Si hay cambio de posición → ejecutar
            if not np.isclose(new_target_pct, last_position_pct, atol=1e-6):
                # Slippage: en bps sobre el open
                slip_bps = cfg.slippage_bps + cfg.slippage_vol_mult * recent_vol.iloc[i] * 10_000
                # Direction-aware: paga slippage al comprar (precio sube), recibe al vender (precio baja)
                direction = np.sign(new_target_pct - last_position_pct)
                exec_price = open_t * (1 + direction * slip_bps / 10_000)

                target_position_value = new_target_pct * current_equity
                target_shares = target_position_value / exec_price
                shares_delta = target_shares - position_shares
                trade_value = abs(shares_delta) * exec_price

                # Fee
                fee = trade_value * cfg.fee_bps / 10_000

                # Actualizar cash y posición
                cash = cash - shares_delta * exec_price - fee
                position_shares = target_shares

                trades_records.append({
                    'timestamp': t,
                    'direction': direction,
                    'shares_delta': shares_delta,
                    'exec_price': exec_price,
                    'slippage_bps': slip_bps,
                    'fee': fee,
                    'equity_before': current_equity,
                })

                last_position_pct = new_target_pct

            # 4. Mark-to-market al close de t
            equity.iloc[i] = cash + position_shares * close_t

        # ============================================================
        # Construir resultados
        # ============================================================
        returns = equity.pct_change().fillna(0)
        positions = target_pct.copy()
        trades = pd.DataFrame(trades_records)

        # PnL por trade (par compra-venta)
        trade_pnls = []
        if len(trades) > 0:
            for i in range(0, len(trades) - 1, 2):
                if i + 1 < len(trades):
                    entry = trades.iloc[i]
                    exit_ = trades.iloc[i + 1]
                    pnl = -(entry['shares_delta'] * entry['exec_price'] +
                            exit_['shares_delta'] * exit_['exec_price'] +
                            entry['fee'] + exit_['fee'])
                    trade_pnls.append(pnl)

        # ============================================================
        # MÉTRICAS
        # ============================================================
        ann_factor = annualization_factor(prices.index)
        mdd, peak_date, trough_date = max_drawdown(equity)

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]

        metrics = {
            'initial_capital': cfg.initial_capital,
            'final_equity': float(equity.iloc[-1]),
            'total_return': float(equity.iloc[-1] / cfg.initial_capital - 1),
            'cagr': cagr(equity),
            'volatility': float(returns.std() * np.sqrt(ann_factor)),
            'sharpe': sharpe_ratio(returns, cfg.risk_free_rate, ann_factor),
            'sortino': sortino_ratio(returns, cfg.risk_free_rate, ann_factor),
            'calmar': calmar_ratio(equity),
            'max_drawdown': mdd,
            'mdd_peak': peak_date,
            'mdd_trough': trough_date,
            'n_trades': len(trades),
            'win_rate': len(wins) / max(1, len(trade_pnls)),
            'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0,
            'avg_trade_return': float(np.mean(trade_pnls) / cfg.initial_capital) if trade_pnls else 0,
            'avg_win_loss_ratio': (np.mean(wins) / abs(np.mean(losses))) if wins and losses else 0,
            'total_fees_paid': float(trades['fee'].sum()) if len(trades) > 0 else 0,
        }

        return BacktestResult(
            equity=equity,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
        )
