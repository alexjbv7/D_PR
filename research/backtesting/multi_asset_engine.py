"""
Multi-Asset Backtest Engine
===========================
Motor que entiende microstructure de FX y futuros: opera en lots/contratos,
con spreads, comisiones y swap realistas.

DIFERENCIAS CON `engine.py` (crypto):
- Posiciones en `n_units` (signed) en vez de "fracción de equity"
- P&L vía `instrument.pnl_usd(n_units, p_entry, p_exit, conversion_factor)`
- Costes: spread (en unidades de precio) + commission (USD/unidad/lado) + slippage
- Swap diario para FX (al cierre de sesión NY ~5pm)
- Comisiones por contrato/lado para futuros
- Margen informacional (no se exige aún en este engine)

EJECUCIÓN:
- Señal en t (basada en close[t] o anterior) se EJECUTA al open[t+1]
- Spread se aplica como half-spread en cada lado del trade
- Slippage adicional proporcional a vol reciente

ESTADOS GUARDADOS POR BARRA:
- equity (mark-to-market a close[t])
- n_units (posición al final de t)
- realized_cash (cash settled, sin posiciones abiertas)
- unrealized_pnl (de posición abierta)
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

# Permitir uso como módulo o standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instruments.specs import InstrumentSpec, ForexSpec, FutureSpec, AssetClass
from backtesting.engine import (
    BacktestResult,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    cagr,
    max_drawdown,
    annualization_factor,
)


# =====================================================================
# CONFIG
# =====================================================================

@dataclass
class MultiAssetBacktestConfig:
    """
    Configuración del backtester multi-asset.

    Parameters
    ----------
    initial_capital : float
        Capital inicial en USD.
    extra_slippage_in_price : float
        Slippage adicional al spread, en unidades de precio. Se suma al medio-spread.
    slippage_vol_mult : float
        Slippage proporcional a vol reciente. Ej: 0.3 → +0.3 × vol_20bar (en precio).
    apply_swap : bool
        Si True, aplica swap diario para FX (al close del primer bar de cada día UTC).
    risk_free_rate : float
        Anualizado, para Sharpe.
    allow_short : bool
        Permitir posiciones cortas.
    """
    initial_capital: float = 10_000.0
    extra_slippage_in_price: float = 0.0
    slippage_vol_mult: float = 0.0
    apply_swap: bool = True
    risk_free_rate: float = 0.04
    allow_short: bool = True


# =====================================================================
# UTILS
# =====================================================================

def _is_new_day(ts_now, ts_prev) -> bool:
    """True si ts_now y ts_prev caen en días distintos (UTC)."""
    if ts_prev is None:
        return False
    try:
        return ts_now.date() != ts_prev.date()
    except AttributeError:
        return False


# =====================================================================
# BACKTESTER
# =====================================================================

class MultiAssetBacktester:
    """
    Backtester multi-asset con microstructure realista.

    Uso típico:
        from instruments import EURUSD
        from risk.sizing_multi_asset import ATRRiskSizer, compute_atr

        cfg = MultiAssetBacktestConfig(initial_capital=10_000.0)
        sizer = ATRRiskSizer(
            risk_pct=0.005, atr_stop_mult=2.0, instrument=EURUSD,
        )
        bt = MultiAssetBacktester(EURUSD, cfg)
        result = bt.run(prices, signals, sizer)
    """

    def __init__(
        self,
        instrument: InstrumentSpec,
        config: MultiAssetBacktestConfig,
    ):
        self.instrument = instrument
        self.config = config

    # =================================================================
    # MAIN RUN
    # =================================================================

    def run(
        self,
        prices: pd.DataFrame,
        signals: pd.Series,
        position_sizer: Callable,
        usd_conversion_series: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        Parameters
        ----------
        prices : DataFrame con [open, high, low, close] (volume opcional).
                 Debe tener DatetimeIndex.
        signals : Series alineada con prices.index. Valores ∈ {-1, 0, 1}.
                  La señal en t se ejecuta al OPEN de t+1.
        position_sizer : Callable. Firma:
                  (signal, current_equity, current_price, current_atr,
                   current_date=None, usd_conversion_factor=1.0) -> n_units (signed)
        usd_conversion_series : Series | None
                  Factor de conversión a USD por timestamp. Default = 1.0
                  (apropiado para EURUSD, GBPUSD, ES, NQ con cuenta USD).
                  Para USDJPY: pasar 1.0 / prices['close'] (aprox).
        """
        cfg = self.config
        instr = self.instrument

        # Validaciones
        if not signals.index.equals(prices.index):
            signals = signals.reindex(prices.index).fillna(0)

        if usd_conversion_series is None:
            usd_conv = pd.Series(1.0, index=prices.index)
        else:
            usd_conv = usd_conversion_series.reindex(prices.index).fillna(1.0)

        # Pre-computa ATR para el sizer (ATR(14) por defecto)
        from risk.sizing_multi_asset import compute_atr  # late import evita ciclos
        atr_series = compute_atr(prices, period=14)

        # Volatilidad reciente para slippage dinámico (en precio)
        ret_log = np.log(prices["close"] / prices["close"].shift(1))
        recent_vol_price = (
            (ret_log.rolling(20).std() * prices["close"]).fillna(0.0)
        )

        # Señal desplazada: la señal en t se ejecuta al open de t+1.
        # Equivalentemente: target_signal[t] proviene de signals[t-1].
        target_signal = signals.shift(1).fillna(0)
        if not cfg.allow_short:
            target_signal = target_signal.clip(lower=0)

        # ============================================================
        # ESTADO
        # ============================================================
        n_bars = len(prices)
        equity = pd.Series(index=prices.index, dtype=float)
        positions = pd.Series(index=prices.index, dtype=float)
        equity.iloc[0] = cfg.initial_capital
        positions.iloc[0] = 0.0

        cash_realized = cfg.initial_capital  # USD que ya están "en banco"
        n_units = 0.0
        entry_price = None
        prev_ts = None

        trades_records: list[dict] = []

        # ============================================================
        # LOOP PRINCIPAL
        # ============================================================
        for i in range(1, n_bars):
            t = prices.index[i]
            open_t = float(prices["open"].iloc[i])
            close_t = float(prices["close"].iloc[i])
            conv_t = float(usd_conv.iloc[i])

            # ---- 1. Equity al OPEN (mark-to-market) ----
            if n_units != 0 and entry_price is not None:
                unrealized = instr.pnl_usd(
                    n_units, entry_price, open_t, conv_t
                )
            else:
                unrealized = 0.0
            current_equity_open = cash_realized + unrealized

            # ---- 2. Determinar target n_units ----
            target_sig = float(target_signal.iloc[i])
            atr_t = float(atr_series.iloc[i]) if not np.isnan(atr_series.iloc[i]) else 0.0
            target_units_raw = position_sizer(
                target_sig,
                current_equity_open,
                open_t,
                atr_t,
                current_date=t.date() if hasattr(t, "date") else None,
                usd_conversion_factor=conv_t,
            )
            target_units = instr.round_to_min_increment(target_units_raw)

            # ---- 3. Si hay cambio → ejecutar ----
            if not np.isclose(target_units, n_units, atol=1e-9):
                delta = target_units - n_units
                direction = 1 if delta > 0 else -1

                # Slippage total = half_spread + extra + vol-prop
                vol_slip = cfg.slippage_vol_mult * recent_vol_price.iloc[i]
                half_spread = instr.typical_spread_in_price / 2.0
                exec_slip_price = half_spread + cfg.extra_slippage_in_price + vol_slip
                exec_price = open_t + direction * exec_slip_price

                # Si hay posición abierta: realizar P&L de la porción cerrada
                # (caso simple: el delta cierra completamente y/o abre nueva)
                realized_pnl = 0.0
                if n_units != 0 and entry_price is not None:
                    # Si el signo cambia, primero cerramos n_units a exec_price,
                    # luego abrimos target_units desde exec_price.
                    # Si solo aumentamos en mismo signo: weighted-avg entry.
                    if (n_units > 0 and target_units < 0) or \
                       (n_units < 0 and target_units > 0) or \
                       (target_units == 0):
                        realized_pnl = instr.pnl_usd(
                            n_units, entry_price, exec_price, conv_t
                        )
                        cash_realized += realized_pnl
                        if target_units == 0:
                            entry_price = None
                        else:
                            entry_price = exec_price
                    elif np.sign(n_units) == np.sign(target_units):
                        # Mismo signo: actualizar entry como weighted-avg si AUMENTA;
                        # si DISMINUYE (parcial close), realizar PnL del delta cerrado.
                        if abs(target_units) > abs(n_units):
                            # AUMENTO: weighted-avg entry
                            new_units_added = target_units - n_units
                            entry_price = (
                                (entry_price * n_units + exec_price * new_units_added)
                                / target_units
                            )
                        else:
                            # DISMINUCIÓN parcial: realizar PnL del closed_delta
                            closed_units = n_units - target_units
                            realized_pnl = instr.pnl_usd(
                                closed_units, entry_price, exec_price, conv_t
                            )
                            cash_realized += realized_pnl
                            # entry_price del remanente se mantiene
                else:
                    # Sin posición previa, abrir nueva
                    entry_price = exec_price

                # Comisión: por unidad por lado
                commission = abs(delta) * instr.commission_per_unit_per_side_usd
                cash_realized -= commission

                trades_records.append({
                    "timestamp": t,
                    "delta_units": float(delta),
                    "direction": int(direction),
                    "exec_price": float(exec_price),
                    "slippage_in_price": float(exec_slip_price),
                    "commission_usd": float(commission),
                    "realized_pnl_usd": float(realized_pnl),
                    "n_units_after": float(target_units),
                    "equity_before": float(current_equity_open),
                })

                n_units = target_units

            # ---- 4. Swap diario (FX, si activo) ----
            if cfg.apply_swap and instr.is_forex and n_units != 0:
                if _is_new_day(t, prev_ts):
                    if n_units > 0:
                        swap = n_units * instr.swap_long_usd_per_unit_per_day
                    else:
                        swap = abs(n_units) * instr.swap_short_usd_per_unit_per_day
                    cash_realized += swap

            # ---- 5. Equity al CLOSE (MTM) ----
            if n_units != 0 and entry_price is not None:
                unrealized_close = instr.pnl_usd(
                    n_units, entry_price, close_t, conv_t
                )
            else:
                unrealized_close = 0.0
            equity.iloc[i] = cash_realized + unrealized_close
            positions.iloc[i] = n_units

            prev_ts = t

        # ============================================================
        # POST-PROCESS
        # ============================================================
        returns = equity.pct_change().fillna(0)
        trades = pd.DataFrame(trades_records)

        # PnL por trade (en este modelo simple, contamos round-trips usando
        # realized_pnl_usd != 0 como cierre)
        trade_pnls: list[float] = []
        if len(trades) > 0:
            closed_mask = trades["realized_pnl_usd"] != 0.0
            trade_pnls = trades.loc[closed_mask, "realized_pnl_usd"].tolist()

        # Métricas
        ann_factor = annualization_factor(prices.index)
        mdd, peak_date, trough_date = max_drawdown(equity)

        wins = [p for p in trade_pnls if p > 0]
        losses = [p for p in trade_pnls if p < 0]

        metrics = {
            "instrument": instr.symbol,
            "asset_class": instr.asset_class.value,
            "initial_capital": cfg.initial_capital,
            "final_equity": float(equity.iloc[-1]),
            "total_return": float(equity.iloc[-1] / cfg.initial_capital - 1),
            "cagr": cagr(equity),
            "volatility": float(returns.std() * np.sqrt(ann_factor)),
            "sharpe": sharpe_ratio(returns, cfg.risk_free_rate, ann_factor),
            "sortino": sortino_ratio(returns, cfg.risk_free_rate, ann_factor),
            "calmar": calmar_ratio(equity),
            "max_drawdown": mdd,
            "mdd_peak": peak_date,
            "mdd_trough": trough_date,
            "n_trades": len(trades),
            "n_round_trips": len(trade_pnls),
            "win_rate": (
                len(wins) / max(1, len(trade_pnls))
                if trade_pnls else 0.0
            ),
            "profit_factor": (
                sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 0.0
            ),
            "avg_trade_pnl_usd": (
                float(np.mean(trade_pnls)) if trade_pnls else 0.0
            ),
            "avg_win_loss_ratio": (
                (np.mean(wins) / abs(np.mean(losses)))
                if wins and losses else 0.0
            ),
            "total_commission_usd": (
                float(trades["commission_usd"].sum()) if len(trades) > 0 else 0.0
            ),
        }

        return BacktestResult(
            equity=equity,
            returns=returns,
            positions=positions,
            trades=trades,
            metrics=metrics,
        )
