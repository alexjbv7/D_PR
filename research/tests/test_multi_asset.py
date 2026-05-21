"""
Tests del motor multi-asset.

Casos críticos validados:
1. P&L correcto en pips (FX) y ticks (futuros) con valores conocidos.
2. Comisión por lado se aplica.
3. Spread reduce P&L.
4. Round trip cerrado tiene P&L = (price_exit - price_entry) × multiplier × N.
5. Short positions funcionan (PnL invertido).
6. Sizing ATR-based produce el risk_usd esperado al activar el stop.
7. Round to min_increment respeta micro-lots y contratos enteros.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from instruments import EURUSD, ES, NQ, USDJPY
from instruments.specs import ForexSpec
from backtesting.multi_asset_engine import (
    MultiAssetBacktester,
    MultiAssetBacktestConfig,
)
from risk.sizing_multi_asset import (
    ATRRiskSizer,
    FixedUnitsSizer,
    compute_atr,
)


def _approx(a, b, tol=1e-3):
    return abs(a - b) < tol


def _make_prices(closes, freq="1h", start="2024-01-01"):
    """Helper: construye OHLCV simple con high/low triviales."""
    idx = pd.date_range(start, periods=len(closes), freq=freq)
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": closes,
        "high": closes * 1.0001,
        "low": closes * 0.9999,
        "close": closes,
        "volume": np.full(len(closes), 1000.0),
    }, index=idx)


# =====================================================================
# 1. PNL CORRECTO EN PIPS (EURUSD)
# =====================================================================

def test_eurusd_pnl_one_lot_one_pip():
    """1 lot EURUSD, +1 pip → +$10 (cuenta USD)."""
    p = EURUSD.pnl_usd(
        n_units=1.0,
        price_entry=1.1000,
        price_exit=1.1001,  # +1 pip
        usd_conversion_factor=1.0,
    )
    assert _approx(p, 10.0), f"Esperaba +$10, obtuvimos {p}"
    print("✓ test_eurusd_pnl_one_lot_one_pip: PASSED")


def test_eurusd_pnl_short():
    """1 lot SHORT EURUSD, -1 pip movimiento → +$10 (gano siendo short)."""
    p = EURUSD.pnl_usd(
        n_units=-1.0,
        price_entry=1.1000,
        price_exit=1.0999,  # -1 pip → short gana
        usd_conversion_factor=1.0,
    )
    assert _approx(p, 10.0), f"Esperaba +$10 short, obtuvimos {p}"
    print("✓ test_eurusd_pnl_short: PASSED")


def test_eurusd_micro_lot_pip():
    """0.01 lot (micro) × +1 pip → +$0.10."""
    p = EURUSD.pnl_usd(0.01, 1.1000, 1.1001, 1.0)
    assert _approx(p, 0.10), f"Esperaba +$0.10, obtuvimos {p}"
    print("✓ test_eurusd_micro_lot_pip: PASSED")


# =====================================================================
# 2. PNL FUTUROS (ES, NQ)
# =====================================================================

def test_es_pnl_one_contract_one_tick():
    """1 contrato ES, +1 tick (0.25) → +$12.50."""
    p = ES.pnl_usd(1.0, 4500.00, 4500.25, 1.0)
    assert _approx(p, 12.50), f"Esperaba +$12.50, obtuvimos {p}"
    print("✓ test_es_pnl_one_contract_one_tick: PASSED")


def test_nq_pnl_one_contract_one_point():
    """1 contrato NQ, +1 punto → +$20."""
    p = NQ.pnl_usd(1.0, 16000.00, 16001.00, 1.0)
    assert _approx(p, 20.0), f"Esperaba +$20, obtuvimos {p}"
    print("✓ test_nq_pnl_one_contract_one_point: PASSED")


def test_es_pnl_two_contracts_short_loss():
    """2 contratos SHORT ES, precio sube +10 puntos → -$1000 (pérdida = 2×50×10)."""
    p = ES.pnl_usd(-2.0, 4500.0, 4510.0, 1.0)
    assert _approx(p, -1000.0), f"Esperaba -$1000, obtuvimos {p}"
    print("✓ test_es_pnl_two_contracts_short_loss: PASSED")


# =====================================================================
# 3. ROUND-TO-MIN-INCREMENT
# =====================================================================

def test_round_lots_to_micro():
    """1.234 lots → 1.23 lots (round down a 0.01)."""
    rounded = EURUSD.round_to_min_increment(1.234)
    assert _approx(rounded, 1.23), f"Esperaba 1.23, obtuvimos {rounded}"
    print("✓ test_round_lots_to_micro: PASSED")


def test_round_contracts_to_integer():
    """3.7 contratos ES → 3 contratos (round down a 1.0)."""
    rounded = ES.round_to_min_increment(3.7)
    assert _approx(rounded, 3.0), f"Esperaba 3, obtuvimos {rounded}"
    print("✓ test_round_contracts_to_integer: PASSED")


def test_round_negative_preserves_sign():
    """-3.7 → -3 (round hacia cero)."""
    rounded = ES.round_to_min_increment(-3.7)
    assert _approx(rounded, -3.0), f"Esperaba -3, obtuvimos {rounded}"
    print("✓ test_round_negative_preserves_sign: PASSED")


# =====================================================================
# 4. ATR SIZER
# =====================================================================

def test_atr_sizer_targets_correct_risk():
    """
    Con risk_pct=1%, equity=$10k, ATR=10 puntos en ES (multiplier 50), stop_mult=2:
      stop_distance_usd = 2 × 10 × 50 = $1000 por contrato
      risk_usd = $100
      n_contracts ≈ 100 / 1000 = 0.1 → round_down → 0 (sub-mínimo)

    Test: con equity grande para que dé contratos enteros.
    """
    sizer = ATRRiskSizer(
        risk_pct=0.01, atr_stop_mult=2.0, instrument=ES,
        max_units_per_trade=10,
    )
    # Equity $100k, ATR 10 puntos
    n = sizer(
        signal=1, current_equity=100_000.0,
        current_price=4500.0, current_atr=10.0,
    )
    # risk_usd = 1000, stop_usd_per_contract = 1000 → n = 1
    assert _approx(n, 1.0), f"Esperaba 1 contrato, obtuvimos {n}"
    print("✓ test_atr_sizer_targets_correct_risk: PASSED")


def test_atr_sizer_zero_signal_returns_zero():
    sizer = ATRRiskSizer(risk_pct=0.01, atr_stop_mult=2.0, instrument=EURUSD)
    n = sizer(signal=0, current_equity=10_000.0, current_price=1.1, current_atr=0.001)
    assert _approx(n, 0.0)
    print("✓ test_atr_sizer_zero_signal_returns_zero: PASSED")


def test_atr_sizer_short_returns_negative():
    sizer = ATRRiskSizer(risk_pct=0.01, atr_stop_mult=2.0, instrument=ES,
                          max_units_per_trade=10)
    n = sizer(
        signal=-1, current_equity=100_000.0,
        current_price=4500.0, current_atr=10.0,
    )
    assert n < 0, f"Short signal debe dar n_units negativo, obtuvimos {n}"
    print("✓ test_atr_sizer_short_returns_negative: PASSED")


# =====================================================================
# 5. ENGINE: PRICE INCREASE + LONG → POSITIVE EQUITY
# =====================================================================

def test_engine_long_es_uptrend_makes_money():
    """
    ES sube linealmente +10 puntos. Long 1 contrato → +$500 bruto, menos costes.
    """
    closes = np.linspace(4500.0, 4510.0, 50)  # 50 bars subiendo
    prices = _make_prices(closes)
    signals = pd.Series(1, index=prices.index)  # always long

    sizer = FixedUnitsSizer(n_units=1.0, instrument=ES)
    cfg = MultiAssetBacktestConfig(
        initial_capital=20_000.0,
        extra_slippage_in_price=0.0,
        slippage_vol_mult=0.0,
        apply_swap=False,
    )
    bt = MultiAssetBacktester(ES, cfg)
    res = bt.run(prices, signals, sizer)

    assert res.equity.iloc[-1] > cfg.initial_capital, \
        f"Esperaba equity > {cfg.initial_capital}, obtuvimos {res.equity.iloc[-1]}"
    # P&L bruto esperado: 1 contrato × $50 × ~10 puntos = ~$500
    # Menos: 1 entrada (commission $1.50) + spread (1 tick × $50 = $12.5)
    # Equity final ≈ $20,000 + $500 - $1.5 - $12.5 ≈ $20,486
    expected_pnl_min = 450  # con margen para spread y comisión
    actual_pnl = res.equity.iloc[-1] - cfg.initial_capital
    assert actual_pnl > expected_pnl_min, \
        f"PnL debería ser ~+$500 antes de costes, obtuvimos +${actual_pnl:.2f}"
    print(f"✓ test_engine_long_es_uptrend_makes_money: PASSED  "
          f"(PnL=+${actual_pnl:.2f})")


def test_engine_long_es_downtrend_loses_money():
    closes = np.linspace(4500.0, 4490.0, 50)
    prices = _make_prices(closes)
    signals = pd.Series(1, index=prices.index)

    sizer = FixedUnitsSizer(n_units=1.0, instrument=ES)
    cfg = MultiAssetBacktestConfig(initial_capital=20_000.0, apply_swap=False)
    bt = MultiAssetBacktester(ES, cfg)
    res = bt.run(prices, signals, sizer)

    assert res.equity.iloc[-1] < cfg.initial_capital, \
        f"Long en bajada debe perder. Equity final: {res.equity.iloc[-1]}"
    print(f"✓ test_engine_long_es_downtrend_loses_money: PASSED  "
          f"(PnL=${res.equity.iloc[-1] - cfg.initial_capital:.2f})")


def test_engine_short_es_downtrend_makes_money():
    closes = np.linspace(4500.0, 4490.0, 50)
    prices = _make_prices(closes)
    signals = pd.Series(-1, index=prices.index)

    sizer = FixedUnitsSizer(n_units=1.0, instrument=ES)
    cfg = MultiAssetBacktestConfig(initial_capital=20_000.0, apply_swap=False,
                                    allow_short=True)
    bt = MultiAssetBacktester(ES, cfg)
    res = bt.run(prices, signals, sizer)

    assert res.equity.iloc[-1] > cfg.initial_capital, \
        f"Short en bajada debe ganar. Equity final: {res.equity.iloc[-1]}"
    print(f"✓ test_engine_short_es_downtrend_makes_money: PASSED  "
          f"(PnL=+${res.equity.iloc[-1] - cfg.initial_capital:.2f})")


def test_engine_eurusd_pnl_consistent():
    """
    EURUSD sube +10 pips (de 1.1000 a 1.1010). 0.10 lots long → +$10 bruto.
    """
    closes = np.linspace(1.1000, 1.1010, 30)
    prices = _make_prices(closes)
    signals = pd.Series(1, index=prices.index)

    sizer = FixedUnitsSizer(n_units=0.10, instrument=EURUSD)
    cfg = MultiAssetBacktestConfig(
        initial_capital=10_000.0,
        apply_swap=False,
    )
    bt = MultiAssetBacktester(EURUSD, cfg)
    res = bt.run(prices, signals, sizer)

    pnl = res.equity.iloc[-1] - cfg.initial_capital
    # Esperado bruto: 0.10 × 100,000 × 0.0010 = $10
    # Menos costes: 1 trade × ($3.50 commission + 0.5 pip spread × 0.10 lot × 100k = $5)
    # ≈ $10 - $3.50 - $5 = $1.50 mínimo
    assert pnl > 0, f"Debe ser positivo, obtuvimos {pnl}"
    assert pnl < 10, f"Debe ser < $10 después de costes, obtuvimos {pnl}"
    print(f"✓ test_engine_eurusd_pnl_consistent: PASSED  (PnL=+${pnl:.2f})")


# =====================================================================
# 6. SPREAD Y COMISIÓN REDUCEN PNL
# =====================================================================

def test_spread_and_commission_reduce_equity():
    """Backtest sin trade vs con trade: el trade implica costes."""
    closes = np.full(20, 4500.0)  # precio constante → P&L bruto = 0
    prices = _make_prices(closes)
    # Una entrada y salida fuerza dos trades
    signals = pd.Series(0, index=prices.index)
    signals.iloc[2] = 1
    signals.iloc[10] = 0

    sizer = FixedUnitsSizer(n_units=1.0, instrument=ES)
    cfg = MultiAssetBacktestConfig(initial_capital=20_000.0, apply_swap=False)
    bt = MultiAssetBacktester(ES, cfg)
    res = bt.run(prices, signals, sizer)

    # Sin movimiento de precio, el equity final debe ser MENOR (pagamos costes)
    assert res.equity.iloc[-1] < cfg.initial_capital, \
        f"Costes deben reducir equity. Final: {res.equity.iloc[-1]}"
    cost = cfg.initial_capital - res.equity.iloc[-1]
    print(f"✓ test_spread_and_commission_reduce_equity: PASSED  "
          f"(coste total ≈ ${cost:.2f})")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    print("\nEjecutando tests del motor multi-asset...\n")
    tests = [
        test_eurusd_pnl_one_lot_one_pip,
        test_eurusd_pnl_short,
        test_eurusd_micro_lot_pip,
        test_es_pnl_one_contract_one_tick,
        test_nq_pnl_one_contract_one_point,
        test_es_pnl_two_contracts_short_loss,
        test_round_lots_to_micro,
        test_round_contracts_to_integer,
        test_round_negative_preserves_sign,
        test_atr_sizer_targets_correct_risk,
        test_atr_sizer_zero_signal_returns_zero,
        test_atr_sizer_short_returns_negative,
        test_engine_long_es_uptrend_makes_money,
        test_engine_long_es_downtrend_loses_money,
        test_engine_short_es_downtrend_makes_money,
        test_engine_eurusd_pnl_consistent,
        test_spread_and_commission_reduce_equity,
    ]
    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"✗ {t.__name__}: FAILED — {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"✗ {t.__name__}: ERROR — {type(e).__name__}: {e}")
            failures.append(t.__name__)

    print()
    if failures:
        print(f"\n✗✗✗ {len(failures)} TESTS FALLARON: {failures}")
        sys.exit(1)
    print(f"✓✓✓ {len(tests)} TESTS PASADOS ✓✓✓")
