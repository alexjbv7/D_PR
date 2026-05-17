"""
Tests de Kelly Fraccional.

Casos validados:
 1. f* = 0 cuando p_win < p_breakeven (EV negativo).
 2. f* > 0 cuando p_win > p_breakeven (EV positivo).
 3. Kelly capeado a max_risk_pct aunque f* sea muy alto.
 4. kelly_fraction=0.25 produce exactamente 25% de f*.
 5. Kelly = 0 cuando signal = 0.
 6. kelly_breakeven_probability es correcto: b=1 -> p=0.5, b=2 -> p=1/3.
 7. expected_value: EV positivo cuando p > p_break, negativo cuando p < p_break.
 8. KellyAtrSizer.__call__ devuelve 0 cuando Kelly dice no apostar.
 9. KellyAtrSizer.__call__ devuelve n_units > 0 para señal con EV positivo.
10. KellyAtrSizer calcula R:R desde atr_tp_mult/atr_sl_mult.
11. extract_p_win extrae la columna correcta para long y short.
12. estimate_rr_ratio calcula ratio correcto con trades sintéticos.
13. estimate_rr_ratio devuelve 1.0 con pocos trades.
14. sizing_report tiene todas las claves esperadas.
15. Kelly mayor con mayor p_win (monotónico).
16. KellyAtrSizer devuelve 0 si ATR = 0.
17. daily_loss_pct_pause bloquea trades cuando se alcanza límite diario.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from risk.kelly import (
    kelly_fraction_binary,
    kelly_breakeven_probability,
    expected_value,
    estimate_rr_ratio,
    extract_p_win,
    KellyAtrSizer,
)


# =====================================================================
# MOCK INSTRUMENT (sin importar instruments.specs)
# =====================================================================

class MockInstrument:
    """Instrumento simple para tests."""
    usd_per_unit_per_price_point = 10.0   # como ES: $10 por punto
    min_size_increment = 1.0

    def round_to_min_increment(self, n):
        return float(int(n / self.min_size_increment) * self.min_size_increment)


# =====================================================================
# TESTS KELLY FORMULA
# =====================================================================

def test_kelly_zero_when_ev_negative():
    """f* <= 0 cuando p_win < p_breakeven -> Kelly = 0."""
    rr = 1.0        # b=1: necesitas p > 0.5
    p_lose = 0.40   # p < 0.5 -> EV negativo
    f = kelly_fraction_binary(p_lose, rr)
    assert f == 0.0, f"Kelly debe ser 0 cuando EV negativo, obtuvimos {f}"
    print("OK test_kelly_zero_when_ev_negative")


def test_kelly_positive_when_ev_positive():
    """f* > 0 cuando p_win > p_breakeven -> Kelly > 0."""
    rr = 1.5        # p_break = 1/2.5 = 0.40
    p_win = 0.55    # p > 0.40 -> EV positivo
    f = kelly_fraction_binary(p_win, rr)
    assert f > 0.0, f"Kelly debe ser > 0 cuando EV positivo, obtuvimos {f}"
    print(f"OK test_kelly_positive_when_ev_positive (f={f:.4f})")


def test_kelly_exact_breakeven():
    """En p = p_breakeven, f* = 0 exacto."""
    rr = 2.0
    p_break = kelly_breakeven_probability(rr)   # = 1/3
    f = kelly_fraction_binary(p_break, rr)
    assert f == 0.0, f"Kelly en breakeven debe ser 0, obtuvimos {f}"
    print(f"OK test_kelly_exact_breakeven (p_break={p_break:.4f})")


def test_kelly_fraction_scales_correctly():
    """kelly_fraction=0.5 devuelve exactamente el doble que kelly_fraction=0.25."""
    rr = 2.0
    p_win = 0.60
    f_quarter = kelly_fraction_binary(p_win, rr, kelly_fraction=0.25)
    f_half = kelly_fraction_binary(p_win, rr, kelly_fraction=0.50)
    assert abs(f_half - 2 * f_quarter) < 1e-9, (
        f"f_half={f_half:.6f} deberia ser 2 x f_quarter={f_quarter:.6f}"
    )
    print(f"OK test_kelly_fraction_scales_correctly (quarter={f_quarter:.4f}, half={f_half:.4f})")


def test_kelly_capped_at_kelly_fraction():
    """f_kelly nunca supera kelly_fraction, incluso con p=1."""
    rr = 10.0
    p_win = 0.999
    kelly_frac = 0.25
    f = kelly_fraction_binary(p_win, rr, kelly_fraction=kelly_frac)
    assert f <= kelly_frac + 1e-9, f"Kelly no puede superar kelly_fraction, obtuvimos {f}"
    print(f"OK test_kelly_capped_at_kelly_fraction (f={f:.4f})")


def test_kelly_monotonic_in_p():
    """Mayor p_win -> mayor Kelly (monotonicidad)."""
    rr = 1.5
    probs = [0.42, 0.50, 0.60, 0.70, 0.80, 0.90]
    kellys = [kelly_fraction_binary(p, rr) for p in probs]
    for i in range(len(kellys) - 1):
        assert kellys[i] <= kellys[i+1], (
            f"Kelly no monótono: p={probs[i]}->{probs[i+1]}, "
            f"f={kellys[i]:.4f}->{kellys[i+1]:.4f}"
        )
    print(f"OK test_kelly_monotonic_in_p {[round(k,3) for k in kellys]}")


# =====================================================================
# TESTS FUNCIONES AUXILIARES
# =====================================================================

def test_breakeven_prob_b1():
    """b=1 (R:R igualado) -> necesitas exactamente p=0.5 para EV=0."""
    p_break = kelly_breakeven_probability(1.0)
    assert abs(p_break - 0.5) < 1e-9, f"p_break(b=1)={p_break}, esperaba 0.5"
    print("OK test_breakeven_prob_b1")


def test_breakeven_prob_b2():
    """b=2 -> p_break = 1/3 = 0.333..."""
    p_break = kelly_breakeven_probability(2.0)
    assert abs(p_break - 1/3) < 1e-9, f"p_break(b=2)={p_break}, esperaba 1/3"
    print("OK test_breakeven_prob_b2")


def test_expected_value_positive():
    """EV > 0 cuando p > p_breakeven."""
    rr = 1.5
    p_break = kelly_breakeven_probability(rr)
    ev = expected_value(p_break + 0.05, rr)
    assert ev > 0, f"EV debe ser > 0 cuando p > p_break, obtuvimos {ev}"
    print(f"OK test_expected_value_positive (EV={ev:.4f})")


def test_expected_value_negative():
    """EV < 0 cuando p < p_breakeven."""
    rr = 1.5
    p_break = kelly_breakeven_probability(rr)
    ev = expected_value(p_break - 0.05, rr)
    assert ev < 0, f"EV debe ser < 0 cuando p < p_break, obtuvimos {ev}"
    print(f"OK test_expected_value_negative (EV={ev:.4f})")


def test_estimate_rr_correct():
    """estimate_rr_ratio calcula avg_win/avg_loss correctamente."""
    wins = np.full(30, 0.015)    # 30 trades ganadores de 1.5%
    losses = np.full(20, -0.010) # 20 trades perdedores de 1.0%
    trades = np.concatenate([wins, losses])
    rr = estimate_rr_ratio(trades, min_trades=10)
    assert abs(rr - 1.5) < 1e-6, f"R:R esperado 1.5, obtuvimos {rr}"
    print(f"OK test_estimate_rr_correct (R:R={rr:.4f})")


def test_estimate_rr_fallback_few_trades():
    """Con pocos trades devuelve 1.0 (prior neutral)."""
    trades = np.array([0.01, -0.01, 0.02])
    rr = estimate_rr_ratio(trades, min_trades=20)
    assert rr == 1.0, f"R:R debe ser 1.0 con pocos trades, obtuvimos {rr}"
    print("OK test_estimate_rr_fallback_few_trades")


def test_extract_p_win_long():
    """extract_p_win extrae P(y=+1) para señal long."""
    class_labels = [-1, 0, 1]
    proba = np.array([0.10, 0.30, 0.60])  # P(+1) = 0.60
    p = extract_p_win(proba, signal=1, class_labels=class_labels)
    assert abs(p - 0.60) < 1e-9, f"P(win|long) esperado 0.60, obtuvimos {p}"
    print("OK test_extract_p_win_long")


def test_extract_p_win_short():
    """extract_p_win extrae P(y=-1) para señal short."""
    class_labels = [-1, 0, 1]
    proba = np.array([0.55, 0.25, 0.20])  # P(-1) = 0.55
    p = extract_p_win(proba, signal=-1, class_labels=class_labels)
    assert abs(p - 0.55) < 1e-9, f"P(win|short) esperado 0.55, obtuvimos {p}"
    print("OK test_extract_p_win_short")


# =====================================================================
# TESTS KellyAtrSizer
# =====================================================================

def test_kelly_sizer_zero_on_neutral_signal():
    """signal=0 siempre devuelve 0 unidades."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(instrument=inst)
    n = sizer(signal=0, p_win=0.8, current_equity=10000,
               current_price=4500, current_atr=20)
    assert n == 0.0
    print("OK test_kelly_sizer_zero_on_neutral_signal")


def test_kelly_sizer_zero_on_zero_atr():
    """ATR=0 devuelve 0 (no se puede calcular stop)."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(instrument=inst)
    n = sizer(signal=1, p_win=0.8, current_equity=10000,
               current_price=4500, current_atr=0)
    assert n == 0.0
    print("OK test_kelly_sizer_zero_on_zero_atr")


def test_kelly_sizer_zero_when_low_probability():
    """p_win < p_breakeven -> Kelly = 0 -> 0 unidades."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(instrument=inst, rr_ratio=1.5, kelly_fraction=0.25)
    # p_break(b=1.5) = 0.40; p=0.30 < 0.40 -> EV negativo
    n = sizer(signal=1, p_win=0.30, current_equity=10000,
               current_price=4500, current_atr=20)
    assert n == 0.0, f"Debe ser 0 con EV negativo, obtuvimos {n}"
    print("OK test_kelly_sizer_zero_when_low_probability")


def test_kelly_sizer_positive_units_when_positive_ev():
    """p_win > p_breakeven -> Kelly > 0 -> n_units > 0."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(
        instrument=inst,
        rr_ratio=1.5,
        kelly_fraction=0.25,
        max_risk_pct=0.02,
    )
    # p=0.70 > p_break=0.40 con b=1.5 -> EV positivo
    n = sizer(signal=1, p_win=0.70, current_equity=100000,
               current_price=4500, current_atr=20)
    assert n > 0, f"Debe tener n_units > 0 con EV positivo, obtuvimos {n}"
    print(f"OK test_kelly_sizer_positive_units_when_positive_ev (n={n})")


def test_kelly_sizer_short_signal_negative_units():
    """signal=-1 y p_win alto -> n_units < 0 (short)."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(instrument=inst, rr_ratio=2.0, kelly_fraction=0.25)
    n = sizer(signal=-1, p_win=0.70, current_equity=100000,
               current_price=4500, current_atr=20)
    assert n < 0, f"Short debe dar n_units < 0, obtuvimos {n}"
    print(f"OK test_kelly_sizer_short_signal_negative_units (n={n})")


def test_kelly_rr_from_atr_multiples():
    """Si atr_tp_mult=3, atr_sl_mult=2 -> R:R = 1.5."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(
        instrument=inst, atr_sl_mult=2.0, atr_tp_mult=3.0
    )
    assert abs(sizer.effective_rr_ratio - 1.5) < 1e-9, (
        f"R:R esperado 1.5, obtuvimos {sizer.effective_rr_ratio}"
    )
    print(f"OK test_kelly_rr_from_atr_multiples (R:R={sizer.effective_rr_ratio})")


def test_kelly_risk_pct_capped():
    """risk_pct nunca supera max_risk_pct."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(
        instrument=inst,
        rr_ratio=5.0,          # R:R muy alto
        kelly_fraction=0.25,
        max_risk_pct=0.01,     # cap en 1%
    )
    risk_pct = sizer.compute_risk_pct(p_win=0.99)
    assert risk_pct <= 0.01 + 1e-9, f"risk_pct no debe superar 0.01, obtuvimos {risk_pct}"
    print(f"OK test_kelly_risk_pct_capped (risk_pct={risk_pct:.4f})")


def test_kelly_sizing_report_keys():
    """sizing_report() debe contener todas las claves esperadas."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(instrument=inst, rr_ratio=1.5)
    report = sizer.sizing_report(p_win=0.60, equity=50000, atr=15.0)
    expected_keys = {
        "p_win", "rr_ratio", "p_breakeven", "edge", "expected_value",
        "f_star", "kelly_fraction_param", "f_kelly", "risk_pct_applied",
        "risk_usd", "atr", "stop_distance_usd_per_unit",
        "n_units_raw", "n_units_final",
    }
    missing = expected_keys - set(report.keys())
    assert len(missing) == 0, f"Claves faltantes en sizing_report: {missing}"
    print(f"OK test_kelly_sizing_report_keys")


def test_kelly_daily_pause():
    """Tras alcanzar daily loss limit, devuelve 0 unidades."""
    import datetime
    inst = MockInstrument()
    sizer = KellyAtrSizer(
        instrument=inst,
        rr_ratio=2.0,
        daily_loss_pct_pause=0.03,
    )
    day = datetime.date(2025, 1, 15)

    # Primera llamada del día: registra equity inicial
    sizer(signal=1, p_win=0.70, current_equity=100_000,
          current_price=4500, current_atr=20, current_date=day)

    # Simulamos que el equity cayó 4% (> 3% límite)
    n = sizer(signal=1, p_win=0.70, current_equity=96_000,
              current_price=4500, current_atr=20, current_date=day)
    assert n == 0.0, f"Debe pausar tras daily loss, obtuvimos {n}"
    print("OK test_kelly_daily_pause")


def test_kelly_higher_p_gives_more_units():
    """Mayor confianza del modelo -> más unidades (monotonicidad)."""
    inst = MockInstrument()
    sizer = KellyAtrSizer(instrument=inst, rr_ratio=1.5, kelly_fraction=0.25)
    params = dict(signal=1, current_equity=100_000, current_price=4500, current_atr=20)
    n_low = sizer(p_win=0.45, **params)
    n_mid = sizer(p_win=0.60, **params)
    n_high = sizer(p_win=0.80, **params)
    assert n_low <= n_mid <= n_high, (
        f"Monotonicidad rota: p=[0.45,0.60,0.80] -> n=[{n_low},{n_mid},{n_high}]"
    )
    print(f"OK test_kelly_higher_p_gives_more_units "
          f"(p=[0.45,0.60,0.80] -> n=[{n_low},{n_mid},{n_high}])")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("\nEjecutando tests de Kelly fraccional...\n")

    tests = [
        test_kelly_zero_when_ev_negative,
        test_kelly_positive_when_ev_positive,
        test_kelly_exact_breakeven,
        test_kelly_fraction_scales_correctly,
        test_kelly_capped_at_kelly_fraction,
        test_kelly_monotonic_in_p,
        test_breakeven_prob_b1,
        test_breakeven_prob_b2,
        test_expected_value_positive,
        test_expected_value_negative,
        test_estimate_rr_correct,
        test_estimate_rr_fallback_few_trades,
        test_extract_p_win_long,
        test_extract_p_win_short,
        test_kelly_sizer_zero_on_neutral_signal,
        test_kelly_sizer_zero_on_zero_atr,
        test_kelly_sizer_zero_when_low_probability,
        test_kelly_sizer_positive_units_when_positive_ev,
        test_kelly_sizer_short_signal_negative_units,
        test_kelly_rr_from_atr_multiples,
        test_kelly_risk_pct_capped,
        test_kelly_sizing_report_keys,
        test_kelly_daily_pause,
        test_kelly_higher_p_gives_more_units,
    ]

    failures = []
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"  FAIL {t.__name__}: {e}")
            failures.append(t.__name__)
        except Exception as e:
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
            failures.append(t.__name__)

    print()
    if failures:
        print(f"FAILED: {len(failures)} tests fallaron: {failures}")
        sys.exit(1)
    print(f"ALL PASSED: {len(tests)} tests OK")
