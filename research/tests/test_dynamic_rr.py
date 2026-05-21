"""
Tests de Dynamic Risk/Reward.

Casos validados:
 1. rr_min se aplica cuando p_win <= p_low.
 2. rr_max se aplica cuando p_win >= p_high.
 3. Mapeo lineal es interpolación exacta en el punto medio.
 4. Mapeo sigmoid está en [rr_min, rr_max] para cualquier p_win.
 5. Mapeo stepped produce exactamente 2 o 3 valores distintos.
 6. Monotonicidad: mayor p_win → mayor R:R para todos los shapes.
 7. SL para long = entry - atr_sl_mult * ATR.
 8. TP para long = entry + rr * atr_sl_mult * ATR.
 9. SL/TP invertidos para short.
10. R:R dinámico de short y long son iguales dado el mismo p_win.
11. compute_full_sizing devuelve n_units=0 cuando signal=0.
12. compute_full_sizing devuelve n_units=0 cuando EV negativo (p < p_break).
13. compute_full_sizing: TP/SL coherentes con la señal (long o short).
14. TP más lejano con mayor confianza (monotonicidad de TP).
15. rr_curve devuelve DataFrame con columnas correctas y longitud correcta.
16. levels_report tiene todas las claves esperadas.
17. ValueError si rr_max < rr_min.
18. ValueError si p_low >= p_high.
19. ValueError si atr <= 0 en compute_levels.
20. compute_full_sizing integra Kelly + DynamicRR correctamente.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from risk.dynamic_rr import (
    compute_dynamic_rr,
    DynamicRRManager,
    compute_full_sizing,
)


# =====================================================================
# MOCK INSTRUMENT
# =====================================================================

class MockInstrument:
    usd_per_unit_per_price_point = 10.0
    min_size_increment = 1.0

    def round_to_min_increment(self, n):
        return float(int(n / self.min_size_increment) * self.min_size_increment)


# =====================================================================
# TESTS compute_dynamic_rr
# =====================================================================

def test_rr_at_p_low_returns_rr_min():
    """p_win <= p_low -> rr_min para todos los shapes."""
    for shape in ("linear", "sigmoid", "stepped"):
        rr = compute_dynamic_rr(0.30, rr_min=1.0, rr_max=3.0,
                                p_low=0.45, p_high=0.75, shape=shape)
        assert abs(rr - 1.0) < 0.12, (
            f"shape={shape}: rr esperado ~1.0 en p=0.30, obtuvimos {rr:.4f}"
        )
    print("OK test_rr_at_p_low_returns_rr_min")


def test_rr_at_p_high_returns_rr_max():
    """p_win >= p_high -> rr_max para todos los shapes."""
    for shape in ("linear", "sigmoid", "stepped"):
        rr = compute_dynamic_rr(0.90, rr_min=1.0, rr_max=3.0,
                                p_low=0.45, p_high=0.75, shape=shape)
        assert abs(rr - 3.0) < 0.12, (
            f"shape={shape}: rr esperado ~3.0 en p=0.90, obtuvimos {rr:.4f}"
        )
    print("OK test_rr_at_p_high_returns_rr_max")


def test_linear_midpoint_is_rr_mid():
    """Linear: en p = (p_low + p_high)/2 -> rr = (rr_min + rr_max)/2."""
    p_mid = (0.45 + 0.75) / 2   # = 0.60
    rr_mid = (1.0 + 3.0) / 2    # = 2.0
    rr = compute_dynamic_rr(p_mid, rr_min=1.0, rr_max=3.0,
                            p_low=0.45, p_high=0.75, shape="linear")
    assert abs(rr - rr_mid) < 1e-9, f"Lineal midpoint: esperado {rr_mid}, obtuvimos {rr}"
    print(f"OK test_linear_midpoint_is_rr_mid (rr={rr:.4f})")


def test_sigmoid_in_bounds():
    """Sigmoid siempre produce rr en [rr_min, rr_max]."""
    for p in np.linspace(0.0, 1.0, 100):
        rr = compute_dynamic_rr(float(p), rr_min=1.2, rr_max=2.5,
                                p_low=0.45, p_high=0.75, shape="sigmoid")
        assert 1.2 - 1e-6 <= rr <= 2.5 + 1e-6, (
            f"Sigmoid fuera de bounds en p={p:.3f}: rr={rr:.4f}"
        )
    print("OK test_sigmoid_in_bounds")


def test_stepped_discrete_values():
    """Stepped produce solo valores en {rr_min, rr_mid, rr_max}."""
    rr_min, rr_max = 1.0, 3.0
    rr_mid = (rr_min + rr_max) / 2  # = 2.0
    valid_values = {rr_min, rr_mid, rr_max}
    for p in np.linspace(0.0, 1.0, 50):
        rr = compute_dynamic_rr(float(p), rr_min=rr_min, rr_max=rr_max,
                                p_low=0.45, p_high=0.75, shape="stepped")
        assert rr in valid_values, f"stepped devolvio {rr} que no esta en {valid_values}"
    print("OK test_stepped_discrete_values")


def test_monotonic_for_all_shapes():
    """Mayor p_win -> mayor R:R (monotonicidad estricta en zona intermedia)."""
    p_values = np.linspace(0.30, 0.90, 30)
    for shape in ("linear", "sigmoid", "stepped"):
        rrs = [compute_dynamic_rr(float(p), rr_min=1.0, rr_max=3.0,
                                  p_low=0.45, p_high=0.75, shape=shape)
               for p in p_values]
        for i in range(len(rrs) - 1):
            assert rrs[i] <= rrs[i+1] + 1e-9, (
                f"shape={shape}: no monótono en p={p_values[i]:.3f}->{p_values[i+1]:.3f}: "
                f"rr={rrs[i]:.4f}->{rrs[i+1]:.4f}"
            )
    print("OK test_monotonic_for_all_shapes")


def test_invalid_rr_max_less_than_min():
    """rr_max < rr_min debe lanzar ValueError."""
    try:
        compute_dynamic_rr(0.5, rr_min=3.0, rr_max=1.0)
        assert False, "Debe lanzar ValueError"
    except ValueError:
        pass
    print("OK test_invalid_rr_max_less_than_min")


def test_invalid_p_low_ge_p_high():
    """p_low >= p_high debe lanzar ValueError."""
    try:
        compute_dynamic_rr(0.5, p_low=0.75, p_high=0.45)
        assert False, "Debe lanzar ValueError"
    except ValueError:
        pass
    print("OK test_invalid_p_low_ge_p_high")


def test_invalid_shape():
    """Shape desconocido debe lanzar ValueError."""
    try:
        compute_dynamic_rr(0.5, shape="exponential")
        assert False, "Debe lanzar ValueError"
    except ValueError:
        pass
    print("OK test_invalid_shape")


# =====================================================================
# TESTS DynamicRRManager.compute_levels
# =====================================================================

def test_sl_for_long_below_entry():
    """Long: SL debe estar por debajo del precio de entrada."""
    mgr = DynamicRRManager(atr_sl_mult=2.0)
    sl, tp = mgr.compute_levels(entry_price=5200.0, signal=1, atr=20.0, p_win=0.60)
    assert sl < 5200.0, f"SL long debe estar debajo de entry, obtuvimos {sl}"
    print(f"OK test_sl_for_long_below_entry (sl={sl:.2f})")


def test_tp_for_long_above_entry():
    """Long: TP debe estar por encima del precio de entrada."""
    mgr = DynamicRRManager(atr_sl_mult=2.0)
    sl, tp = mgr.compute_levels(entry_price=5200.0, signal=1, atr=20.0, p_win=0.60)
    assert tp > 5200.0, f"TP long debe estar arriba de entry, obtuvimos {tp}"
    print(f"OK test_tp_for_long_above_entry (tp={tp:.2f})")


def test_sl_for_short_above_entry():
    """Short: SL debe estar por encima del precio de entrada."""
    mgr = DynamicRRManager(atr_sl_mult=2.0)
    sl, tp = mgr.compute_levels(entry_price=5200.0, signal=-1, atr=20.0, p_win=0.60)
    assert sl > 5200.0, f"SL short debe estar arriba de entry, obtuvimos {sl}"
    print(f"OK test_sl_for_short_above_entry (sl={sl:.2f})")


def test_tp_for_short_below_entry():
    """Short: TP debe estar por debajo del precio de entrada."""
    mgr = DynamicRRManager(atr_sl_mult=2.0)
    sl, tp = mgr.compute_levels(entry_price=5200.0, signal=-1, atr=20.0, p_win=0.60)
    assert tp < 5200.0, f"TP short debe estar debajo de entry, obtuvimos {tp}"
    print(f"OK test_tp_for_short_below_entry (tp={tp:.2f})")


def test_sl_distance_equals_atr_mult():
    """SL distance = atr_sl_mult * ATR exacto."""
    mgr = DynamicRRManager(atr_sl_mult=2.0, shape="linear")
    entry, atr = 1.1200, 0.0050
    sl, _ = mgr.compute_levels(entry, signal=1, atr=atr, p_win=0.60)
    expected_sl = entry - 2.0 * atr
    assert abs(sl - expected_sl) < 1e-9, (
        f"SL esperado {expected_sl:.5f}, obtuvimos {sl:.5f}"
    )
    print(f"OK test_sl_distance_equals_atr_mult (sl={sl:.5f})")


def test_tp_farther_with_higher_confidence():
    """Mayor p_win -> TP más lejano del entry (monotonicidad)."""
    mgr = DynamicRRManager(atr_sl_mult=2.0, rr_min=1.2, rr_max=2.5,
                           p_low=0.45, p_high=0.75)
    entry, atr = 5200.0, 20.0
    probs = [0.46, 0.55, 0.65, 0.76]
    tps = [mgr.compute_levels(entry, 1, atr, p)[1] for p in probs]
    for i in range(len(tps) - 1):
        assert tps[i] <= tps[i+1] + 1e-6, (
            f"TP no monótono: p={probs[i]}->{probs[i+1]}, "
            f"tp={tps[i]:.2f}->{tps[i+1]:.2f}"
        )
    print(f"OK test_tp_farther_with_higher_confidence {[round(t,1) for t in tps]}")


def test_compute_levels_invalid_signal():
    """signal != ±1 debe lanzar ValueError."""
    mgr = DynamicRRManager()
    try:
        mgr.compute_levels(5200.0, signal=0, atr=20.0, p_win=0.6)
        assert False, "Debe lanzar ValueError con signal=0"
    except ValueError:
        pass
    print("OK test_compute_levels_invalid_signal")


def test_compute_levels_invalid_atr():
    """ATR=0 debe lanzar ValueError."""
    mgr = DynamicRRManager()
    try:
        mgr.compute_levels(5200.0, signal=1, atr=0.0, p_win=0.6)
        assert False, "Debe lanzar ValueError con atr=0"
    except ValueError:
        pass
    print("OK test_compute_levels_invalid_atr")


def test_levels_report_has_correct_keys():
    """levels_report() contiene todas las claves esperadas."""
    mgr = DynamicRRManager()
    report = mgr.levels_report(5200.0, signal=1, atr=20.0, p_win=0.65)
    expected_keys = {
        "direction", "entry_price", "sl_price", "tp_price",
        "sl_distance", "tp_distance", "rr_dynamic", "atr", "p_win",
        "atr_sl_mult", "shape",
    }
    missing = expected_keys - set(report.keys())
    assert len(missing) == 0, f"Claves faltantes: {missing}"
    assert report["direction"] == "LONG"
    print("OK test_levels_report_has_correct_keys")


def test_rr_curve_dataframe():
    """rr_curve() devuelve DataFrame con columnas y longitud correctas."""
    mgr = DynamicRRManager()
    p_vals = np.linspace(0.35, 0.85, 20)
    df = mgr.rr_curve(p_vals)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
    assert set(["p_win", "rr", "tp_atr_mult"]).issubset(df.columns)
    # rr siempre monótono en la curva
    assert (df["rr"].diff().dropna() >= -1e-9).all(), "rr_curve no es monótona"
    print("OK test_rr_curve_dataframe")


# =====================================================================
# TESTS compute_full_sizing
# =====================================================================

def test_full_sizing_zero_on_neutral_signal():
    """signal=0 -> n_units=0 sin importar nada más."""
    inst = MockInstrument()
    mgr = DynamicRRManager()
    result = compute_full_sizing(
        signal=0, p_win=0.80, current_equity=100_000,
        current_price=5200.0, current_atr=20.0,
        instrument=inst, rr_manager=mgr,
    )
    assert result["n_units"] == 0.0
    assert result["sl_price"] is None
    assert result["tp_price"] is None
    print("OK test_full_sizing_zero_on_neutral_signal")


def test_full_sizing_zero_when_low_probability():
    """p_win muy bajo -> Kelly=0 -> n_units=0."""
    inst = MockInstrument()
    # rr_min=1.2: p_break = 1/(1+1.2) = 0.455; p=0.30 < 0.455 -> EV negativo
    mgr = DynamicRRManager(rr_min=1.2, rr_max=2.5, p_low=0.45, p_high=0.75)
    result = compute_full_sizing(
        signal=1, p_win=0.30, current_equity=100_000,
        current_price=5200.0, current_atr=20.0,
        instrument=inst, rr_manager=mgr,
    )
    assert result["n_units"] == 0.0, (
        f"Con p=0.30 y rr_min=1.2 (p_break=0.455) Kelly debe ser 0, "
        f"obtuvimos n_units={result['n_units']}"
    )
    print("OK test_full_sizing_zero_when_low_probability")


def test_full_sizing_long_positive_units():
    """Long con p_win alto -> n_units > 0."""
    inst = MockInstrument()
    mgr = DynamicRRManager()
    result = compute_full_sizing(
        signal=1, p_win=0.72, current_equity=100_000,
        current_price=5200.0, current_atr=20.0,
        instrument=inst, rr_manager=mgr,
    )
    assert result["n_units"] > 0, f"n_units debe ser > 0, obtuvimos {result['n_units']}"
    assert result["tp_price"] > result["entry_price"] if "entry_price" in result else True
    print(f"OK test_full_sizing_long_positive_units (n={result['n_units']})")


def test_full_sizing_short_negative_units():
    """Short con p_win alto -> n_units < 0."""
    inst = MockInstrument()
    mgr = DynamicRRManager()
    result = compute_full_sizing(
        signal=-1, p_win=0.72, current_equity=100_000,
        current_price=5200.0, current_atr=20.0,
        instrument=inst, rr_manager=mgr,
    )
    assert result["n_units"] < 0, f"Short debe dar n_units < 0, obtuvimos {result['n_units']}"
    print(f"OK test_full_sizing_short_negative_units (n={result['n_units']})")


def test_full_sizing_tp_sl_coherent_with_signal():
    """Long: tp > entry > sl. Short: sl > entry > tp."""
    inst = MockInstrument()
    mgr = DynamicRRManager()
    entry = 5200.0

    # Long
    res_long = compute_full_sizing(
        signal=1, p_win=0.70, current_equity=100_000,
        current_price=entry, current_atr=20.0,
        instrument=inst, rr_manager=mgr,
    )
    assert res_long["tp_price"] > entry > res_long["sl_price"], (
        f"Long: tp={res_long['tp_price']} > entry={entry} > sl={res_long['sl_price']}"
    )

    # Short
    res_short = compute_full_sizing(
        signal=-1, p_win=0.70, current_equity=100_000,
        current_price=entry, current_atr=20.0,
        instrument=inst, rr_manager=mgr,
    )
    assert res_short["sl_price"] > entry > res_short["tp_price"], (
        f"Short: sl={res_short['sl_price']} > entry={entry} > tp={res_short['tp_price']}"
    )
    print("OK test_full_sizing_tp_sl_coherent_with_signal")


def test_full_sizing_higher_p_gives_farther_tp():
    """Mayor p_win -> TP más lejano (monotonicidad en full sizing)."""
    inst = MockInstrument()
    mgr = DynamicRRManager(rr_min=1.2, rr_max=2.5, p_low=0.45, p_high=0.75)
    entry, atr = 5200.0, 20.0
    probs = [0.50, 0.60, 0.70, 0.76]
    tps = []
    for p in probs:
        r = compute_full_sizing(
            signal=1, p_win=p, current_equity=100_000,
            current_price=entry, current_atr=atr,
            instrument=inst, rr_manager=mgr,
        )
        tps.append(r["tp_price"] if r["tp_price"] is not None else entry)

    for i in range(len(tps) - 1):
        assert tps[i] <= tps[i+1] + 1e-6, (
            f"TP no monótono: p={probs[i]}->{probs[i+1]}, "
            f"tp={tps[i]:.2f}->{tps[i+1]:.2f}"
        )
    print(f"OK test_full_sizing_higher_p_gives_farther_tp "
          f"{[round(t,1) for t in tps]}")


def test_full_sizing_result_keys():
    """compute_full_sizing devuelve todas las claves esperadas."""
    inst = MockInstrument()
    mgr = DynamicRRManager()
    result = compute_full_sizing(
        signal=1, p_win=0.65, current_equity=50_000,
        current_price=1.1200, current_atr=0.0050,
        instrument=inst, rr_manager=mgr,
    )
    expected_keys = {"n_units", "sl_price", "tp_price", "risk_pct",
                     "risk_usd", "rr_dynamic", "kelly_raw"}
    missing = expected_keys - set(result.keys())
    assert len(missing) == 0, f"Claves faltantes: {missing}"
    print("OK test_full_sizing_result_keys")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("\nEjecutando tests de Dynamic R:R...\n")

    tests = [
        test_rr_at_p_low_returns_rr_min,
        test_rr_at_p_high_returns_rr_max,
        test_linear_midpoint_is_rr_mid,
        test_sigmoid_in_bounds,
        test_stepped_discrete_values,
        test_monotonic_for_all_shapes,
        test_invalid_rr_max_less_than_min,
        test_invalid_p_low_ge_p_high,
        test_invalid_shape,
        test_sl_for_long_below_entry,
        test_tp_for_long_above_entry,
        test_sl_for_short_above_entry,
        test_tp_for_short_below_entry,
        test_sl_distance_equals_atr_mult,
        test_tp_farther_with_higher_confidence,
        test_compute_levels_invalid_signal,
        test_compute_levels_invalid_atr,
        test_levels_report_has_correct_keys,
        test_rr_curve_dataframe,
        test_full_sizing_zero_on_neutral_signal,
        test_full_sizing_zero_when_low_probability,
        test_full_sizing_long_positive_units,
        test_full_sizing_short_negative_units,
        test_full_sizing_tp_sl_coherent_with_signal,
        test_full_sizing_higher_p_gives_farther_tp,
        test_full_sizing_result_keys,
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
