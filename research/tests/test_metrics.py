"""
Tests unitarios de métricas avanzadas.

Estos tests validan que las fórmulas implementan correctamente:
- Mertens SE (con y sin asimetría/curtosis)
- Probabilistic Sharpe Ratio
- Deflated Sharpe Ratio
- Bootstrap CI
- Tail risk

USO:
    cd quant_bot
    python -m pytest tests/test_metrics.py -v
    # o sin pytest:
    python tests/test_metrics.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import norm

from metrics.advanced import (
    sharpe_ratio_se_mertens,
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    bootstrap_sharpe_ci,
    value_at_risk,
    conditional_var,
    tail_ratio,
    returns_skewness,
    returns_kurtosis,
    is_oos_degradation,
    compute_advanced_metrics,
)
from metrics.objective import (
    HardConstraints,
    ObjectiveSpec,
    evaluate_objective,
)


def _approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# =====================================================================
# MOMENTOS
# =====================================================================

def test_kurtosis_gaussian_is_three():
    """Distribución gaussiana grande debe tener kurtosis raw ~ 3."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(100_000)
    k = returns_kurtosis(x, excess=False)
    assert abs(k - 3.0) < 0.1, f"Kurtosis raw de gaussiana debe ser ~3, obtuvimos {k}"
    print("✓ test_kurtosis_gaussian_is_three: PASSED")


def test_skewness_symmetric_is_zero():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(100_000)
    s = returns_skewness(x)
    assert abs(s) < 0.05, f"Skew gaussiano debe ser ~0, obtuvimos {s}"
    print("✓ test_skewness_symmetric_is_zero: PASSED")


def test_kurtosis_excess_vs_raw():
    x = np.random.default_rng(2).standard_normal(50_000)
    k_raw = returns_kurtosis(x, excess=False)
    k_exc = returns_kurtosis(x, excess=True)
    assert _approx(k_raw - k_exc, 3.0, tol=1e-9), \
        f"raw - excess debe = 3, obtuvimos {k_raw - k_exc}"
    print("✓ test_kurtosis_excess_vs_raw: PASSED")


# =====================================================================
# MERTENS SE
# =====================================================================

def test_mertens_se_gaussian_collapses_to_classic():
    """
    Bajo gaussiano (skew=0, kurt_raw=3), Mertens SE colapsa a la fórmula clásica:
        SE(SR) = sqrt( (1 + SR²/2) / (n-1) )
    """
    sr = 0.1
    n = 1000
    se_mertens = sharpe_ratio_se_mertens(sr, n, skew=0.0, kurt_raw=3.0)
    se_classic = np.sqrt((1 + 0.5 * sr ** 2) / (n - 1))
    assert _approx(se_mertens, se_classic, tol=1e-10), \
        f"Mertens debería colapsar a clásico bajo gaussiano: " \
        f"{se_mertens} vs {se_classic}"
    print("✓ test_mertens_se_gaussian_collapses_to_classic: PASSED")


def test_mertens_se_negative_skew_increases_se():
    """Skew negativo aumenta el SE (más incertidumbre)."""
    sr = 0.15
    n = 500
    se_neg_skew = sharpe_ratio_se_mertens(sr, n, skew=-1.0, kurt_raw=5.0)
    se_zero_skew = sharpe_ratio_se_mertens(sr, n, skew=0.0, kurt_raw=5.0)
    assert se_neg_skew > se_zero_skew, \
        "Skew negativo debe AUMENTAR SE (cola izquierda fea)"
    print("✓ test_mertens_se_negative_skew_increases_se: PASSED")


# =====================================================================
# PSR
# =====================================================================

def test_psr_zero_sr_is_half():
    """PSR(0) cuando SR_obs = SR_benchmark = 0 → ~0.5 (límite Φ(0))."""
    psr = probabilistic_sharpe_ratio(
        sr=0.0, n=1000, skew=0.0, kurt_raw=3.0, sr_benchmark=0.0
    )
    assert abs(psr - 0.5) < 1e-3, f"PSR(0,0) debe ~0.5, obtuvimos {psr}"
    print("✓ test_psr_zero_sr_is_half: PASSED")


def test_psr_high_sr_high_n_approaches_one():
    """SR claramente positivo + n grande → PSR ~ 1."""
    psr = probabilistic_sharpe_ratio(
        sr=0.2, n=10_000, skew=0.0, kurt_raw=3.0, sr_benchmark=0.0
    )
    assert psr > 0.99, f"PSR alto esperado, obtuvimos {psr}"
    print("✓ test_psr_high_sr_high_n_approaches_one: PASSED")


def test_psr_negative_skew_reduces_psr():
    """Misma SR positiva pero skew negativo → PSR menor (más riesgo de cola)."""
    psr_clean = probabilistic_sharpe_ratio(0.1, 1000, skew=0.0, kurt_raw=3.0)
    psr_dirty = probabilistic_sharpe_ratio(0.1, 1000, skew=-1.5, kurt_raw=8.0)
    assert psr_dirty < psr_clean, \
        f"Skew negativo + kurt alto debería reducir PSR. " \
        f"clean={psr_clean}, dirty={psr_dirty}"
    print("✓ test_psr_negative_skew_reduces_psr: PASSED")


# =====================================================================
# DSR
# =====================================================================

def test_dsr_with_one_trial_equals_psr():
    """Con n_trials=1, DSR colapsa a PSR(0)."""
    dsr = deflated_sharpe_ratio(0.1, 1000, skew=0.0, kurt_raw=3.0, n_trials=1)
    psr = probabilistic_sharpe_ratio(0.1, 1000, skew=0.0, kurt_raw=3.0)
    assert _approx(dsr, psr, tol=1e-9), \
        f"DSR(n_trials=1) debe = PSR(0), obtuvimos {dsr} vs {psr}"
    print("✓ test_dsr_with_one_trial_equals_psr: PASSED")


def test_dsr_decreases_with_more_trials():
    """A más trials probadas, DSR cae (selection bias mayor)."""
    common = dict(sr=0.1, n=1000, skew=0.0, kurt_raw=3.0, sr_trials_std=0.05)
    dsr_few = deflated_sharpe_ratio(**common, n_trials=10)
    dsr_many = deflated_sharpe_ratio(**common, n_trials=10_000)
    assert dsr_many < dsr_few, \
        f"Más trials debe reducir DSR. few={dsr_few}, many={dsr_many}"
    print("✓ test_dsr_decreases_with_more_trials: PASSED")


# =====================================================================
# BOOTSTRAP
# =====================================================================

def test_bootstrap_ci_contains_point_estimate_with_high_prob():
    """El CI bootstrap al 95% debería contener el Sharpe puntual."""
    rng = np.random.default_rng(42)
    # SR anualizado ~ 1 con n=2000 retornos diarios
    r = rng.normal(loc=0.001, scale=0.015, size=2000)
    point_sr_annual = np.sqrt(252) * r.mean() / r.std(ddof=1)
    lo, hi = bootstrap_sharpe_ci(r, periods_per_year=252, n_boot=1000, seed=42)
    assert lo <= point_sr_annual <= hi, \
        f"CI debería contener point: {lo} <= {point_sr_annual} <= {hi}"
    print(f"✓ test_bootstrap_ci_contains_point_estimate_with_high_prob: "
          f"PASSED (SR={point_sr_annual:.3f}, CI=[{lo:.3f}, {hi:.3f}])")


# =====================================================================
# TAIL RISK
# =====================================================================

def test_var_is_negative_for_zero_mean():
    rng = np.random.default_rng(3)
    r = rng.normal(0, 0.01, 5000)
    var = value_at_risk(r, alpha=0.05)
    assert var < 0, f"VaR de retornos centrados debe ser <0, obtuvimos {var}"
    print("✓ test_var_is_negative_for_zero_mean: PASSED")


def test_cvar_worse_than_var():
    rng = np.random.default_rng(4)
    r = rng.normal(0, 0.01, 10_000)
    var = value_at_risk(r, alpha=0.05)
    cvar = conditional_var(r, alpha=0.05)
    assert cvar <= var, \
        f"CVaR debe ser <= VaR (más negativo), obtuvimos cvar={cvar} var={var}"
    print("✓ test_cvar_worse_than_var: PASSED")


def test_tail_ratio_symmetric_is_one():
    rng = np.random.default_rng(5)
    r = rng.normal(0, 0.01, 50_000)
    tr = tail_ratio(r)
    assert 0.85 < tr < 1.15, f"Tail ratio simétrico debe ~1, obtuvimos {tr}"
    print("✓ test_tail_ratio_symmetric_is_one: PASSED")


# =====================================================================
# IS vs OOS
# =====================================================================

def test_haircut_zero_when_equal():
    deg = is_oos_degradation(1.5, 1.5)
    assert _approx(deg["haircut"], 0.0)
    print("✓ test_haircut_zero_when_equal: PASSED")


def test_haircut_50pct_when_halved():
    deg = is_oos_degradation(2.0, 1.0)
    assert _approx(deg["haircut"], 0.5)
    print("✓ test_haircut_50pct_when_halved: PASSED")


# =====================================================================
# COMPUTE ADVANCED METRICS (smoke test)
# =====================================================================

def test_compute_advanced_metrics_smoke():
    rng = np.random.default_rng(6)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    out = compute_advanced_metrics(r, periods_per_year=252, n_trials=1)
    required = {
        "n_returns", "sharpe_annual", "psr", "dsr",
        "skew", "kurtosis_raw", "var_5pct_period", "cvar_5pct_period",
        "tail_ratio", "sharpe_bootstrap_ci_annual",
    }
    assert required.issubset(out.keys()), f"Faltan claves: {required - out.keys()}"
    assert out["n_returns"] == 1000
    print("✓ test_compute_advanced_metrics_smoke: PASSED")


# =====================================================================
# OBJECTIVE
# =====================================================================

def test_objective_rejects_high_dd():
    """Una serie con MaxDD enorme debe ser rechazada por constraints."""
    rng = np.random.default_rng(7)
    # Serie con drawdown horrible: tendencia bajista al final
    r = pd.Series(np.concatenate([
        rng.normal(0.001, 0.005, 500),
        rng.normal(-0.005, 0.005, 500),
    ]))
    spec = ObjectiveSpec(
        primary="dsr",
        constraints=HardConstraints(max_drawdown=0.10, min_n_trades=0,
                                    min_psr=0.0, min_dsr=0.0,
                                    min_sharpe_annual=-10.0),
        n_trials=1,
        periods_per_year=252,
    )
    res = evaluate_objective(r, spec)
    assert res.verdict == "REJECT_CONSTRAINTS", \
        f"Esperaba rechazo por DD, obtuvimos {res.verdict}"
    assert any("MaxDD" in v for v in res.constraint_violations)
    print("✓ test_objective_rejects_high_dd: PASSED")


def test_objective_accepts_clean_series():
    """Serie con SR alto, sin drawdown extremo, debe aceptarse."""
    rng = np.random.default_rng(8)
    r = pd.Series(rng.normal(0.002, 0.005, 2000))  # SR anual ~ 6
    spec = ObjectiveSpec(
        primary="dsr",
        constraints=HardConstraints(
            max_drawdown=0.50, min_skew=-2.0, min_n_trades=0,
            min_psr=0.5, min_dsr=0.5, min_sharpe_annual=0.0,
            max_tail_ratio_inverse=10.0,
        ),
        n_trials=1,
        periods_per_year=252,
    )
    res = evaluate_objective(r, spec)
    assert res.verdict == "ACCEPT", \
        f"Esperaba ACCEPT, obtuvimos {res.verdict}: {res.constraint_violations}"
    print("✓ test_objective_accepts_clean_series: PASSED")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    print("\nEjecutando tests de métricas avanzadas...\n")
    tests = [
        test_kurtosis_gaussian_is_three,
        test_skewness_symmetric_is_zero,
        test_kurtosis_excess_vs_raw,
        test_mertens_se_gaussian_collapses_to_classic,
        test_mertens_se_negative_skew_increases_se,
        test_psr_zero_sr_is_half,
        test_psr_high_sr_high_n_approaches_one,
        test_psr_negative_skew_reduces_psr,
        test_dsr_with_one_trial_equals_psr,
        test_dsr_decreases_with_more_trials,
        test_bootstrap_ci_contains_point_estimate_with_high_prob,
        test_var_is_negative_for_zero_mean,
        test_cvar_worse_than_var,
        test_tail_ratio_symmetric_is_one,
        test_haircut_zero_when_equal,
        test_haircut_50pct_when_halved,
        test_compute_advanced_metrics_smoke,
        test_objective_rejects_high_dd,
        test_objective_accepts_clean_series,
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
