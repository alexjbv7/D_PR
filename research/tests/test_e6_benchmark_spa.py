"""
Tests del núcleo torch-free de E6 (arbitraje D · data-snooping + benchmark).

Reality Check de White, su variante studentizada y la sensibilidad al benchmark.
Las series de benchmark reales (buy-and-hold/neutral) las produce el gate.
"""
import numpy as np
import pytest

from models.drl.e6_benchmark_spa import (
    benchmark_sensitivity,
    reality_check_pvalue,
    stationary_bootstrap_indices,
    studentized_reality_check_pvalue,
)


# ---- stationary bootstrap ----

def test_stationary_bootstrap_valid_indices():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(100, 5.0, rng)
    assert idx.shape == (100,)
    assert idx.min() >= 0 and idx.max() < 100


# ---- benchmark sensitivity ----

def test_benchmark_sensitivity_beats_zero():
    rng = np.random.default_rng(1)
    model = rng.normal(0.001, 0.01, 500)              # deriva positiva
    benches = {"zero": np.zeros(500), "buy_and_hold": rng.normal(0.002, 0.01, 500)}
    sens = benchmark_sensitivity(model, benches)
    assert sens["zero"].model_beats                   # bate al risk-free
    assert sens["zero"].sharpe_diff == pytest.approx(sens["zero"].sharpe_model, abs=1e-9)


def test_benchmark_sensitivity_shape_mismatch_raises():
    with pytest.raises(ValueError):
        benchmark_sensitivity(np.zeros(100), {"zero": np.zeros(50)})


# ---- Reality Check ----

def _perf_with_one_winner(n=600, L=10, edge=0.0015, seed=3):
    rng = np.random.default_rng(seed)
    perf = rng.normal(0.0, 0.01, size=(n, L))          # L configs nulas
    perf[:, 0] += edge                                 # config 0 con edge real
    return perf


def test_rc_rejects_when_real_edge_exists():
    perf = _perf_with_one_winner(edge=0.002)
    p = reality_check_pvalue(perf, n_boot=500, seed=0)
    assert p < 0.05                                    # detecta la config ganadora


def test_rc_does_not_reject_under_null():
    rng = np.random.default_rng(7)
    perf = rng.normal(0.0, 0.01, size=(600, 10))       # todas nulas
    p = reality_check_pvalue(perf, n_boot=500, seed=0)
    assert p > 0.10                                    # no falso positivo


def test_rc_pvalue_monotone_in_edge():
    weak = reality_check_pvalue(_perf_with_one_winner(edge=0.0005), n_boot=500, seed=0)
    strong = reality_check_pvalue(_perf_with_one_winner(edge=0.003), n_boot=500, seed=0)
    assert strong <= weak


# ---- studentized RC (estilo SPA) ----

def test_studentized_rc_rejects_real_edge():
    perf = _perf_with_one_winner(edge=0.002)
    res = studentized_reality_check_pvalue(perf, n_boot=500, seed=0)
    assert res["p_value"] < 0.05
    assert res["n_configs"] == 10
    assert np.isfinite(res["t_stat"])


def test_studentized_rc_null_not_rejected():
    rng = np.random.default_rng(11)
    perf = rng.normal(0.0, 0.01, size=(600, 10))
    res = studentized_reality_check_pvalue(perf, n_boot=500, seed=0)
    assert res["p_value"] > 0.10
