"""
Tests del núcleo torch-free de E2 (arbitraje D · barrido de λ).

Grid, % flat, turnover y la selección del λ óptimo (incluido el caso "todo colapsa
a flat" → el 'no edge' es artefacto de λ agresivos, RC-5). El orquestador que
reentrena el DQN se ejercita en el venv con torch.
"""
import numpy as np
import pytest

from models.drl.e2_lambda_sweep import (
    LambdaPoint,
    LambdaResult,
    flat_fraction,
    lambda_grid,
    select_optimal_lambda,
    turnover,
)


def _res(w_dd, w_vol, w_idle, lb95, flat, sharpe=None, turn=0.3) -> LambdaResult:
    return LambdaResult(
        point=LambdaPoint(w_dd, w_vol, w_idle),
        sharpe_mean=lb95 + 0.2 if sharpe is None else sharpe,
        lb95=lb95, flat_fraction=flat, turnover=turn, n_seeds=20, n_oos_bars=500,
    )


# ---- grid ----

def test_lambda_grid_cartesian():
    g = lambda_grid([1.0, 2.0], [0.5], [0.001, 0.01])
    assert len(g) == 4
    assert LambdaPoint(2.0, 0.5, 0.01) in g


def test_lambda_grid_empty_axis_raises():
    with pytest.raises(ValueError):
        lambda_grid([], [0.5], [0.001])


# ---- métricas de actividad ----

def test_flat_fraction():
    assert flat_fraction(np.array([0.0, 1.0, 0.0, -1.0])) == 0.5
    assert flat_fraction(np.zeros(10)) == 1.0


def test_turnover_counts_entry_and_flips():
    # 0->+1 (1), +1->+1 (0), +1->-1 (2), -1->0 (1) => media (1+0+2+1)/4 = 1.0
    assert turnover(np.array([1.0, 1.0, -1.0, 0.0])) == pytest.approx(1.0)


# ---- selección del λ óptimo ----

def test_selects_max_lb95_among_non_flat():
    results = [
        _res(2.0, 0.5, 0.001, lb95=0.10, flat=0.40),
        _res(1.0, 0.2, 0.001, lb95=0.35, flat=0.50),   # mejor LB95, no colapsa
        _res(5.0, 2.0, 0.010, lb95=0.50, flat=0.95),   # mejor LB95 pero colapsa a flat
    ]
    v = select_optimal_lambda(results, max_flat=0.90)
    assert not v.all_flat_collapsed
    assert v.best.point == LambdaPoint(1.0, 0.2, 0.001)   # excluye el colapsado


def test_all_flat_collapse_flagged():
    results = [
        _res(5.0, 2.0, 0.01, lb95=0.4, flat=0.97),
        _res(4.0, 1.5, 0.02, lb95=0.5, flat=0.99),
    ]
    v = select_optimal_lambda(results, max_flat=0.90)
    assert v.all_flat_collapsed
    assert v.best is None
    assert "artefacto" in v.reason.lower()


def test_select_prefers_lb95_over_mean():
    results = [
        _res(1.0, 0.5, 0.001, lb95=-0.1, flat=0.3, sharpe=1.5),  # media alta, cota mala
        _res(2.0, 0.5, 0.001, lb95=0.25, flat=0.3, sharpe=0.6),  # cota buena
    ]
    v = select_optimal_lambda(results, max_flat=0.90)
    assert v.best.point == LambdaPoint(2.0, 0.5, 0.001)


def test_empty_results_raises():
    with pytest.raises(ValueError):
        select_optimal_lambda([])
