"""
Tests del núcleo torch-free de E1 (arbitraje D · gate pre-registrado).

Cubren lo que NO necesita torch/xgboost: reglas deterministas, intervalos de
confianza del Sharpe y el veredicto del gate pre-registrado (Rama A / A' / B).
Los runners de DQN/XGBoost se ejercitan aparte en el venv con esas dependencias.
"""
import numpy as np
import pytest

from models.drl.e1_baseline_comparison import (
    E1Verdict,
    ModelSharpeResult,
    _ann_sharpe,
    block_bootstrap_sharpe_ci,
    evaluate_e1_decision,
    meanrev_positions,
    momentum_positions,
    sharpe_ci_from_seeds,
)


def _result(name: str, lb95: float, mean: float | None = None) -> ModelSharpeResult:
    mean = lb95 + 0.3 if mean is None else mean
    return ModelSharpeResult(
        name=name, sharpe_by_seed=np.array([mean]), mean=mean, lb95=lb95,
        p95=mean + 0.3, n_seeds=20, n_oos_bars=500, ci_method="seeds",
    )


# ---- reglas deterministas ----

def test_momentum_long_in_uptrend():
    closes = np.linspace(100, 200, 60)          # tendencia alcista clara
    pos = momentum_positions(closes, lookback=20)
    assert pos[:20].tolist() == [0.0] * 20      # warm-up flat (sin look-ahead)
    assert np.all(pos[25:] == 1.0)              # largo en tendencia


def test_momentum_short_in_downtrend():
    closes = np.linspace(200, 100, 60)
    pos = momentum_positions(closes, lookback=20)
    assert np.all(pos[25:] == -1.0)


def test_meanrev_contrarian_on_spike():
    closes = np.concatenate([np.full(40, 100.0), np.array([130.0, 131.0])])
    pos = meanrev_positions(closes, lookback=20, z_entry=1.0)
    assert pos[-1] < 0.0                        # precio muy por encima → short


# ---- Sharpe + IC ----

def test_ann_sharpe_basic():
    assert np.isnan(_ann_sharpe(np.array([0.01])))     # < 2 puntos
    assert _ann_sharpe(np.zeros(10)) == 0.0            # vol nula → 0
    assert _ann_sharpe(np.array([0.01, 0.02, 0.015, 0.005])) > 0.0


def test_sharpe_ci_from_seeds_ordering():
    sharpes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mean, lb, p95 = sharpe_ci_from_seeds(sharpes)
    assert lb <= mean <= p95
    assert lb == pytest.approx(np.percentile(sharpes, 5))


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, size=600)         # deriva positiva pequeña
    mean, lb, p95, dist = block_bootstrap_sharpe_ci(returns, n_boot=500, seed=1)
    assert lb <= mean <= p95
    assert np.isfinite(lb) and np.isfinite(p95)
    assert len(dist) > 0


# ---- veredicto del gate pre-registrado ----

def test_verdict_branch_A_strong():
    res = {"dqn": _result("dqn", lb95=0.35), "xgboost": _result("xgboost", lb95=0.10)}
    v = evaluate_e1_decision(res, materiality=0.20)
    assert v.branch == "A" and v.best_model == "dqn" and not v.directional_falsified


def test_verdict_branch_A_prime_marginal():
    res = {"xgboost": _result("xgboost", lb95=0.12), "momentum": _result("momentum", lb95=0.05)}
    v = evaluate_e1_decision(res, materiality=0.20)
    assert v.branch == "A_prime" and v.best_model == "xgboost"


def test_verdict_branch_B_falsified():
    res = {"dqn": _result("dqn", lb95=-0.10), "momentum": _result("momentum", lb95=-0.02)}
    v = evaluate_e1_decision(res, materiality=0.20)
    assert v.branch == "B" and v.directional_falsified is True


def test_verdict_picks_best_by_lb95_not_mean():
    # 'risky' tiene mejor media pero peor cota inferior → no debe ganar.
    risky = ModelSharpeResult("risky", np.array([1.5]), mean=1.5, lb95=-0.3, p95=2.0,
                              n_seeds=20, n_oos_bars=500, ci_method="seeds")
    steady = ModelSharpeResult("steady", np.array([0.6]), mean=0.6, lb95=0.30, p95=0.9,
                               n_seeds=20, n_oos_bars=500, ci_method="seeds")
    v = evaluate_e1_decision({"risky": risky, "steady": steady}, materiality=0.20)
    assert v.best_model == "steady" and v.branch == "A"


def test_empty_results_raises():
    with pytest.raises(ValueError):
        evaluate_e1_decision({})
