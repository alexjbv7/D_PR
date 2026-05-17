"""
Tests del Walk-Forward Runner integrado.

Casos validados:
 1. PSR = 1.0 con retornos perfectamente positivos.
 2. PSR ~ 0.5 con retornos aleatorios (sin señal).
 3. PSR < 0.5 con retornos negativos.
 4. DSR <= PSR (DSR penaliza por múltiples pruebas).
 5. compute_oos_metrics devuelve claves correctas.
 6. WalkForwardRunner.run completa sin error en datos sintéticos.
 7. OOS signals no contiene índices del train (no leakage).
 8. OOS periods no se solapan entre folds.
 9. Número de folds es el esperado según config.
10. oos_signals solo contiene valores en {-1, 0, +1}.
11. oos_proba suma a 1 por fila.
12. oos_sizing tiene las columnas correctas.
13. summary() produce string no vacío.
14. feature_importance_agg no está vacío si track_importance=True.
15. Calibración report tiene verdict en {OK, MARGINAL, POOR}.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.walk_forward_runner import (
    probabilistic_sharpe_ratio,
    deflated_sharpe_ratio,
    compute_oos_metrics,
    WalkForwardConfig,
    WalkForwardRunner,
)


# =====================================================================
# HELPERS
# =====================================================================

def _make_dataset(n=600, n_features=8, seed=42):
    """Dataset sintético con señal real en feat_0."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        index=dates,
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(
        np.where(X["feat_0"] > 0.6, 1, np.where(X["feat_0"] < -0.6, -1, 0)),
        index=dates,
        name="label",
    )
    prices = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n))),
        index=dates,
        name="close",
    )
    return X, y, prices


def _make_small_runner():
    """Runner con configuración mínima para tests rápidos."""
    cfg = WalkForwardConfig(
        train_size=200,
        test_size=60,
        embargo=3,
        calib_frac=0.20,
        calib_method="sigmoid",
        track_importance=True,
        shap_sample_size=50,
        xgb_params={"n_estimators": 30, "max_depth": 3},
    )
    return WalkForwardRunner(cfg)


# =====================================================================
# TESTS PSR / DSR
# =====================================================================

def test_psr_perfect_returns():
    """Retornos siempre positivos -> PSR cerca de 1."""
    rng = np.random.default_rng(0)
    returns = rng.normal(loc=0.005, scale=0.001, size=252)
    psr = probabilistic_sharpe_ratio(returns)
    assert psr > 0.99, f"PSR esperado > 0.99 con retornos perfectos, obtuvimos {psr:.4f}"
    print(f"OK test_psr_perfect_returns (PSR={psr:.4f})")


def test_psr_random_returns():
    """Retornos sin señal -> PSR no debe ser confidentemente alto (< 0.95)."""
    # Promediamos sobre múltiples seeds para evitar falsos positivos por varianza muestral
    psrs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        returns = rng.normal(loc=0.0, scale=0.01, size=252)
        psrs.append(probabilistic_sharpe_ratio(returns))
    mean_psr = np.mean(psrs)
    # Con retornos N(0, σ), en promedio PSR ~ 0.5 (puede variar mucho en una sola seed)
    assert mean_psr < 0.85, f"PSR medio sin señal no debe ser alto: {mean_psr:.4f}"
    assert mean_psr > 0.05, f"PSR medio sin señal no debe ser cercano a 0: {mean_psr:.4f}"
    print(f"OK test_psr_random_returns (PSR_medio={mean_psr:.4f})")


def test_psr_negative_returns():
    """Retornos negativos -> PSR < 0.5."""
    rng = np.random.default_rng(2)
    returns = rng.normal(loc=-0.005, scale=0.01, size=252)
    psr = probabilistic_sharpe_ratio(returns)
    assert psr < 0.5, f"PSR con retornos negativos debe ser < 0.5, obtuvimos {psr:.4f}"
    print(f"OK test_psr_negative_returns (PSR={psr:.4f})")


def test_dsr_leq_psr():
    """DSR <= PSR: DSR penaliza por múltiples pruebas."""
    rng = np.random.default_rng(3)
    returns = rng.normal(loc=0.002, scale=0.01, size=252)
    psr = probabilistic_sharpe_ratio(returns)
    dsr = deflated_sharpe_ratio(returns, n_trials=10)
    assert dsr <= psr + 1e-9, f"DSR ({dsr:.4f}) debe ser <= PSR ({psr:.4f})"
    print(f"OK test_dsr_leq_psr (PSR={psr:.4f}, DSR={dsr:.4f})")


def test_dsr_decreases_with_more_trials():
    """Más trials -> DSR más bajo (penalización mayor)."""
    rng = np.random.default_rng(4)
    returns = rng.normal(loc=0.003, scale=0.01, size=252)
    dsr_1 = deflated_sharpe_ratio(returns, n_trials=1)
    dsr_5 = deflated_sharpe_ratio(returns, n_trials=5)
    dsr_20 = deflated_sharpe_ratio(returns, n_trials=20)
    assert dsr_1 >= dsr_5 >= dsr_20, (
        f"DSR debe decrecer con trials: {dsr_1:.4f} >= {dsr_5:.4f} >= {dsr_20:.4f}"
    )
    print(f"OK test_dsr_decreases_with_more_trials "
          f"(1={dsr_1:.4f}, 5={dsr_5:.4f}, 20={dsr_20:.4f})")


def test_psr_few_samples_returns_zero():
    """Con menos de 4 muestras, PSR debe ser 0."""
    returns = np.array([0.01, 0.02, 0.01])
    psr = probabilistic_sharpe_ratio(returns)
    assert psr == 0.0, f"PSR con < 4 muestras debe ser 0, obtuvimos {psr}"
    print("OK test_psr_few_samples_returns_zero")


# =====================================================================
# TESTS compute_oos_metrics
# =====================================================================

def test_oos_metrics_keys():
    """compute_oos_metrics devuelve todas las claves esperadas."""
    rng = np.random.default_rng(5)
    n = 100
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    signals = pd.Series(rng.choice([-1, 0, 1], size=n), index=dates)
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)
    metrics = compute_oos_metrics(signals, prices)
    expected = {"sharpe", "psr", "n_trades", "coverage", "max_drawdown"}
    missing = expected - set(metrics.keys())
    assert len(missing) == 0, f"Claves faltantes: {missing}"
    print("OK test_oos_metrics_keys")


def test_oos_metrics_no_trades():
    """Sin trades (señal siempre 0) -> n_trades=0."""
    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    signals = pd.Series(0, index=dates)
    prices = pd.Series(range(100, 150), index=dates, dtype=float)
    metrics = compute_oos_metrics(signals, prices)
    assert metrics["n_trades"] == 0
    assert metrics["coverage"] == 0.0
    print("OK test_oos_metrics_no_trades")


# =====================================================================
# TESTS WALK-FORWARD RUNNER
# =====================================================================

def test_runner_completes_without_error():
    """WalkForwardRunner.run completa en datos sintéticos."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    assert result is not None
    print("OK test_runner_completes_without_error")


def test_runner_n_folds_correct():
    """Número de folds correcto según config."""
    X, y, prices = _make_dataset(n=600)
    cfg = WalkForwardConfig(
        train_size=200, test_size=60, embargo=3,
        xgb_params={"n_estimators": 20, "max_depth": 2},
    )
    runner = WalkForwardRunner(cfg)
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    # Con n=600, train=200, test=60, embargo=3: aprox (600-200-3-60)//60 + 1 = 5 folds
    from models.validation import WalkForwardSplitter
    splitter = WalkForwardSplitter(200, 60, embargo=3)
    expected_folds = splitter.get_n_splits(X)
    assert len(result.fold_results) == expected_folds, (
        f"Esperado {expected_folds} folds, obtuvimos {len(result.fold_results)}"
    )
    print(f"OK test_runner_n_folds_correct ({len(result.fold_results)} folds)")


def test_oos_signals_valid_values():
    """oos_signals solo contiene {-1, 0, +1}."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    unique_vals = set(result.oos_signals.unique())
    assert unique_vals.issubset({-1, 0, 1}), f"Valores inesperados: {unique_vals}"
    print(f"OK test_oos_signals_valid_values (valores={sorted(unique_vals)})")


def test_oos_no_overlap_between_folds():
    """Índices OOS de diferentes folds no se solapan."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    all_indices = []
    for fr in result.fold_results:
        all_indices.extend(fr.oos_signals.index.tolist())
    assert len(all_indices) == len(set(map(str, all_indices))), (
        "Índices OOS duplicados entre folds"
    )
    print("OK test_oos_no_overlap_between_folds")


def test_no_train_indices_in_test():
    """Los índices de test nunca pertenecen al train (anti-leakage)."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    for fr in result.fold_results:
        train_set = set(pd.date_range(fr.train_start, fr.train_end, freq="B"))
        test_set = set(fr.oos_signals.index)
        overlap = train_set & test_set
        assert len(overlap) == 0, (
            f"Fold {fr.fold_idx+1}: {len(overlap)} índices solapan train y test"
        )
    print("OK test_no_train_indices_in_test")


def test_oos_proba_sums_to_one():
    """Probabilidades calibradas suman a 1 por fila."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    row_sums = result.oos_proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-4), (
        f"Probabilidades no suman a 1. Max desviacion: {(row_sums - 1).abs().max():.6f}"
    )
    print("OK test_oos_proba_sums_to_one")


def test_oos_sizing_columns():
    """oos_sizing tiene las columnas esperadas."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    expected_cols = {"signal", "p_win", "n_units", "sl_price", "tp_price",
                     "risk_pct", "risk_usd", "rr_dynamic", "kelly_raw"}
    missing = expected_cols - set(result.oos_sizing.columns)
    assert len(missing) == 0, f"Columnas faltantes: {missing}"
    print(f"OK test_oos_sizing_columns {list(result.oos_sizing.columns)}")


def test_summary_is_non_empty_string():
    """summary() produce un string con contenido."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    summary = result.summary()
    assert isinstance(summary, str)
    assert len(summary) > 100
    assert "WALK-FORWARD" in summary
    print("OK test_summary_is_non_empty_string")


def test_feature_importance_not_empty():
    """feature_importance_agg no vacío con track_importance=True."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    assert not result.feature_importance_agg.empty, (
        "feature_importance_agg no debe estar vacío"
    )
    assert "median_importance" in result.feature_importance_agg.columns
    print(f"OK test_feature_importance_not_empty "
          f"({len(result.feature_importance_agg)} features)")


def test_calibration_verdict_is_valid():
    """Cada fold tiene un verdict de calibración válido."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    valid_verdicts = {"OK", "MARGINAL", "POOR"}
    for fr in result.fold_results:
        v = fr.calibration.get("verdict")
        assert v in valid_verdicts, f"Fold {fr.fold_idx+1}: verdict inválido '{v}'"
    print(f"OK test_calibration_verdict_is_valid "
          f"{[fr.calibration.get('verdict') for fr in result.fold_results]}")


def test_global_metrics_has_psr():
    """global_metrics contiene PSR tras run con prices."""
    X, y, prices = _make_dataset(n=600)
    runner = _make_small_runner()
    result = runner.run(X, y, prices=prices, all_classes=[-1, 0, 1])
    assert "psr" in result.global_metrics, "global_metrics debe tener 'psr'"
    psr = result.global_metrics["psr"]
    assert 0.0 <= psr <= 1.0, f"PSR debe estar en [0,1], obtuvimos {psr}"
    print(f"OK test_global_metrics_has_psr (PSR={psr:.4f})")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("\nEjecutando tests del Walk-Forward Runner...\n")

    tests = [
        test_psr_perfect_returns,
        test_psr_random_returns,
        test_psr_negative_returns,
        test_dsr_leq_psr,
        test_dsr_decreases_with_more_trials,
        test_psr_few_samples_returns_zero,
        test_oos_metrics_keys,
        test_oos_metrics_no_trades,
        test_runner_completes_without_error,
        test_runner_n_folds_correct,
        test_oos_signals_valid_values,
        test_oos_no_overlap_between_folds,
        test_no_train_indices_in_test,
        test_oos_proba_sums_to_one,
        test_oos_sizing_columns,
        test_summary_is_non_empty_string,
        test_feature_importance_not_empty,
        test_calibration_verdict_is_valid,
        test_global_metrics_has_psr,
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
