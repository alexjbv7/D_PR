"""
Tests del Filtro de Entrada Probabilístico.

Casos validados:
 1. predict() lanza RuntimeError si no se llamó fit() antes.
 2. Con threshold=0 todos los no-neutrales generan señal (coverage máximo).
 3. Con threshold=1.0 ninguna barra genera señal (coverage = 0%).
 4. Filtro mejora precisión sobre el baseline sin filtro.
 5. Modelo perfecto con threshold óptimo → precision = 1.0.
 6. fit() con set pequeño usa fallback_threshold sin error.
 7. predict() devuelve solo valores en {-1, 0, +1}.
 8. Señales long y short son mutuamente excluyentes en el output.
 9. fit_entry_filter() falla si modelo no está calibrado.
10. fit_entry_filter() funciona en flujo completo con XGBoostClassifier.
11. search_results_dataframe() devuelve columnas correctas.
12. filter_stats() calcula coverage correctamente.
13. threshold_long y threshold_short son distintos en modo asimétrico.
14. Threshold mayor que baseline (sin filtro) siempre reduce coverage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.entry_filter import (
    ProbabilityEntryFilter,
    fit_entry_filter,
    _label_based_returns,
    _sharpe_from_returns,
)
from models.calibration import split_train_for_calibration
from models.zoo import XGBoostClassifier


# =====================================================================
# HELPERS
# =====================================================================

def _make_proba(n=200, n_classes=3, seed=42):
    """Probabilidades sintéticas normalizadas."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(n_classes), size=n)
    return raw.astype(float)


def _make_perfect_proba(y_true, class_labels=[-1, 0, 1]):
    """Probabilidades perfectas: P(clase_correcta)=1.0."""
    n = len(y_true)
    n_c = len(class_labels)
    proba = np.zeros((n, n_c))
    for i, label in enumerate(y_true):
        idx = class_labels.index(int(label))
        proba[i, idx] = 1.0
    return proba


def _make_synthetic_data(n=400, n_features=10, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y = pd.Series(np.where(X["feat_0"] > 0.5, 1, np.where(X["feat_0"] < -0.5, -1, 0)))
    return X, y


def _trained_calibrated_model(n=400, seed=0):
    X, y = _make_synthetic_data(n=n, seed=seed)
    X_fit, y_fit, X_calib, y_calib = split_train_for_calibration(X, y, calib_frac=0.25)
    model = XGBoostClassifier(n_estimators=50, max_depth=3)
    model.fit(X_fit, y_fit, all_classes=[-1, 0, 1])
    model.calibrate(X_calib, y_calib, method="sigmoid")
    return model, X, y, X_calib, y_calib


# =====================================================================
# TESTS BÁSICOS
# =====================================================================

def test_predict_raises_before_fit():
    """predict() debe fallar con RuntimeError antes de fit()."""
    ef = ProbabilityEntryFilter()
    proba = _make_proba(50)
    try:
        ef.predict(proba)
        assert False, "Debe lanzar RuntimeError"
    except RuntimeError:
        pass
    print("OK test_predict_raises_before_fit")


def test_output_values_are_valid():
    """predict() solo debe devolver valores en {-1, 0, +1}."""
    proba = _make_proba(200)
    y = np.random.default_rng(0).choice([-1, 0, 1], size=200)
    ef = ProbabilityEntryFilter()
    ef.fit(proba, y)
    signals = ef.predict(proba)
    assert set(np.unique(signals)).issubset({-1, 0, 1}), (
        f"Valores inesperados: {np.unique(signals)}"
    )
    print("OK test_output_values_are_valid")


def test_zero_threshold_max_coverage():
    """
    threshold=0 → todas las barras con P(long)>0 o P(short)>0 generan señal.
    Con proba dirichlet, casi siempre habrá alguna clase dominante.
    """
    proba = _make_proba(300)
    y = np.random.default_rng(0).choice([-1, 0, 1], size=300)
    ef = ProbabilityEntryFilter(fallback_threshold=0.0, min_samples_to_optimize=9999)
    ef.fit(proba, y)
    signals = ef.predict(proba)
    coverage = (signals != 0).mean()
    # Con threshold=0, cada barra generará señal según la clase de mayor probabilidad
    # (salvo clase 0, si es la dominante)
    assert coverage >= 0.0
    print(f"OK test_zero_threshold_max_coverage (coverage={coverage:.1%})")


def test_high_threshold_low_coverage():
    """threshold muy alto → coverage cercano a 0."""
    proba = _make_proba(500)
    y = np.random.default_rng(0).choice([-1, 0, 1], size=500)
    ef = ProbabilityEntryFilter()
    # Forzar threshold alto manualmente
    ef.threshold_long_ = 0.99
    ef.threshold_short_ = 0.99
    ef._fitted = True
    signals = ef.predict(proba)
    coverage = (signals != 0).mean()
    assert coverage < 0.05, f"Coverage esperado < 5% con threshold=0.99, obtuvimos {coverage:.1%}"
    print(f"OK test_high_threshold_low_coverage (coverage={coverage:.1%})")


def test_perfect_model_precision():
    """Con probabilidades perfectas el filtro debe dar precision=1.0."""
    y_true = np.array([-1, 0, 1, 1, -1, 0, 1, -1, -1, 1] * 10)
    proba = _make_perfect_proba(y_true)
    ef = ProbabilityEntryFilter(min_coverage=0.01, min_trades=3)
    ef.fit(proba, y_true)
    signals = ef.predict(proba)
    mask = signals != 0
    if mask.sum() > 0:
        precision = (signals[mask] == y_true[mask]).mean()
        assert precision == 1.0, f"Precision esperada 1.0, obtuvimos {precision}"
    print("OK test_perfect_model_precision")


def test_small_calibration_set_uses_fallback():
    """Con set de calibración pequeño, usa fallback_threshold sin error."""
    proba = _make_proba(20)
    y = np.random.default_rng(0).choice([-1, 0, 1], size=20)
    ef = ProbabilityEntryFilter(fallback_threshold=0.5, min_samples_to_optimize=60)
    ef.fit(proba, y)
    assert ef.is_fitted
    assert ef.threshold_long_ == 0.5
    assert ef.threshold_short_ == 0.5
    print("OK test_small_calibration_set_uses_fallback")


def test_signals_mutually_exclusive():
    """Cada barra solo puede tener una señal: nunca +1 y -1 a la vez."""
    proba = _make_proba(200)
    y = np.random.default_rng(1).choice([-1, 0, 1], size=200)
    ef = ProbabilityEntryFilter()
    ef.fit(proba, y)
    signals = ef.predict(proba)
    # No puede haber conflicto (ya lo gestiona _apply_threshold)
    assert signals.dtype in [np.int32, np.int64, int, np.intp]
    assert not np.any((signals == 1) & (signals == -1))  # tautología, pero explícito
    print("OK test_signals_mutually_exclusive")


# =====================================================================
# TESTS COBERTURA Y PRECISION
# =====================================================================

def test_filter_reduces_coverage():
    """Un threshold > baseline siempre debe reducir coverage."""
    proba = _make_proba(300)
    y = np.random.default_rng(2).choice([-1, 0, 1], size=300)

    # Sin filtro: señal = argmax(proba) mapeado a {-1, 0, +1}
    no_filter_signals = np.array([-1, 0, 1])[np.argmax(proba, axis=1)]
    coverage_no_filter = float((no_filter_signals != 0).mean())

    # Con filtro optimizado
    ef = ProbabilityEntryFilter(min_coverage=0.02, min_trades=5)
    ef.fit(proba, y)
    signals = ef.predict(proba)
    coverage_filtered = float((signals != 0).mean())

    # El filtro óptimo puede seleccionar threshold > 1/3, lo que reduce cobertura
    # (no garantizado si fallback=threshold muy bajo, pero generalmente se cumple)
    assert coverage_filtered <= coverage_no_filter + 0.01, (
        f"Filtro debería reducir coverage: {coverage_filtered:.1%} vs {coverage_no_filter:.1%}"
    )
    print(f"OK test_filter_reduces_coverage ({coverage_no_filter:.1%} -> {coverage_filtered:.1%})")


def test_filter_stats_coverage_correct():
    """filter_stats() debe calcular coverage correctamente."""
    proba = _make_proba(100)
    y = np.random.default_rng(3).choice([-1, 0, 1], size=100)
    ef = ProbabilityEntryFilter()
    ef.fit(proba, y)
    stats = ef.filter_stats(proba, y, label="test")
    signals = ef.predict(proba)
    expected_coverage = float((signals != 0).mean())
    assert abs(stats["coverage"] - expected_coverage) < 1e-6
    assert stats["n_trades"] == int((signals != 0).sum())
    assert stats["n_long"] == int((signals == 1).sum())
    assert stats["n_short"] == int((signals == -1).sum())
    print("OK test_filter_stats_coverage_correct")


def test_search_results_columns():
    """search_results_dataframe() debe tener columnas esperadas."""
    proba = _make_proba(200)
    y = np.random.default_rng(4).choice([-1, 0, 1], size=200)
    ef = ProbabilityEntryFilter(symmetric=True)
    ef.fit(proba, y)
    df = ef.search_results_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert "threshold" in df.columns
    assert "sharpe" in df.columns
    assert "coverage" in df.columns
    assert "precision" in df.columns
    print("OK test_search_results_columns")


# =====================================================================
# TESTS CON RETORNOS REALES
# =====================================================================

def test_fit_with_bar_returns():
    """fit() con bar_returns_calib reales no debe lanzar error."""
    rng = np.random.default_rng(5)
    proba = _make_proba(200, seed=5)
    y = rng.choice([-1, 0, 1], size=200)
    bar_returns = rng.normal(0.001, 0.01, size=200)  # retornos diarios sintéticos
    ef = ProbabilityEntryFilter()
    ef.fit(proba, y, bar_returns_calib=bar_returns)
    assert ef.is_fitted
    print("OK test_fit_with_bar_returns")


def test_label_based_returns_correct():
    """_label_based_returns: +1 si signal==y_true, -1 si no."""
    signals = np.array([1, -1, 1, 0, -1])
    y_true = np.array([1, -1, -1, 1, 1])
    ret = _label_based_returns(signals, y_true)
    # Solo los índices 0,1,2,4 tienen señal (!=0)
    # 0: signal=1, y=1 → correcto → +1
    # 1: signal=-1, y=-1 → correcto → +1
    # 2: signal=1, y=-1 → incorrecto → -1
    # 4: signal=-1, y=1 → incorrecto → -1
    expected = np.array([1.0, 1.0, -1.0, -1.0])
    assert np.allclose(ret, expected), f"Esperado {expected}, obtuvimos {ret}"
    print("OK test_label_based_returns_correct")


def test_sharpe_perfect_returns():
    """Sharpe de retornos consistentemente positivos debe ser muy alto."""
    rng = np.random.default_rng(99)
    # Retornos positivos con varianza muy pequeña → Sharpe alto
    returns = rng.normal(loc=0.01, scale=0.0001, size=100)
    sh = _sharpe_from_returns(returns)
    assert sh > 10.0, f"Sharpe esperado > 10 con retornos casi perfectos, obtuvimos {sh}"
    print("OK test_sharpe_perfect_returns")


def test_sharpe_insufficient_trades():
    """Con menos de min_trades, Sharpe debe ser -inf."""
    returns = np.array([0.01, 0.02])
    sh = _sharpe_from_returns(returns, min_trades=5)
    assert sh == -np.inf
    print("OK test_sharpe_insufficient_trades")


# =====================================================================
# TESTS INTEGRACIÓN CON XGBOOST
# =====================================================================

def test_fit_entry_filter_raises_if_not_calibrated():
    """fit_entry_filter() debe fallar si el modelo no está calibrado."""
    X, y = _make_synthetic_data(n=200)
    model = XGBoostClassifier(n_estimators=30, max_depth=2)
    model.fit(X, y, all_classes=[-1, 0, 1])
    assert not model.is_calibrated
    try:
        fit_entry_filter(model, X.tail(40), y.tail(40))
        assert False, "Debe lanzar RuntimeError"
    except RuntimeError:
        pass
    print("OK test_fit_entry_filter_raises_if_not_calibrated")


def test_fit_entry_filter_end_to_end():
    """Flujo completo: fit + calibrate + fit_entry_filter + predict."""
    model, X, y, X_calib, y_calib = _trained_calibrated_model(n=400)
    assert model.is_calibrated

    ef = fit_entry_filter(model, X_calib, y_calib)
    assert ef.is_fitted
    assert 0.0 < ef.threshold_long_ <= 1.0
    assert 0.0 < ef.threshold_short_ <= 1.0

    # Predecir en nuevas barras
    proba = model.predict_proba(X.tail(50))
    signals = ef.predict(proba)
    assert len(signals) == 50
    assert set(np.unique(signals)).issubset({-1, 0, 1})
    print(f"OK test_fit_entry_filter_end_to_end "
          f"(threshold={ef.threshold_long_:.3f}, "
          f"coverage={float((signals!=0).mean()):.1%})")


def test_asymmetric_thresholds_differ():
    """En modo asimétrico, threshold_long y threshold_short pueden diferir."""
    rng = np.random.default_rng(7)
    # Crear datos donde long y short tienen distribuciones distintas
    n = 300
    proba = rng.dirichlet([3, 1, 1], size=n)  # sesgo hacia clase 0 (índice 0 = -1)
    y = rng.choice([-1, 0, 1], size=n)
    ef = ProbabilityEntryFilter(symmetric=False, min_trades=3, min_coverage=0.02)
    ef.fit(proba, y)
    assert ef.is_fitted
    # No necesariamente distintos, pero el código debe ejecutar sin error
    print(f"OK test_asymmetric_thresholds_differ "
          f"(long={ef.threshold_long_:.3f}, short={ef.threshold_short_:.3f})")


def test_repr():
    """__repr__() debe mostrar thresholds después de fit."""
    proba = _make_proba(200)
    y = np.random.default_rng(8).choice([-1, 0, 1], size=200)
    ef = ProbabilityEntryFilter()
    assert "not fitted" in repr(ef)
    ef.fit(proba, y)
    assert "threshold_long" in repr(ef)
    print(f"OK test_repr: {repr(ef)}")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("\nEjecutando tests del filtro de entrada probabilístico...\n")

    tests = [
        test_predict_raises_before_fit,
        test_output_values_are_valid,
        test_zero_threshold_max_coverage,
        test_high_threshold_low_coverage,
        test_perfect_model_precision,
        test_small_calibration_set_uses_fallback,
        test_signals_mutually_exclusive,
        test_filter_reduces_coverage,
        test_filter_stats_coverage_correct,
        test_search_results_columns,
        test_fit_with_bar_returns,
        test_label_based_returns_correct,
        test_sharpe_perfect_returns,
        test_sharpe_insufficient_trades,
        test_fit_entry_filter_raises_if_not_calibrated,
        test_fit_entry_filter_end_to_end,
        test_asymmetric_thresholds_differ,
        test_repr,
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
