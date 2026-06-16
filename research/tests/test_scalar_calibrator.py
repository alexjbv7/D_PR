"""
Tests del ScalarProbabilityCalibrator (E3 paso 2 · arbitraje D).

Verifican el calibrador 1D que convierte el softmax ordinal del DQN en una
probabilidad usable (habilita p_win_calibrated=True y el guard de sizing R-02).
Sin torch: prueban solo la parte estadística (sklearn).

1. La calibración isotónica reduce el ECE de un score sobreconfiado.
2. Corrige la sobreconfianza: calibrado(p_alto) < p_alto.
3. Monotonía: el mapeo isotónico es no-decreciente.
4. Fallback a sigmoid cuando hay pocas muestras.
5. Outcomes de una sola clase -> mapeo constante, sin crash.
6. __call__ devuelve un float escalar en [0,1]; transform respeta longitud.
"""
import numpy as np
import pytest

from models.calibration import (
    ScalarProbabilityCalibrator,
    expected_calibration_error,
)


def _overconfident_data(n: int = 400, seed: int = 7):
    """Scores crudos altos con tasa real de acierto mucho menor (sobreconfianza)."""
    rng = np.random.default_rng(seed)
    raw = np.linspace(0.50, 0.95, n)
    true_p = 0.45 + 0.20 * (raw - 0.50) / 0.45    # en [0.45, 0.65]: muy por debajo de raw
    outcomes = (rng.random(n) < true_p).astype(float)
    return raw, outcomes


def test_isotonic_reduces_ece():
    raw, outcomes = _overconfident_data()
    cal = ScalarProbabilityCalibrator(method="isotonic", min_samples_isotonic=80)
    cal.fit(raw, outcomes)
    ece_before = expected_calibration_error(outcomes, raw)
    ece_after = expected_calibration_error(outcomes, cal.transform(raw))
    assert ece_after < ece_before
    assert ece_after < 0.10          # calibración aceptable in-sample


def test_corrects_overconfidence():
    raw, outcomes = _overconfident_data()
    cal = ScalarProbabilityCalibrator().fit(raw, outcomes)
    # un score crudo de 0.90 NO debe seguir valiendo ~0.90 tras calibrar
    assert cal(0.90) < 0.90
    assert 0.0 <= cal(0.90) <= 1.0


def test_monotonic_mapping():
    raw, outcomes = _overconfident_data()
    cal = ScalarProbabilityCalibrator().fit(raw, outcomes)
    grid = np.linspace(0.5, 0.95, 50)
    mapped = cal.transform(grid)
    assert np.all(np.diff(mapped) >= -1e-9)   # no-decreciente (isotónica)


def test_sigmoid_fallback_small_sample():
    raw, outcomes = _overconfident_data(n=30)
    cal = ScalarProbabilityCalibrator(method="isotonic", min_samples_isotonic=80)
    cal.fit(raw, outcomes)
    assert cal._kind == "sigmoid"          # cayó a Platt por muestra pequeña
    assert cal.is_fitted


def test_single_class_outcomes_constant():
    raw = np.linspace(0.5, 0.9, 100)
    outcomes = np.ones(100)                 # todas ganadoras -> degenerado
    cal = ScalarProbabilityCalibrator().fit(raw, outcomes)
    assert cal._kind == "constant"
    assert cal(0.6) == pytest.approx(1.0)
    assert cal(0.9) == pytest.approx(1.0)   # sin crash, constante


def test_scalar_and_array_api():
    raw, outcomes = _overconfident_data()
    cal = ScalarProbabilityCalibrator().fit(raw, outcomes)
    out = cal(0.7)
    assert isinstance(out, float) and 0.0 <= out <= 1.0
    arr = cal.transform([0.6, 0.7, 0.8])
    assert arr.shape == (3,)
    assert np.all((arr >= 0.0) & (arr <= 1.0))


def test_requires_fit_before_use():
    cal = ScalarProbabilityCalibrator()
    with pytest.raises(RuntimeError):
        cal(0.7)


def test_mismatched_shapes_raise():
    cal = ScalarProbabilityCalibrator()
    with pytest.raises(ValueError):
        cal.fit(np.array([0.5, 0.6]), np.array([1.0]))
