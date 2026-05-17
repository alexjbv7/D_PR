"""
Tests de calibración y feature selection.

Casos validados (sin red, sin datos reales):
 1. ECE = 0 cuando el modelo es perfecto.
 2. ECE = 0.5 para un modelo completamente descalibrado.
 3. Brier score < 0.25 para modelo informativo.
 4. split_train_for_calibration respeta orden temporal.
 5. IsotonicCalibrator falla si se llama antes de fit (RuntimeError).
 6. XGBoostClassifier.calibrate() funciona en datos sintéticos multiclase.
 7. predict_proba calibrada suma a 1 por fila.
 8. is_calibrated es False antes y True después de calibrar.
 9. compute_gain_importance devuelve Series normalizada [0,1].
10. compute_permutation_importance devuelve DataFrame con columnas correctas.
11. compute_shap_importance devuelve Series normalizada con features correctas.
12. aggregate_fold_importances agrega correctamente.
13. select_features_to_drop identifica features de importancia cero.
14. Feature con alta importancia NO es eliminada.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from models.calibration import (
    expected_calibration_error,
    brier_score,
    IsotonicCalibrator,
    split_train_for_calibration,
)
from models.feature_selection import (
    compute_gain_importance,
    compute_permutation_importance,
    compute_shap_importance,
    aggregate_fold_importances,
    select_features_to_drop,
)
from models.zoo import XGBoostClassifier


# =====================================================================
# HELPERS
# =====================================================================

def _make_synthetic_data(n=400, n_features=10, seed=0):
    """Dataset sintético multiclase {-1, 0, +1} con señal real."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    # Señal real: feat_0 determina la clase (informativa)
    y = pd.Series(np.where(X["feat_0"] > 0.5, 1, np.where(X["feat_0"] < -0.5, -1, 0)))
    return X, y


def _trained_model(n=400, seed=0) -> XGBoostClassifier:
    X, y = _make_synthetic_data(n=n, seed=seed)
    model = XGBoostClassifier(n_estimators=50, max_depth=3)
    model.fit(X, y, all_classes=[-1, 0, 1])
    return model, X, y


# =====================================================================
# TESTS ECE / BRIER
# =====================================================================

def test_ece_perfect_model():
    """Modelo perfecto: predicción de probabilidad es exactamente 0 o 1."""
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_proba = np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    ece = expected_calibration_error(y_true, y_proba)
    assert ece < 1e-9, f"ECE perfecto debe ser 0, obtuvimos {ece}"
    print("OK test_ece_perfect_model")


def test_ece_worst_model():
    """Modelo completamente invertido: dice 1.0 donde es 0 y viceversa."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_proba = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    ece = expected_calibration_error(y_true, y_proba)
    assert ece > 0.5, f"ECE modelo invertido debe ser alto, obtuvimos {ece}"
    print("OK test_ece_worst_model")


def test_brier_perfect():
    y_true = np.array([1, 0, 1, 0])
    y_proba = np.array([1.0, 0.0, 1.0, 0.0])
    bs = brier_score(y_true, y_proba)
    assert bs < 1e-9, f"Brier perfecto debe ser 0, obtuvimos {bs}"
    print("OK test_brier_perfect")


def test_brier_uninformative():
    """Modelo que siempre predice 0.5: Brier = 0.25."""
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_proba = np.full(8, 0.5)
    bs = brier_score(y_true, y_proba)
    assert abs(bs - 0.25) < 1e-9, f"Brier uninformativo debe ser 0.25, obtuvimos {bs}"
    print("OK test_brier_uninformative")


# =====================================================================
# TESTS SPLIT
# =====================================================================

def test_split_train_respects_temporal_order():
    """Los últimos n*calib_frac registros van a calibración."""
    X = pd.DataFrame({"a": range(100)})
    y = pd.Series(range(100))
    X_fit, y_fit, X_calib, y_calib = split_train_for_calibration(X, y, calib_frac=0.2)
    assert len(X_fit) == 80
    assert len(X_calib) == 20
    # El último de fit precede al primero de calib (orden temporal)
    assert X_fit.index[-1] < X_calib.index[0]
    print("OK test_split_train_respects_temporal_order")


def test_split_train_no_overlap():
    X = pd.DataFrame({"a": range(50)})
    y = pd.Series(range(50))
    X_fit, _, X_calib, _ = split_train_for_calibration(X, y, calib_frac=0.3)
    overlap = set(X_fit.index) & set(X_calib.index)
    assert len(overlap) == 0, f"Solapamiento entre fit y calibración: {overlap}"
    print("OK test_split_train_no_overlap")


# =====================================================================
# TESTS CALIBRADOR
# =====================================================================

def test_calibrator_raises_before_fit():
    cal = IsotonicCalibrator()
    try:
        cal.predict_proba(np.zeros((5, 3)))
        assert False, "Debe lanzar RuntimeError"
    except RuntimeError:
        pass
    print("OK test_calibrator_raises_before_fit")


def test_xgboost_calibrate_end_to_end():
    """Calibrar XGBoostClassifier multiclase y verificar que predict_proba funciona."""
    model, X, y = _trained_model(n=400)
    X_fit, y_fit, X_calib, y_calib = split_train_for_calibration(X, y, calib_frac=0.25)

    # Re-entrena en el split de fit
    model2 = XGBoostClassifier(n_estimators=50, max_depth=3)
    model2.fit(X_fit, y_fit, all_classes=[-1, 0, 1])

    assert not model2.is_calibrated
    model2.calibrate(X_calib, y_calib, method="sigmoid")
    assert model2.is_calibrated

    proba = model2.predict_proba(X.head(20))
    assert proba.shape == (20, 3), f"Shape esperado (20,3), obtuvimos {proba.shape}"
    print("OK test_xgboost_calibrate_end_to_end")


def test_predict_proba_sums_to_one():
    """Probabilidades calibradas deben sumar a 1 por fila."""
    model, X, y = _trained_model(n=400)
    X_fit, y_fit, X_calib, y_calib = split_train_for_calibration(X, y, calib_frac=0.25)

    model2 = XGBoostClassifier(n_estimators=50, max_depth=3)
    model2.fit(X_fit, y_fit, all_classes=[-1, 0, 1])
    model2.calibrate(X_calib, y_calib, method="sigmoid")

    proba = model2.predict_proba(X.head(50))
    row_sums = proba.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-5), (
        f"Filas no suman a 1. Max desviación: {np.abs(row_sums - 1).max()}"
    )
    print("OK test_predict_proba_sums_to_one")


def test_is_calibrated_flag():
    model2 = XGBoostClassifier(n_estimators=30, max_depth=2)
    X, y = _make_synthetic_data(n=200)
    model2.fit(X, y, all_classes=[-1, 0, 1])
    assert not model2.is_calibrated
    model2.calibrate(X.tail(40), y.tail(40), method="sigmoid")
    assert model2.is_calibrated
    print("OK test_is_calibrated_flag")


# =====================================================================
# TESTS FEATURE SELECTION
# =====================================================================

def test_gain_importance_normalized():
    """gain importance debe sumar a 1.0."""
    model, X, y = _trained_model()
    imp = compute_gain_importance(model)
    assert isinstance(imp, pd.Series)
    assert abs(imp.sum() - 1.0) < 1e-6, f"Sum={imp.sum()}, esperaba 1.0"
    assert (imp >= 0).all(), "Importancias deben ser >= 0"
    print("OK test_gain_importance_normalized")


def test_gain_importance_feat0_high():
    """feat_0 es la feature más informativa en nuestros datos sintéticos."""
    model, X, y = _trained_model(n=600)
    imp = compute_gain_importance(model)
    top_feature = imp.index[0]
    assert top_feature == "feat_0", (
        f"Se esperaba feat_0 como la más importante, obtuvimos {top_feature}"
    )
    print("OK test_gain_importance_feat0_high")


def test_permutation_importance_shape():
    """permutation importance devuelve DataFrame con las columnas correctas."""
    model, X, y = _trained_model(n=300)
    X_oos = X.tail(60)
    y_oos = y.tail(60)
    perm = compute_permutation_importance(model, X_oos, y_oos, n_repeats=5)
    assert isinstance(perm, pd.DataFrame)
    assert set(["mean", "std", "ci_lower", "ci_upper"]).issubset(perm.columns)
    assert len(perm) == len(model.feature_names_)
    print("OK test_permutation_importance_shape")


def test_shap_importance_normalized():
    """SHAP importance debe sumar a 1.0 y tener los features correctos."""
    model, X, y = _trained_model(n=400)
    shap_imp = compute_shap_importance(model, X, sample_size=100)
    assert isinstance(shap_imp, pd.Series)
    assert set(shap_imp.index) == set(model.feature_names_), (
        "Los índices de SHAP deben coincidir con feature_names_"
    )
    assert abs(shap_imp.sum() - 1.0) < 1e-5, f"Sum SHAP={shap_imp.sum()}"
    print("OK test_shap_importance_normalized")


def test_aggregate_fold_importances():
    """aggregate_fold_importances debe producir estadísticas correctas."""
    features = ["a", "b", "c"]
    fold0 = pd.Series([0.5, 0.3, 0.2], index=features)
    fold1 = pd.Series([0.4, 0.4, 0.2], index=features)
    fold2 = pd.Series([0.6, 0.2, 0.2], index=features)

    agg = aggregate_fold_importances([fold0, fold1, fold2])
    assert "median_importance" in agg.columns
    assert "frac_folds_nonzero" in agg.columns
    # 'a' debe tener mediana 0.5
    assert abs(agg.loc["a", "median_importance"] - 0.5) < 1e-9
    # Todos nonzero en todos los folds
    assert agg.loc["a", "frac_folds_nonzero"] == 1.0
    print("OK test_aggregate_fold_importances")


def test_select_features_to_drop_zero_importance():
    """Feature con importancia 0 en todos los folds debe ser eliminada."""
    features = ["signal", "noise"]
    fold0 = pd.Series([0.9, 0.0], index=features)
    fold1 = pd.Series([0.8, 0.0], index=features)
    agg = aggregate_fold_importances([fold0, fold1])
    to_drop = select_features_to_drop(agg, median_threshold=0.001, frac_nonzero_threshold=0.1)
    assert "noise" in to_drop, f"'noise' debería eliminarse, obtuvimos: {to_drop}"
    assert "signal" not in to_drop, "'signal' NO debe eliminarse"
    print("OK test_select_features_to_drop_zero_importance")


def test_select_features_high_importance_kept():
    """Feature con alta importancia nunca debe ser eliminada."""
    features = ["feat_0", "feat_1"]
    fold0 = pd.Series([0.8, 0.2], index=features)
    agg = aggregate_fold_importances([fold0])
    to_drop = select_features_to_drop(agg, median_threshold=0.005)
    assert "feat_0" not in to_drop
    print("OK test_select_features_high_importance_kept")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    import os
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    print("\nEjecutando tests de calibración y feature selection...\n")
    tests = [
        test_ece_perfect_model,
        test_ece_worst_model,
        test_brier_perfect,
        test_brier_uninformative,
        test_split_train_respects_temporal_order,
        test_split_train_no_overlap,
        test_calibrator_raises_before_fit,
        test_xgboost_calibrate_end_to_end,
        test_predict_proba_sums_to_one,
        test_is_calibrated_flag,
        test_gain_importance_normalized,
        test_gain_importance_feat0_high,
        test_permutation_importance_shape,
        test_shap_importance_normalized,
        test_aggregate_fold_importances,
        test_select_features_to_drop_zero_importance,
        test_select_features_high_importance_kept,
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
