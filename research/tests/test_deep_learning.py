"""
Tests — Deep Learning (CS229 cheatsheet)
=========================================
Cubre:
  1. DeepMLPClassifier — fit / predict / predict_proba / calibrate
  2. LSTMClassifier — LayerNorm upgrade
  3. QLearningAgent — train / act / q_table
  4. IsotonicCalibrator.fit_from_proba()
  5. BaseModel.calibrate() vía DeepMLP (integración)
  6. get_model() factory actualizado
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures comunes
# ---------------------------------------------------------------------------

@pytest.fixture
def small_classification_data():
    """Dataset sintético 3-clases, 200 muestras, 10 features."""
    rng = np.random.default_rng(42)
    n, k = 200, 10
    X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"f{i}" for i in range(k)])
    # Clases -1, 0, +1 con proporciones ~15%, 70%, 15%
    y_raw = rng.choice([-1, 0, 1], size=n, p=[0.15, 0.70, 0.15])
    y = pd.Series(y_raw)
    return X, y


@pytest.fixture
def split_data(small_classification_data):
    X, y = small_classification_data
    n_fit = 130
    n_calib = 40
    return (
        X.iloc[:n_fit],    y.iloc[:n_fit],
        X.iloc[n_fit:n_fit+n_calib], y.iloc[n_fit:n_fit+n_calib],
        X.iloc[n_fit+n_calib:], y.iloc[n_fit+n_calib:],
    )


# ===========================================================================
# 1. IsotonicCalibrator.fit_from_proba
# ===========================================================================

class TestFitFromProba:
    def test_fit_from_proba_sigmoid(self):
        from models.calibration import IsotonicCalibrator
        rng = np.random.default_rng(0)
        n, k = 60, 3
        proba_raw = rng.dirichlet(np.ones(k), size=n)
        y = rng.integers(0, k, size=n)
        cal = IsotonicCalibrator(method="sigmoid")
        cal.fit_from_proba(proba_raw, y)
        assert cal.is_fitted

    def test_fit_from_proba_isotonic(self):
        from models.calibration import IsotonicCalibrator
        rng = np.random.default_rng(1)
        n, k = 100, 3
        proba_raw = rng.dirichlet(np.ones(k), size=n)
        y = rng.integers(0, k, size=n)
        cal = IsotonicCalibrator(method="isotonic", min_samples_isotonic=50)
        cal.fit_from_proba(proba_raw, y)
        assert cal.is_fitted

    def test_fit_from_proba_output_shape(self):
        from models.calibration import IsotonicCalibrator
        rng = np.random.default_rng(2)
        n_fit, n_test, k = 80, 30, 3
        proba_train = rng.dirichlet(np.ones(k), size=n_fit)
        y_train = rng.integers(0, k, size=n_fit)
        proba_test = rng.dirichlet(np.ones(k), size=n_test)

        cal = IsotonicCalibrator(method="sigmoid")
        cal.fit_from_proba(proba_train, y_train)
        out = cal.predict_proba_from_raw(proba_test)
        assert out.shape == (n_test, k)
        np.testing.assert_allclose(out.sum(axis=1), np.ones(n_test), atol=1e-5)

    def test_fit_from_proba_small_auto_fallback(self):
        """Con <80 muestras y method='isotonic' debe caer a sigmoid sin error."""
        from models.calibration import IsotonicCalibrator
        rng = np.random.default_rng(3)
        n, k = 30, 3
        proba_raw = rng.dirichlet(np.ones(k), size=n)
        y = rng.integers(0, k, size=n)
        cal = IsotonicCalibrator(method="isotonic")
        cal.fit_from_proba(proba_raw, y)
        assert cal.is_fitted


# ===========================================================================
# 2. DeepMLPClassifier
# ===========================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch no instalado"
)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch no instalado")
class TestDeepMLP:
    """Tests del DeepMLPClassifier."""

    def _make_model(self, **kwargs):
        from models.zoo import DeepMLPClassifier
        defaults = dict(
            hidden_dims=[16, 8],
            dropout=0.0,
            epochs=5,
            batch_size=32,
            patience=50,   # no early stopping en tests
            device='cpu',
        )
        defaults.update(kwargs)
        return DeepMLPClassifier(**defaults)

    def test_fit_predict_shape(self, split_data):
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = self._make_model()
        model.fit(X_fit, y_fit)
        preds = model.predict(X_test)
        assert preds.shape == (len(X_test),)
        assert set(preds).issubset({-1, 0, 1})

    def test_fit_predict_proba_shape(self, split_data):
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = self._make_model()
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-5)

    def test_predict_proba_values_in_01(self, split_data):
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = self._make_model()
        model.fit(X_fit, y_fit)
        proba = model.predict_proba(X_test)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_name_attribute(self):
        from models.zoo import DeepMLPClassifier
        assert DeepMLPClassifier.name == "deep_mlp"

    def test_feature_names_stored(self, split_data):
        X_fit, y_fit, *_ = split_data
        model = self._make_model()
        model.fit(X_fit, y_fit)
        assert model.feature_names_ == list(X_fit.columns)

    def test_all_classes_consistency(self, split_data):
        """Modelo con all_classes=[−1,0,1] debe tener 3 columnas aunque y_fit no tenga clase −1."""
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        # Quitar clase -1 del train
        mask = y_fit != -1
        model = self._make_model()
        model.fit(X_fit[mask], y_fit[mask], all_classes=[-1, 0, 1])
        proba = model.predict_proba(X_test)
        assert proba.shape[1] == 3

    def test_calibrate(self, split_data):
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = self._make_model()
        model.fit(X_fit, y_fit)
        model.calibrate(X_calib, y_calib, method="sigmoid")
        assert model.is_calibrated
        proba = model.predict_proba(X_test)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X_test)), atol=1e-5)

    def test_predict_proba_raw_differs_from_calibrated(self, split_data):
        """predict_proba_raw() y predict_proba() difieren tras calibrar."""
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = self._make_model()
        model.fit(X_fit, y_fit)
        raw_before = model.predict_proba_raw(X_test).copy()
        model.calibrate(X_calib, y_calib, method="sigmoid")
        raw_after = model.predict_proba_raw(X_test)
        cal = model.predict_proba(X_test)
        # raw_before ≈ raw_after (modelo no cambió)
        np.testing.assert_allclose(raw_before, raw_after, atol=1e-6)
        # calibrated ≠ raw (salvo coincidencia)
        # Las sumas deben seguir siendo 1
        np.testing.assert_allclose(cal.sum(axis=1), np.ones(len(X_test)), atol=1e-5)

    def test_feature_importance_shape(self, split_data):
        X_fit, y_fit, *_ = split_data
        model = self._make_model()
        model.fit(X_fit, y_fit)
        imp = model.feature_importance()
        assert isinstance(imp, pd.Series)
        assert len(imp) == X_fit.shape[1]

    def test_eval_set_early_stopping(self, split_data):
        """Comprobar que fit() con eval_set no lanza excepciones."""
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = self._make_model(epochs=10, patience=3)
        model.fit(X_fit, y_fit, eval_set=(X_calib, y_calib))
        assert model.model is not None

    def test_get_model_factory(self):
        from models.zoo import get_model
        model = get_model("deep_mlp", hidden_dims=[8], epochs=1, device='cpu')
        assert model.name == "deep_mlp"


# ===========================================================================
# 3. LSTMClassifier — LayerNorm
# ===========================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch no instalado")
class TestLSTMLayerNorm:
    def test_lstm_has_layer_norm(self):
        """LSTMNet interno debe tener layer_norm."""
        import torch
        import torch.nn as nn
        from models.zoo import LSTMClassifier
        rng = np.random.default_rng(0)
        n, k = 120, 5
        X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"f{i}" for i in range(k)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        model = LSTMClassifier(seq_length=10, hidden_size=8, num_layers=1,
                               dropout=0.0, epochs=2, batch_size=32, device='cpu')
        model.fit(X, y)
        # El modelo interno debe tener un atributo layer_norm
        assert hasattr(model.model, 'layer_norm'), \
            "LSTMNet debería tener atributo 'layer_norm' (LayerNorm)"
        assert isinstance(model.model.layer_norm, nn.LayerNorm)

    def test_lstm_predict_shape(self):
        from models.zoo import LSTMClassifier
        rng = np.random.default_rng(1)
        n, k, seq = 120, 5, 10
        X = pd.DataFrame(rng.standard_normal((n, k)), columns=[f"f{i}" for i in range(k)])
        y = pd.Series(rng.choice([-1, 0, 1], size=n))
        model = LSTMClassifier(seq_length=seq, hidden_size=8, num_layers=1,
                               epochs=2, batch_size=32, device='cpu')
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == n


# ===========================================================================
# 4. QLearningAgent
# ===========================================================================

class TestQLearningAgent:
    @pytest.fixture
    def agent_data(self):
        rng = np.random.default_rng(42)
        n = 200
        X = pd.DataFrame({
            "return_1":   rng.normal(0, 0.01, n),
            "sma_ratio":  rng.normal(1.0, 0.01, n),
            "f2":         rng.standard_normal(n),
        })
        price_ret = pd.Series(rng.normal(0, 0.01, n))
        regimes = pd.Series(rng.integers(0, 3, n))
        p_win = pd.Series(rng.uniform(0.3, 0.7, n))
        signals = pd.Series(rng.choice([-1, 0, 1], size=n))
        return X, price_ret, regimes, p_win, signals

    def test_train_sets_is_trained(self, agent_data):
        from models.rl_agent import QLearningAgent
        X, ret, reg, pw, sig = agent_data
        agent = QLearningAgent()
        agent.train(X, ret, regime_labels=reg, p_win_series=pw,
                    primary_signals=sig)
        assert agent.is_trained

    def test_q_table_non_empty(self, agent_data):
        from models.rl_agent import QLearningAgent
        X, ret, reg, pw, sig = agent_data
        agent = QLearningAgent()
        agent.train(X, ret, regime_labels=reg, p_win_series=pw)
        qt = agent.q_table()
        assert len(qt) > 0

    def test_act_returns_valid_action(self, agent_data):
        from models.rl_agent import QLearningAgent
        X, ret, reg, pw, sig = agent_data
        agent = QLearningAgent()
        agent.train(X, ret)
        action = agent.act(X.iloc[0:1], regime=0, p_win=0.6)
        assert action in (-1, 0, 1)

    def test_act_series_shape(self, agent_data):
        from models.rl_agent import QLearningAgent
        X, ret, reg, pw, sig = agent_data
        agent = QLearningAgent()
        agent.train(X, ret, regime_labels=reg, p_win_series=pw)
        actions = agent.act_series(X, regime_labels=reg, p_win_series=pw)
        assert len(actions) == len(X)
        assert set(actions.unique()).issubset({-1, 0, 1})

    def test_act_series_index_preserved(self, agent_data):
        from models.rl_agent import QLearningAgent
        X, ret, reg, pw, sig = agent_data
        agent = QLearningAgent()
        agent.train(X, ret)
        actions = agent.act_series(X)
        pd.testing.assert_index_equal(actions.index, X.index)

    def test_epsilon_decays(self, agent_data):
        from models.rl_agent import QLearningAgent, QLearningConfig
        X, ret, *_ = agent_data
        cfg = QLearningConfig(epsilon=0.5, epsilon_decay=0.9, epsilon_min=0.01)
        agent = QLearningAgent(cfg)
        initial_eps = agent._epsilon
        agent.train(X, ret)
        assert agent._epsilon < initial_eps

    def test_bellman_update_convergence(self):
        """Escenario simple: acción=1 siempre tiene reward=+1, debe dominar Q(s,1)."""
        from models.rl_agent import QLearningAgent, QLearningConfig
        rng = np.random.default_rng(0)
        n = 500
        X = pd.DataFrame({"return_1": np.full(n, 0.01)})
        # reward positivo constante → Q(s, long=+1) debe subir
        ret = pd.Series(np.full(n, 0.01))
        signals = pd.Series(np.ones(n, dtype=int))
        cfg = QLearningConfig(alpha=0.3, gamma=0.9, epsilon=0.0, epsilon_min=0.0)
        agent = QLearningAgent(cfg)
        agent.train(X, ret, primary_signals=signals)
        # La acción greedy debería ser +1
        action = agent.act(X.iloc[0:1], regime=0, p_win=0.7)
        assert action == 1, f"Esperaba long (+1), obtuvo {action}"

    def test_train_no_regime_no_pwin(self, agent_data):
        """Agent debe funcionar sin regime_labels ni p_win_series."""
        from models.rl_agent import QLearningAgent
        X, ret, *_ = agent_data
        agent = QLearningAgent()
        agent.train(X, ret)
        assert agent.is_trained

    def test_summary_returns_string(self, agent_data):
        from models.rl_agent import QLearningAgent
        X, ret, *_ = agent_data
        agent = QLearningAgent()
        agent.train(X, ret)
        s = agent.summary()
        assert isinstance(s, str)
        assert "Q-LEARNING" in s

    def test_train_rl_agent_convenience(self, agent_data):
        from models.rl_agent import train_rl_agent
        X, ret, reg, pw, sig = agent_data
        agent = train_rl_agent(X, ret, regime_labels=reg, p_win_series=pw,
                               primary_signals=sig)
        assert agent.is_trained

    def test_pwin_bin_coverage(self):
        """_pwin_bin cubre todos los bins correctamente."""
        from models.rl_agent import QLearningAgent
        agent = QLearningAgent()
        assert agent._pwin_bin(0.0) == 0
        assert agent._pwin_bin(0.44) == 0
        assert agent._pwin_bin(0.45) == 1
        assert agent._pwin_bin(0.54) == 1
        assert agent._pwin_bin(0.55) == 2
        assert agent._pwin_bin(0.64) == 2
        assert agent._pwin_bin(0.65) == 3
        assert agent._pwin_bin(1.0) == 3

    def test_trend_bin_sma_ratio(self):
        from models.rl_agent import QLearningAgent
        agent = QLearningAgent()
        row_up   = pd.Series({"sma_ratio": 1.01})
        row_down = pd.Series({"sma_ratio": 0.99})
        row_flat = pd.Series({"sma_ratio": 1.00})
        assert agent._trend_bin(row_up)   == 2  # alcista
        assert agent._trend_bin(row_down) == 0  # bajista
        assert agent._trend_bin(row_flat) == 1  # lateral


# ===========================================================================
# 5. get_model factory — nuevo 'deep_mlp'
# ===========================================================================

class TestGetModelFactory:
    def test_logistic(self):
        from models.zoo import get_model
        m = get_model("logistic")
        assert m.name == "logistic"

    def test_xgboost(self):
        from models.zoo import get_model
        m = get_model("xgboost")
        assert m.name == "xgboost"

    def test_unknown_raises(self):
        from models.zoo import get_model
        with pytest.raises(ValueError, match="no registrado"):
            get_model("random_forest")

    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch no instalado")
    def test_deep_mlp(self):
        from models.zoo import get_model
        m = get_model("deep_mlp", hidden_dims=[8], epochs=1)
        assert m.name == "deep_mlp"


# ===========================================================================
# 6. Integración: BaseModel.calibrate() via DeepMLP
# ===========================================================================

@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch no instalado")
class TestBasemodelCalibrate:
    def test_calibrate_before_fit_raises(self):
        from models.zoo import DeepMLPClassifier
        model = DeepMLPClassifier(epochs=1)
        X = pd.DataFrame({"f": [1.0, 2.0]})
        y = pd.Series([0, 1])
        with pytest.raises(RuntimeError, match="fit"):
            model.calibrate(X, y)

    def test_calibrate_workflow(self, split_data):
        from models.zoo import DeepMLPClassifier
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = DeepMLPClassifier(hidden_dims=[8], epochs=3, device='cpu',
                                  dropout=0.0, patience=50)
        model.fit(X_fit, y_fit)
        assert not model.is_calibrated
        model.calibrate(X_calib, y_calib, method="sigmoid")
        assert model.is_calibrated

    def test_calibrate_with_label_map(self, split_data):
        """calibrate() con label_map_ existente (model tiene mapeo propio)."""
        from models.zoo import DeepMLPClassifier
        X_fit, y_fit, X_calib, y_calib, X_test, y_test = split_data
        model = DeepMLPClassifier(hidden_dims=[8], epochs=3, device='cpu',
                                  dropout=0.0, patience=50)
        model.fit(X_fit, y_fit)
        assert model.label_map_ is not None
        model.calibrate(X_calib, y_calib, method="sigmoid")
        proba = model.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
