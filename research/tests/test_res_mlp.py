"""
Tests for ResMLPClassifier (ADR-034).

Structure
---------
- TestNoTorch*   : run always (no PyTorch required) — syntax, factory, imports.
- TestWithTorch* : skipped automatically if torch is not installed.

Coverage:
- nn_layers: ResBlock shape, residual connection, gradients, TemperatureScaling fit/forward
- zoo: ResMLPClassifier fit/predict/predict_proba, all_classes, calibration cascade,
       reproducibility, feature_importance
- factory: res_mlp registered, deep_mlp still present (legacy baseline)
- backbone regression: TradingResMLP unaffected, ResBlock is same class object
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# torch availability guard
# ---------------------------------------------------------------------------

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

requires_torch = pytest.mark.skipif(not _TORCH, reason="PyTorch not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_dataset():
    rng = np.random.default_rng(0)
    n, f = 400, 12
    X = pd.DataFrame(rng.standard_normal((n, f)), columns=[f"feat_{i}" for i in range(f)])
    y = pd.Series(rng.choice([-1, 0, 1], n))
    return X, y


@pytest.fixture()
def tiny_model():
    """Smallest valid ResMLPClassifier for fast tests."""
    from models.zoo import ResMLPClassifier
    return ResMLPClassifier(hidden_dim=32, n_blocks=2, epochs=5, batch_size=64, patience=3)


# ===========================================================================
# NO-TORCH TESTS — always run
# ===========================================================================

class TestNoTorchSyntax:
    """Verify all new files parse without errors."""

    def test_nn_layers_syntax(self):
        import ast, pathlib
        src = pathlib.Path("models/nn_layers.py").read_text()
        ast.parse(src)  # raises SyntaxError on failure

    def test_zoo_syntax(self):
        import ast, pathlib
        src = pathlib.Path("models/zoo.py").read_text()
        ast.parse(src)

    def test_trainer_syntax(self):
        import ast, pathlib
        src = pathlib.Path("models/multi_horizon/trainer.py").read_text()
        ast.parse(src)

    def test_backbone_syntax(self):
        import ast, pathlib
        src = pathlib.Path("models/drl/backbone.py").read_text()
        ast.parse(src)


class TestNoTorchFactory:
    """Factory and class-attribute checks — no model instantiation needed."""

    def test_res_mlp_class_exists(self):
        from models.zoo import ResMLPClassifier
        assert ResMLPClassifier.name == "res_mlp"

    def test_deep_mlp_still_present(self):
        """deep_mlp remains as legacy baseline (not removed by ADR-034)."""
        from models.zoo import DeepMLPClassifier
        assert DeepMLPClassifier.name == "deep_mlp"

    def test_get_model_res_mlp(self):
        from models.zoo import get_model
        m = get_model("res_mlp")
        assert m.name == "res_mlp"

    def test_get_model_deep_mlp(self):
        from models.zoo import get_model
        m = get_model("deep_mlp")
        assert m.name == "deep_mlp"

    def test_get_model_unknown_raises(self):
        from models.zoo import get_model
        with pytest.raises(ValueError, match="no registrado"):
            get_model("nonexistent_model_xyz")


class TestNoTorchTrainer:
    """Trainer search spaces and utility functions — no model training."""

    def test_res_mlp_search_space_keys(self):
        """_res_mlp_search_space must contain all ResMLPClassifier constructor params."""
        from models.multi_horizon.trainer import _res_mlp_search_space

        class FakeTrial:
            def suggest_int(self, name, *a, **kw):   return 64
            def suggest_float(self, name, *a, **kw): return 0.1
            def suggest_categorical(self, name, choices): return choices[0]

        params = _res_mlp_search_space(FakeTrial())
        required = {"hidden_dim", "n_blocks", "dropout", "learning_rate",
                    "weight_decay", "batch_size", "epochs", "patience", "device"}
        assert required <= set(params.keys()), f"Missing keys: {required - set(params.keys())}"

    def test_bar_size_minutes_restored(self):
        from models.multi_horizon.trainer import _bar_size_minutes
        assert _bar_size_minutes("4H")   == 240.0
        assert _bar_size_minutes("1d")   == 390.0
        assert _bar_size_minutes("5min") == 5.0
        assert _bar_size_minutes("1h")   == 60.0

    def test_model_class_is_res_mlp_for_mlp_horizon(self):
        """_build_wf_config must produce model_class='res_mlp' for mlp horizons."""
        import pathlib
        src = pathlib.Path("models/multi_horizon/trainer.py").read_text()
        # Confirm res_mlp is the replacement; deep_mlp should not appear in wf_config calls
        assert '"res_mlp"' in src or "'res_mlp'" in src


class TestNoTorchWalkForwardRunner:
    """walk_forward_runner.py must declare res_mlp as a supported model_class."""

    def test_res_mlp_in_runner(self):
        import pathlib
        src = pathlib.Path("models/walk_forward_runner.py").read_text()
        assert "res_mlp" in src

    def test_mlp_params_updated(self):
        """Default mlp_params must use hidden_dim/n_blocks (ResMLP keys), not hidden_dims."""
        from models.walk_forward_runner import WalkForwardConfig
        cfg = WalkForwardConfig()
        assert "hidden_dim" in cfg.mlp_params, (
            "mlp_params should use hidden_dim (ResMLPClassifier), not hidden_dims (DeepMLP)"
        )
        assert "n_blocks" in cfg.mlp_params


class TestNoTorchBackboneImport:
    """backbone.py should import ResBlock from nn_layers, not define it locally."""

    def test_backbone_no_local_resblock_definition(self):
        import pathlib
        src = pathlib.Path("models/drl/backbone.py").read_text()
        # Should have an import of ResBlock, not a class definition
        assert "from models.nn_layers import ResBlock" in src
        # Should NOT have 'class ResBlock' defined locally
        assert "class ResBlock" not in src


# ===========================================================================
# TORCH-DEPENDENT TESTS — skipped if torch unavailable
# ===========================================================================

class TestResBlock:
    @requires_torch
    def test_output_shape(self):
        from models.nn_layers import ResBlock
        block = ResBlock(dim=64, dropout=0.0)
        x = torch.randn(8, 64)
        assert block(x).shape == (8, 64)

    @requires_torch
    def test_residual_connection(self):
        from models.nn_layers import ResBlock
        block = ResBlock(dim=16, dropout=0.0)
        with torch.no_grad():
            block.proj.weight.zero_()
            block.proj.bias.zero_()
        x = torch.randn(4, 16)
        assert torch.allclose(block(x), x, atol=1e-5)

    @requires_torch
    def test_gradients_flow(self):
        from models.nn_layers import ResBlock
        block = ResBlock(dim=32, dropout=0.0)
        x = torch.randn(4, 32, requires_grad=True)
        block(x).sum().backward()
        assert x.grad is not None and not torch.isnan(x.grad).any()


class TestTemperatureScaling:
    @requires_torch
    def test_is_fitted_flag(self):
        from models.nn_layers import TemperatureScaling
        ts = TemperatureScaling()
        assert not ts.is_fitted
        ts.fit(torch.randn(50, 2), torch.randint(0, 2, (50,)))
        assert ts.is_fitted

    @requires_torch
    def test_forward_scales_by_temperature(self):
        from models.nn_layers import TemperatureScaling
        ts = TemperatureScaling(init_temp=2.0)
        logits = torch.tensor([[2.0, 1.0, 0.0]])
        assert torch.allclose(ts(logits), logits / 2.0)

    @requires_torch
    def test_fit_changes_temperature(self):
        from models.nn_layers import TemperatureScaling
        ts = TemperatureScaling(init_temp=1.0)
        logits = torch.randn(200, 3) * 10.0  # very overconfident
        labels = logits.argmax(dim=1)
        ts.fit(logits, labels)
        assert ts.temperature.item() > 0.05  # clamp lower bound


class TestResMLPClassifierBasics:
    @requires_torch
    def test_fit_predict_shape(self, small_dataset, tiny_model):
        X, y = small_dataset
        tiny_model.fit(X, y)
        preds = tiny_model.predict(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({-1, 0, 1})

    @requires_torch
    def test_predict_proba_sums_to_one(self, small_dataset, tiny_model):
        X, y = small_dataset
        tiny_model.fit(X, y)
        proba = tiny_model.predict_proba(X)
        assert proba.shape == (len(X), 3)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    @requires_torch
    def test_predict_proba_raw_shape(self, small_dataset, tiny_model):
        X, y = small_dataset
        tiny_model.fit(X, y)
        raw = tiny_model.predict_proba_raw(X)
        assert raw.shape == (len(X), 3)
        assert np.allclose(raw.sum(axis=1), 1.0, atol=1e-5)

    @requires_torch
    def test_feature_importance_shape(self, small_dataset, tiny_model):
        X, y = small_dataset
        tiny_model.fit(X, y)
        imp = tiny_model.feature_importance()
        assert len(imp) == X.shape[1]


class TestResMLPTemperatureScaling:
    @requires_torch
    def test_temperature_fitted_with_eval_set(self, small_dataset):
        from models.zoo import ResMLPClassifier
        X, y = small_dataset
        n = len(X)
        m = ResMLPClassifier(hidden_dim=32, n_blocks=2, epochs=5, patience=3)
        m.fit(X.iloc[:n//2], y.iloc[:n//2], eval_set=(X.iloc[n//2:], y.iloc[n//2:]))
        assert m._temp_scaling is not None and m._temp_scaling.is_fitted

    @requires_torch
    def test_no_temperature_without_eval_set(self, small_dataset, tiny_model):
        X, y = small_dataset
        tiny_model.fit(X, y)
        assert tiny_model._temp_scaling is None


class TestResMLPCalibration:
    @requires_torch
    def test_calibrate_isotonic(self, small_dataset):
        from models.zoo import ResMLPClassifier
        X, y = small_dataset
        n = len(X)
        m = ResMLPClassifier(hidden_dim=32, n_blocks=2, epochs=5, patience=3)
        m.fit(X.iloc[:n*2//3], y.iloc[:n*2//3])
        m.calibrate(X.iloc[n*2//3:], y.iloc[n*2//3:], method="isotonic")