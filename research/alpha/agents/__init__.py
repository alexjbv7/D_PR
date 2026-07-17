"""
AlphaAgent adapters (ADR-042 §3.1) — one self-contained module per hypothesis.

Exports are resolved lazily (PEP 562, same pattern as ``models.drl``) so that
torch-free consumers can import ``XgbAlphaAgent`` without pulling in torch;
``DqnAlphaAgent`` imports torch on first attribute access.
"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    # DQN adapter (requires torch)
    "DqnAlphaAgent": "alpha.agents.dqn_agent",
    "DQN_HYPOTHESIS": "alpha.agents.dqn_agent",
    "DQN_CONFIG": "alpha.agents.dqn_agent",
    "calibrator_sidecar_path": "alpha.agents.dqn_agent",
    # OOS calibration for DQN p_win (A-003)
    "fit_dqn_fold_calibrator": "alpha.agents.dqn_calibration",
    "collect_dqn_calibration_pairs": "alpha.agents.dqn_calibration",
    # XGBoost adapter (torch-free)
    "XgbAlphaAgent": "alpha.agents.xgb_agent",
    "XGB_HYPOTHESIS": "alpha.agents.xgb_agent",
    "XGB_CONFIG": "alpha.agents.xgb_agent",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Resolve a public symbol from its submodule on first access."""
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    return getattr(import_module(module), name)


def __dir__() -> list[str]:
    return __all__
