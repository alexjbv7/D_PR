"""
Deep reinforcement learning models (ADR-036-038) + DSR gate (ADR-040).

Exports are resolved lazily (PEP 562) so that torch-free consumers — e.g.
``models.drl.dsr_gate`` and its CPU-only tests — can import this package
without pulling in torch. ``from models.drl import DQNTrainer`` still works
exactly as before; torch is imported on first attribute access.
"""

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    # backbone
    "ResBlock": "models.drl.backbone",
    "TradingResMLP": "models.drl.backbone",
    "init_weights": "models.drl.backbone",
    # DQN
    "ReplayBuffer": "models.drl.dqn",
    "TradingDQN": "models.drl.dqn",
    "Transition": "models.drl.dqn",
    # DQN trainer
    "DQNConfig": "models.drl.dqn_trainer",
    "DQNTrainer": "models.drl.dqn_trainer",
    "EpisodeStats": "models.drl.dqn_trainer",
    # PPO
    "PPOConfig": "models.drl.ppo",
    "PPOTrainer": "models.drl.ppo",
    "RolloutBuffer": "models.drl.ppo",
    "TradingActorCritic": "models.drl.ppo",
    "UpdateStats": "models.drl.ppo",
    # SAC
    "SACConfig": "models.drl.sac",
    "SACTrainer": "models.drl.sac",
    "SACUpdateStats": "models.drl.sac",
    "TradingDiscreteActor": "models.drl.sac",
    "TradingQNetwork": "models.drl.sac",
    # DSR gate (ADR-040) — torch-free module
    "AgentSpec": "models.drl.dsr_gate",
    "GateResult": "models.drl.dsr_gate",
    "MIN_EMBARGO_BARS": "models.drl.dsr_gate",
    "buyhold_oos_returns": "models.drl.dsr_gate",
    "evaluate_drl_gate": "models.drl.dsr_gate",
    "evaluate_zero_gate": "models.drl.dsr_gate",
    "make_wf_splitter": "models.drl.dsr_gate",
    "positions_to_returns": "models.drl.dsr_gate",
    "walk_forward_oos_returns": "models.drl.dsr_gate",
    "xgb_oos_returns": "models.drl.dsr_gate",
    # Reward weight search (ADR-041 §5) — torch-free module
    "MAX_FINALISTS": "models.drl.reward_search",
    "MAX_PROXY_TRIALS": "models.drl.reward_search",
    "honest_gate_n_trials": "models.drl.reward_search",
    "proxy_validation_split": "models.drl.reward_search",
    "search_reward_weights": "models.drl.reward_search",
    "suggest_reward_weights": "models.drl.reward_search",
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
