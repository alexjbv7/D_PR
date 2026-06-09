"""Deep reinforcement learning models (ADR-036-038)."""

from models.drl.backbone import ResBlock, TradingResMLP, init_weights
from models.drl.dqn import ReplayBuffer, TradingDQN, Transition
from models.drl.dqn_trainer import DQNConfig, DQNTrainer, EpisodeStats
from models.drl.ppo import PPOConfig, PPOTrainer, RolloutBuffer, TradingActorCritic, UpdateStats
from models.drl.sac import SACConfig, SACTrainer, SACUpdateStats, TradingDiscreteActor, TradingQNetwork

__all__ = [
    # backbone
    "ResBlock",
    "TradingResMLP",
    "init_weights",
    # DQN
    "ReplayBuffer",
    "TradingDQN",
    "Transition",
    # DQN trainer
    "DQNConfig",
    "DQNTrainer",
    "EpisodeStats",
    # PPO
    "PPOConfig",
    "PPOTrainer",
    "RolloutBuffer",
    "TradingActorCritic",
    "UpdateStats",
    # SAC
    "SACConfig",
    "SACTrainer",
    "SACUpdateStats",
    "TradingDiscreteActor",
    "TradingQNetwork",
]
