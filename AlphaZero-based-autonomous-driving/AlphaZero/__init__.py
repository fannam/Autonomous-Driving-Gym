__all__ = [
    "AlphaZeroNetwork",
    "AlphaZeroTrainer",
    "EnvironmentFactory",
    "MCTS",
    "MCTSNode",
    "StackConfig",
    "AlphaZeroConfig",
    "CONFIG_PATH",
    "ACTIVE_SCENARIO",
    "SELF_PLAY_CONFIG",
    "INFERENCE_CONFIG",
    "EVALUATION_CONFIG",
    "init_env",
]


def __getattr__(name):
    if name == "AlphaZeroNetwork":
        from .network.alphazero_network import AlphaZeroNetwork

        return AlphaZeroNetwork
    if name == "AlphaZeroTrainer":
        from .training.trainer import AlphaZeroTrainer

        return AlphaZeroTrainer
    if name == "EnvironmentFactory":
        from .environment.config import EnvironmentFactory

        return EnvironmentFactory
    if name == "init_env":
        from .environment.config import init_env

        return init_env
    if name == "MCTS":
        from .core.mcts import MCTS

        return MCTS
    if name == "MCTSNode":
        from .core.mcts import MCTSNode

        return MCTSNode
    if name in {
        "StackConfig",
        "AlphaZeroConfig",
        "CONFIG_PATH",
        "ACTIVE_SCENARIO",
        "SELF_PLAY_CONFIG",
        "INFERENCE_CONFIG",
        "EVALUATION_CONFIG",
    }:
        from .core.settings import (
            ACTIVE_SCENARIO,
            CONFIG_PATH,
            AlphaZeroConfig,
            EVALUATION_CONFIG,
            INFERENCE_CONFIG,
            SELF_PLAY_CONFIG,
            StackConfig,
        )

        return {
            "StackConfig": StackConfig,
            "AlphaZeroConfig": AlphaZeroConfig,
            "CONFIG_PATH": CONFIG_PATH,
            "ACTIVE_SCENARIO": ACTIVE_SCENARIO,
            "SELF_PLAY_CONFIG": SELF_PLAY_CONFIG,
            "INFERENCE_CONFIG": INFERENCE_CONFIG,
            "EVALUATION_CONFIG": EVALUATION_CONFIG,
        }[name]
    raise AttributeError(f"module 'AlphaZero' has no attribute '{name}'")
