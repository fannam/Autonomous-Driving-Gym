__all__ = [
    "AlphaZeroNetwork",
    "AlphaZeroTrainer",
    "EnvironmentFactory",
    "MCTS",
    "MCTSNode",
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
    raise AttributeError(f"module 'AlphaZero' has no attribute '{name}'")
