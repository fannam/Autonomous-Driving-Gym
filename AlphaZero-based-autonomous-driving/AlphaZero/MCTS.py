try:
    from core.mcts import MCTS, MCTSNode
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .core.mcts import MCTS, MCTSNode

__all__ = ["MCTS", "MCTSNode"]
