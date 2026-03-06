try:
    from network.alphazero_network import AlphaZeroNetwork
except ModuleNotFoundError as exc:
    if exc.name != "network":
        raise
    from .network.alphazero_network import AlphaZeroNetwork

__all__ = ["AlphaZeroNetwork"]
