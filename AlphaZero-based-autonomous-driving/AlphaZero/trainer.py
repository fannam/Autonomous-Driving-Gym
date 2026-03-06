try:
    from training.trainer import AlphaZeroTrainer
except ModuleNotFoundError as exc:
    if exc.name != "training":
        raise
    from .training.trainer import AlphaZeroTrainer

__all__ = ["AlphaZeroTrainer"]
