try:
    from core.policy import softmax_policy
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .core.policy import softmax_policy

__all__ = ["softmax_policy"]
