try:
    from environment.config import EnvironmentFactory, init_env
except ModuleNotFoundError as exc:
    if exc.name != "environment":
        raise
    from .environment.config import EnvironmentFactory, init_env

__all__ = ["EnvironmentFactory", "init_env"]

