try:
    from core.state_stack import (
        StateStackManager,
        get_stack_of_grid_9_layers as get_stack_of_grid,
        init_stack_of_grid_9_layers as init_stack_of_grid,
    )
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .core.state_stack import (
        StateStackManager,
        get_stack_of_grid_9_layers as get_stack_of_grid,
        init_stack_of_grid_9_layers as init_stack_of_grid,
    )

__all__ = ["StateStackManager", "init_stack_of_grid", "get_stack_of_grid"]
