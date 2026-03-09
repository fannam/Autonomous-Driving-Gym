import highway_env
import numpy as np

try:
    from core.settings import StackConfig
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .settings import StackConfig


class StateStackManager:
    """Build and update occupancy-grid stack tensors used by the network."""

    def __init__(self, history_length=5, include_absolute_speed=False):
        self.history_length = history_length
        self.include_absolute_speed = include_absolute_speed

    def init_stack(self, grid_size, ego_position):
        width, height = grid_size
        init_grid = np.zeros((width, height), dtype=np.float32)
        init_grid[ego_position] = 1.0
        history_stack = np.stack([init_grid.copy() for _ in range(self.history_length)], axis=0)

        lane_info_grid = np.zeros((width, height), dtype=np.float32)
        speed_max_grid = np.full((width, height), 0.0, dtype=np.float32)
        speed_min_grid = np.full((width, height), 0.0, dtype=np.float32)

        stack_parts = [
            history_stack,
            lane_info_grid[np.newaxis, :, :],
            speed_max_grid[np.newaxis, :, :],
            speed_min_grid[np.newaxis, :, :],
        ]

        if self.include_absolute_speed:
            ego_absolute_speed_grid = np.full((width, height), 0.0, dtype=np.float32)
            stack_parts.append(ego_absolute_speed_grid[np.newaxis, :, :])

        return np.concatenate(stack_parts, axis=0)

    def update_stack(self, stack, new_grid, env=None):
        """
        Update stack with latest occupancy/lane grid and speed planes.

        If `env` is None, speed planes are left unchanged.
        This keeps compatibility with legacy call sites that only pass
        `(stack, observation_grid)`.
        """
        history = self.history_length
        stack[: history - 1] = stack[1:history]
        stack[history - 1] = new_grid[0]
        stack[history] = new_grid[1]

        if env is None:
            return stack

        ego_vehicle = self._extract_ego_vehicle(env)
        ego_speed = ego_vehicle.speed
        max_speed = ego_vehicle.target_speeds[-1]
        min_speed = ego_vehicle.target_speeds[0]

        relative_speed_max = (ego_speed - max_speed) / (max_speed - min_speed)
        relative_speed_min = (ego_speed - min_speed) / (max_speed - min_speed)

        speed_max_grid = np.full(stack.shape[1:], relative_speed_max, dtype=np.float32)
        speed_min_grid = np.full(stack.shape[1:], relative_speed_min, dtype=np.float32)

        stack[history + 1] = speed_max_grid
        stack[history + 2] = speed_min_grid

        if self.include_absolute_speed:
            ego_absolute_speed_grid = np.full(stack.shape[1:], ego_speed, dtype=np.float32)
            stack[history + 3] = ego_absolute_speed_grid

        return stack

    @staticmethod
    def _extract_ego_vehicle(env):
        for vehicle in env.unwrapped.road.vehicles:
            if isinstance(vehicle, highway_env.vehicle.controller.MDPVehicle):
                return vehicle
        return env.unwrapped.road.vehicles[0]


def _parse_stack_update_args(args):
    if len(args) == 3:
        env, stack, new_grid = args
        return env, stack, new_grid
    if len(args) == 2:
        stack, new_grid = args
        return None, stack, new_grid
    raise TypeError(
        "get_stack_of_grid expects (env, stack, new_grid) or (stack, new_grid) arguments."
    )


def init_state_stack(stack_config: StackConfig):
    manager = StateStackManager(
        history_length=stack_config.history_length,
        include_absolute_speed=stack_config.include_absolute_speed,
    )
    return manager.init_stack(stack_config.grid_size, stack_config.ego_position)


def update_state_stack(*args, stack_config: StackConfig):
    env, stack, new_grid = _parse_stack_update_args(args)
    manager = StateStackManager(
        history_length=stack_config.history_length,
        include_absolute_speed=stack_config.include_absolute_speed,
    )
    return manager.update_stack(stack=stack, new_grid=new_grid, env=env)


def init_stack_of_grid(grid_size, ego_position, history_length=5):
    return init_state_stack(
        StackConfig(
            grid_size=grid_size,
            ego_position=ego_position,
            history_length=history_length,
            include_absolute_speed=False,
        )
    )


def get_stack_of_grid(*args, history_length=5):
    return update_state_stack(
        *args,
        stack_config=StackConfig(history_length=history_length, include_absolute_speed=False),
    )


def init_stack_of_grid_9_layers(grid_size, ego_position, history_length=5):
    return init_state_stack(
        StackConfig(
            grid_size=grid_size,
            ego_position=ego_position,
            history_length=history_length,
            include_absolute_speed=True,
        )
    )


def get_stack_of_grid_9_layers(*args, history_length=5):
    return update_state_stack(
        *args,
        stack_config=StackConfig(history_length=history_length, include_absolute_speed=True),
    )
