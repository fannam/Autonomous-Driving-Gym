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

    def __init__(
        self,
        history_length=5,
        include_absolute_speed=False,
        mark_ego_in_init=False,
        presence_feature_index=0,
        lane_feature_index=1,
        presence_feature_name="presence",
        lane_feature_name="on_lane",
        lane_feature_fallback_name="on_road",
    ):
        self.history_length = history_length
        self.include_absolute_speed = include_absolute_speed
        self.mark_ego_in_init = mark_ego_in_init
        self.presence_feature_index = presence_feature_index
        self.lane_feature_index = lane_feature_index
        self.presence_feature_name = presence_feature_name
        self.lane_feature_name = lane_feature_name
        self.lane_feature_fallback_name = lane_feature_fallback_name

    def init_stack(self, grid_size, ego_position=None):
        width, height = grid_size
        init_grid = np.zeros((width, height), dtype=np.float32)
        if self.mark_ego_in_init:
            if ego_position is None:
                ego_position = (width // 2, height // 2)
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

    def _resolve_feature_index(self, env, feature_names, fallback_index, channel_count):
        if channel_count <= 0:
            raise ValueError("Observation grid has no channels.")

        if env is not None:
            observation_type = getattr(env.unwrapped, "observation_type", None)
            env_feature_names = getattr(observation_type, "features", None)
            if isinstance(env_feature_names, (list, tuple)):
                for feature_name in feature_names:
                    if feature_name and feature_name in env_feature_names:
                        index = env_feature_names.index(feature_name)
                        if index < channel_count:
                            return index
        return int(np.clip(fallback_index, 0, channel_count - 1))

    def _resolve_presence_and_lane_indices(self, env, new_grid):
        channel_count = new_grid.shape[0]
        presence_index = self._resolve_feature_index(
            env=env,
            feature_names=[self.presence_feature_name],
            fallback_index=self.presence_feature_index,
            channel_count=channel_count,
        )
        lane_index = self._resolve_feature_index(
            env=env,
            feature_names=[self.lane_feature_name, self.lane_feature_fallback_name],
            fallback_index=self.lane_feature_index,
            channel_count=channel_count,
        )
        return presence_index, lane_index

    def update_stack(self, stack, new_grid, env=None):
        """
        Update stack with latest occupancy/lane grid and speed planes.

        If `env` is None, speed planes are left unchanged.
        This keeps compatibility with legacy call sites that only pass
        `(stack, observation_grid)`.
        """
        if new_grid.ndim != 3:
            raise ValueError(
                f"Expected observation grid with shape (channels, width, height), got {new_grid.shape}."
            )
        presence_index, lane_index = self._resolve_presence_and_lane_indices(env, new_grid)

        history = self.history_length
        stack[: history - 1] = stack[1:history]
        stack[history - 1] = new_grid[presence_index]
        stack[history] = new_grid[lane_index]

        if env is None:
            return stack

        ego_vehicle = self._extract_ego_vehicle(env)
        ego_speed = ego_vehicle.speed
        if hasattr(ego_vehicle, "target_speeds"):
            max_speed = float(ego_vehicle.target_speeds[-1])
            min_speed = float(ego_vehicle.target_speeds[0])
        else:
            max_speed = float(getattr(ego_vehicle, "MAX_SPEED", 40.0))
            min_speed = float(getattr(ego_vehicle, "MIN_SPEED", -40.0))
        speed_span = max(max_speed - min_speed, 1e-6)

        relative_speed_max = (ego_speed - max_speed) / speed_span
        relative_speed_min = (ego_speed - min_speed) / speed_span

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
        mark_ego_in_init=stack_config.mark_ego_in_init,
        presence_feature_index=stack_config.presence_feature_index,
        lane_feature_index=stack_config.lane_feature_index,
        presence_feature_name=stack_config.presence_feature_name,
        lane_feature_name=stack_config.lane_feature_name,
        lane_feature_fallback_name=stack_config.lane_feature_fallback_name,
    )
    return manager.init_stack(stack_config.grid_size, stack_config.ego_position)


def update_state_stack(*args, stack_config: StackConfig):
    env, stack, new_grid = _parse_stack_update_args(args)
    manager = StateStackManager(
        history_length=stack_config.history_length,
        include_absolute_speed=stack_config.include_absolute_speed,
        mark_ego_in_init=stack_config.mark_ego_in_init,
        presence_feature_index=stack_config.presence_feature_index,
        lane_feature_index=stack_config.lane_feature_index,
        presence_feature_name=stack_config.presence_feature_name,
        lane_feature_name=stack_config.lane_feature_name,
        lane_feature_fallback_name=stack_config.lane_feature_fallback_name,
    )
    return manager.update_stack(stack=stack, new_grid=new_grid, env=env)


def init_stack_of_grid(grid_size, ego_position, history_length=5):
    return init_state_stack(
        StackConfig(
            grid_size=grid_size,
            ego_position=ego_position,
            history_length=history_length,
            include_absolute_speed=False,
            mark_ego_in_init=True,
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
            mark_ego_in_init=True,
        )
    )


def get_stack_of_grid_9_layers(*args, history_length=5):
    return update_state_stack(
        *args,
        stack_config=StackConfig(history_length=history_length, include_absolute_speed=True),
    )
