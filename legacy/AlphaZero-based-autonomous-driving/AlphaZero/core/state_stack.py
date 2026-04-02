import highway_env
import numpy as np

try:
    from core.settings import StackConfig
except ModuleNotFoundError as exc:
    if exc.name != "core":
        raise
    from .settings import StackConfig


class StateStackManager:
    """Build and update occupancy-grid stacks with ego-speed context planes."""

    def __init__(
        self,
        history_length=5,
        append_raw_ego_speed_plane=False,
        mark_ego_in_init=False,
        presence_feature_index=0,
        lane_feature_index=1,
        presence_feature_name="presence",
        lane_feature_name="on_lane",
        lane_feature_fallback_name="on_road",
    ):
        self.history_length = history_length
        self.append_raw_ego_speed_plane = append_raw_ego_speed_plane
        self.mark_ego_in_init = mark_ego_in_init
        self.presence_feature_index = presence_feature_index
        self.lane_feature_index = lane_feature_index
        self.presence_feature_name = presence_feature_name
        self.lane_feature_name = lane_feature_name
        self.lane_feature_fallback_name = lane_feature_fallback_name

    def _plane_count(self) -> int:
        return self.history_length + 5 + int(self.append_raw_ego_speed_plane)

    def _lane_plane_index(self) -> int:
        return self.history_length

    def _speed_norm_plane_index(self) -> int:
        return self.history_length + 1

    def _speed_delta_plane_index(self) -> int:
        return self.history_length + 2

    def _min_speed_plane_index(self) -> int:
        return self.history_length + 3

    def _max_speed_plane_index(self) -> int:
        return self.history_length + 4

    def _raw_ego_speed_plane_index(self) -> int:
        return self.history_length + 5

    def init_stack(self, grid_size, ego_position=None):
        width, height = grid_size
        stack = np.zeros((self._plane_count(), width, height), dtype=np.float32)
        if self.mark_ego_in_init:
            if ego_position is None:
                ego_position = (width // 2, height // 2)
            stack[: self.history_length, ego_position[0], ego_position[1]] = 1.0
        return stack

    @staticmethod
    def _normalize_to_range(value, min_value, max_value):
        span = max(float(max_value) - float(min_value), 1e-6)
        normalized = 2.0 * (float(value) - float(min_value)) / span - 1.0
        return float(np.clip(normalized, -1.0, 1.0))

    @staticmethod
    def _denormalize_from_range(value, min_value, max_value):
        span = max(float(max_value) - float(min_value), 1e-6)
        return 0.5 * (float(value) + 1.0) * span + float(min_value)

    @staticmethod
    def _resolve_speed_range(ego_vehicle):
        target_speeds = getattr(ego_vehicle, "target_speeds", None)
        if target_speeds is not None and len(target_speeds) > 0:
            min_speed = float(np.min(target_speeds))
            max_speed = float(np.max(target_speeds))
        else:
            min_speed = float(getattr(ego_vehicle, "MIN_SPEED", -40.0))
            max_speed = float(getattr(ego_vehicle, "MAX_SPEED", 40.0))

        if min_speed > max_speed:
            min_speed, max_speed = max_speed, min_speed
        if np.isclose(min_speed, max_speed):
            max_speed = min_speed + 1e-6
        return min_speed, max_speed

    @staticmethod
    def _resolve_speed_reference_range(ego_vehicle, min_speed, max_speed):
        reference_min = float(getattr(ego_vehicle, "MIN_SPEED", min_speed))
        reference_max = float(getattr(ego_vehicle, "MAX_SPEED", max_speed))
        reference_min = min(reference_min, min_speed)
        reference_max = max(reference_max, max_speed)

        if reference_min > reference_max:
            reference_min, reference_max = reference_max, reference_min
        if np.isclose(reference_min, reference_max):
            reference_max = reference_min + 1e-6
        return reference_min, reference_max

    def _has_initialized_speed_planes(self, stack):
        return not (
            np.isclose(stack[self._speed_norm_plane_index(), 0, 0], 0.0)
            and np.isclose(stack[self._speed_delta_plane_index(), 0, 0], 0.0)
            and np.isclose(stack[self._min_speed_plane_index(), 0, 0], 0.0)
            and np.isclose(stack[self._max_speed_plane_index(), 0, 0], 0.0)
        )

    def _recover_previous_speed(self, stack, reference_min_speed, reference_max_speed):
        if not self._has_initialized_speed_planes(stack):
            return None

        previous_min_speed = self._denormalize_from_range(
            stack[self._min_speed_plane_index(), 0, 0],
            reference_min_speed,
            reference_max_speed,
        )
        previous_max_speed = self._denormalize_from_range(
            stack[self._max_speed_plane_index(), 0, 0],
            reference_min_speed,
            reference_max_speed,
        )
        if np.isclose(previous_min_speed, previous_max_speed):
            return None

        return self._denormalize_from_range(
            stack[self._speed_norm_plane_index(), 0, 0],
            previous_min_speed,
            previous_max_speed,
        )

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
        stack[self._lane_plane_index()] = new_grid[lane_index]

        if env is None:
            return stack

        ego_vehicle = self._extract_ego_vehicle(env)
        ego_speed = float(ego_vehicle.speed)
        min_speed, max_speed = self._resolve_speed_range(ego_vehicle)
        reference_min_speed, reference_max_speed = self._resolve_speed_reference_range(
            ego_vehicle,
            min_speed,
            max_speed,
        )
        previous_speed = self._recover_previous_speed(
            stack,
            reference_min_speed,
            reference_max_speed,
        )

        speed_span = max(max_speed - min_speed, 1e-6)
        speed_norm = self._normalize_to_range(ego_speed, min_speed, max_speed)
        speed_delta_norm = (
            0.0
            if previous_speed is None
            else float(np.clip((ego_speed - previous_speed) / speed_span, -1.0, 1.0))
        )
        min_speed_norm = self._normalize_to_range(
            min_speed,
            reference_min_speed,
            reference_max_speed,
        )
        max_speed_norm = self._normalize_to_range(
            max_speed,
            reference_min_speed,
            reference_max_speed,
        )

        stack[self._speed_norm_plane_index()].fill(speed_norm)
        stack[self._speed_delta_plane_index()].fill(speed_delta_norm)
        stack[self._min_speed_plane_index()].fill(min_speed_norm)
        stack[self._max_speed_plane_index()].fill(max_speed_norm)

        if self.append_raw_ego_speed_plane:
            stack[self._raw_ego_speed_plane_index()].fill(ego_speed)

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
        "update_state_stack expects (env, stack, new_grid) or (stack, new_grid) arguments."
    )


def _stack_manager_from_config(stack_config: StackConfig) -> StateStackManager:
    return StateStackManager(
        history_length=stack_config.history_length,
        append_raw_ego_speed_plane=stack_config.append_raw_ego_speed_plane,
        mark_ego_in_init=stack_config.mark_ego_in_init,
        presence_feature_index=stack_config.presence_feature_index,
        lane_feature_index=stack_config.lane_feature_index,
        presence_feature_name=stack_config.presence_feature_name,
        lane_feature_name=stack_config.lane_feature_name,
        lane_feature_fallback_name=stack_config.lane_feature_fallback_name,
    )


def init_state_stack(stack_config: StackConfig):
    manager = _stack_manager_from_config(stack_config)
    return manager.init_stack(stack_config.grid_size, stack_config.ego_position)


def update_state_stack(*args, stack_config: StackConfig):
    env, stack, new_grid = _parse_stack_update_args(args)
    manager = _stack_manager_from_config(stack_config)
    return manager.update_stack(stack=stack, new_grid=new_grid, env=env)


def init_stack_of_grid(grid_size, ego_position, history_length=5):
    """Initialize the default stack with normalized ego-speed context planes."""
    return init_state_stack(
        StackConfig(
            grid_size=grid_size,
            ego_position=ego_position,
            history_length=history_length,
            append_raw_ego_speed_plane=False,
            mark_ego_in_init=True,
        )
    )


def get_stack_of_grid(*args, history_length=5):
    """Update the default stack with normalized ego-speed context planes."""
    return update_state_stack(
        *args,
        stack_config=StackConfig(
            history_length=history_length,
            append_raw_ego_speed_plane=False,
        ),
    )


def init_stack_of_grid_with_raw_ego_speed(grid_size, ego_position, history_length=5):
    """Initialize the stack and append a raw ego-speed plane."""
    return init_state_stack(
        StackConfig(
            grid_size=grid_size,
            ego_position=ego_position,
            history_length=history_length,
            append_raw_ego_speed_plane=True,
            mark_ego_in_init=True,
        )
    )


def get_stack_of_grid_with_raw_ego_speed(*args, history_length=5):
    """Update the stack variant that also carries a raw ego-speed plane."""
    return update_state_stack(
        *args,
        stack_config=StackConfig(
            history_length=history_length,
            append_raw_ego_speed_plane=True,
        ),
    )
