from __future__ import annotations

import copy
from collections import OrderedDict
from itertools import product
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from gymnasium import spaces

from highway_env import utils
from highway_env.envs.common.finite_mdp import compute_ttc_grid
from highway_env.envs.common.graphics import EnvViewer
from highway_env.road.lane import AbstractLane, CircularLane, SineLane, StraightLane
from highway_env.utils import Vector
from highway_env.vehicle.kinematics import Vehicle


if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType:
    def __init__(self, env: AbstractEnv, **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class GrayscaleObservation(ObservationType):
    """
    An observation class that collects directly what the simulator renders.

    Also stacks the collected frames as in the nature DQN.
    The observation shape is C x W x H.

    Specific keys are expected in the configuration dictionary passed.
    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "observation_shape": (84, 84)
            "stack_size": 4,
            "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion,
        }
    """

    def __init__(
        self,
        env: AbstractEnv,
        observation_shape: tuple[int, int],
        stack_size: int,
        weights: list[float],
        scaling: float | None = None,
        centering_position: list[float] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(env)
        self.observation_shape = observation_shape
        self.shape = (stack_size,) + self.observation_shape
        self.weights = weights
        self.obs = np.zeros(self.shape, dtype=np.uint8)

        # The viewer configuration can be different between this observation and env.render() (typically smaller)
        viewer_config = env.config.copy()
        viewer_config.update(
            {
                "offscreen_rendering": True,
                "screen_width": self.observation_shape[0],
                "screen_height": self.observation_shape[1],
                "scaling": scaling or viewer_config["scaling"],
                "centering_position": centering_position
                or viewer_config["centering_position"],
            }
        )
        self.viewer = EnvViewer(env, config=viewer_config)

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.shape, low=0, high=255, dtype=np.uint8)

    def observe(self) -> np.ndarray:
        new_obs = self._render_to_grayscale()
        self.obs = np.roll(self.obs, -1, axis=0)
        self.obs[-1, :, :] = new_obs
        return self.obs

    def _render_to_grayscale(self) -> np.ndarray:
        self.viewer.observer_vehicle = self.observer_vehicle
        self.viewer.display()
        raw_rgb = self.viewer.get_image()  # H x W x C
        raw_rgb = np.moveaxis(raw_rgb, 0, 1)
        return np.dot(raw_rgb[..., :3], self.weights).clip(0, 255).astype(np.uint8)


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: AbstractEnv, horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(
                shape=self.observe().shape, low=0, high=1, dtype=np.float32
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(
                (3, 3, int(self.horizon * self.env.config["policy_frequency"]))
            )
        grid = compute_ttc_grid(
            self.env,
            vehicle=self.observer_vehicle,
            time_quantization=1 / self.env.config["policy_frequency"],
            horizon=self.horizon,
        )
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0 : lf + 1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0 : vf + 1, :, :]
        return clamped_grid.astype(np.float32)


class KinematicObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    FEATURES: list[str] = ["presence", "x", "y", "vx", "vy"]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] = None,
        vehicles_count: int = 5,
        features_range: dict[str, list[float]] = None,
        absolute: bool = False,
        order: str = "sorted",
        normalize: bool = True,
        clip: bool = True,
        see_behind: bool = False,
        observe_intentions: bool = False,
        include_obstacles: bool = True,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: Use absolute coordinates
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contains the vehicles behind
        :param observe_intentions: Observe the destinations of other vehicles
        """
        super().__init__(env)
        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions
        self.include_obstacles = include_obstacles

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-np.inf,
            high=np.inf,
            dtype=np.float32,
        )

    def normalize_obs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            side_lanes = self.env.road.network.all_side_lanes(
                self.observer_vehicle.lane_index
            )
            self.features_range = {
                "x": [-5.0 * Vehicle.MAX_SPEED, 5.0 * Vehicle.MAX_SPEED],
                "y": [
                    -AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                    AbstractLane.DEFAULT_WIDTH * len(side_lanes),
                ],
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
                if self.clip:
                    df[feature] = np.clip(df[feature], -1, 1)
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        df = pd.DataFrame.from_records([self.observer_vehicle.to_dict()])
        # Add nearby traffic
        close_vehicles = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
            sort=self.order == "sorted",
            vehicles_only=not self.include_obstacles,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            vehicles_df = pd.DataFrame.from_records(
                [
                    v.to_dict(origin, observe_intentions=self.observe_intentions)
                    for v in close_vehicles[-self.vehicles_count + 1 :]
                ]
            )
            df = pd.concat([df, vehicles_df], ignore_index=True)

        df = df[self.features]

        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)


class OccupancyGridObservation(ObservationType):
    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: list[str] = ["presence", "vx", "vy", "on_road"]
    GRID_SIZE: list[list[float]] = [[-5.5 * 5, 5.5 * 5], [-5.5 * 5, 5.5 * 5]]
    GRID_STEP: list[int] = [5, 5]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] | None = None,
        grid_size: tuple[tuple[float, float], tuple[float, float]] | None = None,
        grid_step: tuple[float, float] | None = None,
        features_range: dict[str, list[float]] = None,
        absolute: bool = False,
        align_to_vehicle_axes: bool = False,
        clip: bool = True,
        as_image: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param grid_size: real world size of the grid [[min_x, max_x], [min_y, max_y]]
        :param grid_step: steps between two cells of the grid [step_x, step_y]
        :param features_range: a dict mapping a feature name to [min, max] values
        :param absolute: use absolute or relative coordinates
        :param align_to_vehicle_axes: if True, the grid axes are aligned with vehicle axes. Else, they are aligned
               with world axes.
        :param clip: clip the observation in [-1, 1]
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = (
            np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        )
        self.grid_step = (
            np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        )
        grid_shape = np.asarray(
            np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / self.grid_step),
            dtype=np.uint8,
        )
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute
        self.align_to_vehicle_axes = align_to_vehicle_axes
        self.clip = clip
        self.as_image = as_image

    def space(self) -> spaces.Space:
        if self.as_image:
            return spaces.Box(shape=self.grid.shape, low=0, high=255, dtype=np.uint8)
        else:
            return spaces.Box(
                shape=self.grid.shape, low=-np.inf, high=np.inf, dtype=np.float32
            )

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
                "vy": [-2 * Vehicle.MAX_SPEED, 2 * Vehicle.MAX_SPEED],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Initialize empty data
            self.grid.fill(np.nan)

            # Get nearby traffic data
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles]
            )
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                if feature in df.columns:  # A vehicle feature
                    for _, vehicle in df[::-1].iterrows():
                        x, y = vehicle["x"], vehicle["y"]
                        # Recover unnormalized coordinates for cell index
                        if "x" in self.features_range:
                            x = utils.lmap(
                                x,
                                [-1, 1],
                                [
                                    self.features_range["x"][0],
                                    self.features_range["x"][1],
                                ],
                            )
                        if "y" in self.features_range:
                            y = utils.lmap(
                                y,
                                [-1, 1],
                                [
                                    self.features_range["y"][0],
                                    self.features_range["y"][1],
                                ],
                            )
                        cell = self.pos_to_index((x, y), relative=not self.absolute)
                        if (
                            0 <= cell[0] < self.grid.shape[-2]
                            and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer, cell[0], cell[1]] = vehicle[feature]
                elif feature == "on_road":
                    self.fill_road_layer_by_lanes(layer)

            obs = self.grid

            if self.clip:
                obs = np.clip(obs, -1, 1)

            if self.as_image:
                obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)

            obs = np.nan_to_num(obs).astype(self.space().dtype)

            return obs

    def pos_to_index(self, position: Vector, relative: bool = False) -> tuple[int, int]:
        """
        Convert a world position to a grid cell index

        If align_to_vehicle_axes the cells are in the vehicle's frame, otherwise in the world frame.

        :param position: a world position
        :param relative: whether the position is already relative to the observer's position
        :return: the pair (i,j) of the cell index
        """
        if not relative:
            position -= self.observer_vehicle.position
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(
                self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position
        return (
            int(np.floor((position[0] - self.grid_size[0, 0]) / self.grid_step[0])),
            int(np.floor((position[1] - self.grid_size[1, 0]) / self.grid_step[1])),
        )

    def index_to_pos(self, index: tuple[int, int]) -> np.ndarray:
        position = np.array(
            [
                (index[0] + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
                (index[1] + 0.5) * self.grid_step[1] + self.grid_size[1, 0],
            ]
        )

        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(
                -self.observer_vehicle.heading
            )
            position = np.array([[c, s], [-s, c]]) @ position

        position += self.observer_vehicle.position
        return position

    def fill_road_layer_by_lanes(
        self, layer_index: int, lane_perception_distance: float = 100
    ) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        Here, we iterate over lanes and regularly placed waypoints on these lanes to fill the corresponding cells.
        This approach is faster if the grid is large and the road network is small.

        :param layer_index: index of the layer in the grid
        :param lane_perception_distance: lanes are rendered +/- this distance from vehicle location
        """
        lane_waypoints_spacing = np.amin(self.grid_step)
        road = self.env.road

        for _from in road.network.graph.keys():
            for _to in road.network.graph[_from].keys():
                for lane in road.network.graph[_from][_to]:
                    origin, _ = lane.local_coordinates(self.observer_vehicle.position)
                    waypoints = np.arange(
                        origin - lane_perception_distance,
                        origin + lane_perception_distance,
                        lane_waypoints_spacing,
                    ).clip(0, lane.length)
                    for waypoint in waypoints:
                        cell = self.pos_to_index(lane.position(waypoint, 0))
                        if (
                            0 <= cell[0] < self.grid.shape[-2]
                            and 0 <= cell[1] < self.grid.shape[-1]
                        ):
                            self.grid[layer_index, cell[0], cell[1]] = 1

    def fill_road_layer_by_cell(self, layer_index) -> None:
        """
        A layer to encode the onroad (1) / offroad (0) information

        In this implementation, we iterate the grid cells and check whether the corresponding world position
        at the center of the cell is onroad/offroad. This approach is faster if the grid is small and the road network large.
        """
        road = self.env.road
        for i, j in product(range(self.grid.shape[-2]), range(self.grid.shape[-1])):
            for _from in road.network.graph.keys():
                for _to in road.network.graph[_from].keys():
                    for lane in road.network.graph[_from][_to]:
                        if lane.on_lane(self.index_to_pos((i, j))):
                            self.grid[layer_index, i, j] = 1


class DetailedOccupancyGridObservation(OccupancyGridObservation):
    """
    A richer occupancy-grid encoding that preserves more vehicle information.

    Differences vs `OccupancyGridObservation`:
    - Vehicles are rasterized as rectangles using their footprint (LENGTH/WIDTH), not a single cell.
    - Additional dynamic channels are available (speed, heading, distance, TTC risk).
    - Optional split-presence channels can isolate target-controlled and IDM/background vehicles.
    - Near vehicles overwrite far vehicles on overlap, so local details are preserved.
    - `on_road` can represent full lane area (default) instead of only centerlines.
    - Optional `on_road_soft_mode` encodes area occupancy ratio per cell for smoother road masks.
    - `on_lane` encodes lane boundaries to preserve lane-limit information.
    """

    FEATURES: list[str] = [
        "presence",
        "vx",
        "vy",
        "speed",
        "cos_h",
        "sin_h",
        "distance",
        "ttc",
        "on_lane",
        "on_road",
    ]

    def __init__(
        self,
        env: AbstractEnv,
        features: list[str] | None = None,
        grid_size: tuple[tuple[float, float], tuple[float, float]] | None = None,
        grid_step: tuple[float, float] | None = None,
        features_range: dict[str, list[float]] | None = None,
        absolute: bool = False,
        align_to_vehicle_axes: bool = False,
        clip: bool = True,
        as_image: bool = False,
        include_ego_vehicle: bool = True,
        on_road_mode: str = "area",
        on_road_soft_mode: bool = False,
        on_road_subsamples: int = 3,
        presence_subsamples: int = 3,
        vehicle_footprint: bool = True,
        footprint_margin: float = 1.0,
        ttc_horizon: float = 10.0,
        distance_normalization: float = 100.0,
        **kwargs: dict,
    ) -> None:
        super().__init__(
            env=env,
            features=features if features is not None else self.FEATURES,
            grid_size=grid_size,
            grid_step=grid_step,
            features_range=features_range,
            absolute=absolute,
            align_to_vehicle_axes=align_to_vehicle_axes,
            clip=clip,
            as_image=as_image,
            **kwargs,
        )
        if on_road_mode not in {"area", "centerline"}:
            raise ValueError(
                f"Unsupported on_road_mode={on_road_mode}. "
                "Use 'area' or 'centerline'."
            )
        if on_road_subsamples < 1:
            raise ValueError("on_road_subsamples must be >= 1.")
        if presence_subsamples < 1:
            raise ValueError("presence_subsamples must be >= 1.")
        self.include_ego_vehicle = include_ego_vehicle
        self.on_road_mode = on_road_mode
        self.on_road_soft_mode = on_road_soft_mode
        self.on_road_subsamples = int(on_road_subsamples)
        self.presence_subsamples = int(presence_subsamples)
        self.vehicle_footprint = vehicle_footprint
        self.footprint_margin = footprint_margin
        self.ttc_horizon = ttc_horizon
        self.distance_normalization = distance_normalization
        self._presence_sample_x = (
            (np.arange(self.presence_subsamples, dtype=np.float32) + 0.5)
            / float(self.presence_subsamples)
            * self.grid_step[0]
        )
        self._presence_sample_y = (
            (np.arange(self.presence_subsamples, dtype=np.float32) + 0.5)
            / float(self.presence_subsamples)
            * self.grid_step[1]
        )
        self._cell_center_local_points = self._build_local_point_grid(subsamples=1)
        self._soft_local_points = self._build_local_point_grid(
            subsamples=self.on_road_subsamples
        )
        self._lane_cache_key: tuple[int, ...] | None = None
        self._lane_spatial_bounds: dict[int, tuple[np.ndarray, np.ndarray] | None] = {}
        self._lane_boundary_cache_key: tuple[tuple[int, ...], float] | None = None
        self._lane_boundary_samples: tuple[
            tuple[AbstractLane, np.ndarray, np.ndarray], ...
        ] = ()

    def __deepcopy__(self, memo):
        cls = self.__class__
        clone = cls.__new__(cls)
        memo[id(self)] = clone

        shared_attrs = {
            "_presence_sample_x",
            "_presence_sample_y",
            "_cell_center_local_points",
            "_soft_local_points",
        }
        reset_attrs = {
            "_lane_cache_key": None,
            "_lane_spatial_bounds": {},
            "_lane_boundary_cache_key": None,
            "_lane_boundary_samples": (),
        }

        for name, value in self.__dict__.items():
            if name in shared_attrs:
                setattr(clone, name, value)
            elif name == "grid":
                setattr(clone, name, value.copy())
            elif name in reset_attrs:
                setattr(clone, name, reset_attrs[name])
            else:
                setattr(clone, name, copy.deepcopy(value, memo))

        return clone

    def _build_local_point_grid(self, subsamples: int) -> np.ndarray:
        width = self.grid.shape[-2]
        height = self.grid.shape[-1]
        base_x = (
            np.arange(width, dtype=np.float32) * self.grid_step[0] + self.grid_size[0, 0]
        )
        base_y = (
            np.arange(height, dtype=np.float32) * self.grid_step[1] + self.grid_size[1, 0]
        )
        offset_x = (
            (np.arange(subsamples, dtype=np.float32) + 0.5)
            / float(subsamples)
            * self.grid_step[0]
        )
        offset_y = (
            (np.arange(subsamples, dtype=np.float32) + 0.5)
            / float(subsamples)
            * self.grid_step[1]
        )

        sample_points = []
        for dx in offset_x:
            for dy in offset_y:
                xx, yy = np.meshgrid(base_x + dx, base_y + dy, indexing="ij")
                sample_points.append(np.stack((xx, yy), axis=-1))
        return np.stack(sample_points, axis=2).astype(np.float32, copy=False)

    def _grid_local_points_to_world(self, local_points: np.ndarray) -> np.ndarray:
        local_points = np.asarray(local_points, dtype=np.float32)
        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(
                -self.observer_vehicle.heading
            )
            rotation = np.array([[c, s], [-s, c]], dtype=np.float32)
            local_points = local_points @ rotation.T
        return local_points + np.asarray(
            self.observer_vehicle.position,
            dtype=np.float32,
        )

    def _ensure_lane_spatial_cache(
        self,
    ) -> tuple[tuple[AbstractLane, ...], dict[int, tuple[np.ndarray, np.ndarray] | None]]:
        lanes = tuple(self.env.road.network.lanes_list())
        lane_cache_key = tuple(id(lane) for lane in lanes)
        if lane_cache_key != self._lane_cache_key:
            self._lane_spatial_bounds = {
                id(lane): self._lane_spatial_bounds_for(lane) for lane in lanes
            }
            self._lane_cache_key = lane_cache_key
        return lanes, self._lane_spatial_bounds

    def _lane_boundary_waypoints(
        self,
        lane: AbstractLane,
        spacing: float,
    ) -> np.ndarray:
        waypoints = np.arange(0.0, lane.length, spacing, dtype=np.float32)
        if waypoints.size == 0 or waypoints[-1] < lane.length:
            waypoints = np.concatenate(
                [waypoints, np.array([lane.length], dtype=np.float32)]
            )
        return waypoints

    def _lane_boundary_world_points_for(
        self,
        lane: AbstractLane,
        waypoints: np.ndarray,
    ) -> np.ndarray:
        lane_type = lane.__class__
        if lane_type is StraightLane:
            half_width = 0.5 * float(lane.width)
            longitudinal = waypoints[:, None]
            direction = np.asarray(lane.direction, dtype=np.float32)[None, :]
            direction_lateral = np.asarray(
                lane.direction_lateral,
                dtype=np.float32,
            )[None, :]
            centerline = (
                np.asarray(lane.start, dtype=np.float32)[None, :]
                + longitudinal * direction
            )
            return np.stack(
                (
                    centerline - half_width * direction_lateral,
                    centerline + half_width * direction_lateral,
                ),
                axis=1,
            ).astype(np.float32, copy=False)

        if lane_type is SineLane:
            half_width = 0.5 * float(lane.width)
            direction = np.asarray(lane.direction, dtype=np.float32)[None, :]
            direction_lateral = np.asarray(
                lane.direction_lateral,
                dtype=np.float32,
            )[None, :]
            lateral_offset = (
                float(lane.amplitude)
                * np.sin(float(lane.pulsation) * waypoints + float(lane.phase))
            )[:, None]
            centerline = (
                np.asarray(lane.start, dtype=np.float32)[None, :]
                + waypoints[:, None] * direction
                + lateral_offset * direction_lateral
            )
            return np.stack(
                (
                    centerline - half_width * direction_lateral,
                    centerline + half_width * direction_lateral,
                ),
                axis=1,
            ).astype(np.float32, copy=False)

        if lane_type is CircularLane:
            half_width = 0.5 * float(lane.width)
            phi = (
                float(lane.direction) * waypoints / float(lane.radius)
                + float(lane.start_phase)
            )
            unit = np.stack((np.cos(phi), np.sin(phi)), axis=1).astype(
                np.float32,
                copy=False,
            )
            center = np.asarray(lane.center, dtype=np.float32)[None, :]
            outer_radius = float(lane.radius) + half_width * float(lane.direction)
            inner_radius = float(lane.radius) - half_width * float(lane.direction)
            return np.stack(
                (
                    center + outer_radius * unit,
                    center + inner_radius * unit,
                ),
                axis=1,
            ).astype(np.float32, copy=False)

        points = np.empty((waypoints.shape[0], 2, 2), dtype=np.float32)
        for index, waypoint in enumerate(waypoints):
            half_width = 0.5 * float(lane.width_at(float(waypoint)))
            points[index, 0] = lane.position(float(waypoint), -half_width)
            points[index, 1] = lane.position(float(waypoint), half_width)
        return points

    def _ensure_lane_boundary_cache(
        self,
    ) -> tuple[tuple[AbstractLane, np.ndarray, np.ndarray], ...]:
        lanes = tuple(self.env.road.network.lanes_list())
        spacing = float(np.amin(self.grid_step))
        cache_key = (tuple(id(lane) for lane in lanes), spacing)
        if cache_key != self._lane_boundary_cache_key:
            self._lane_boundary_samples = tuple(
                (
                    lane,
                    waypoints,
                    self._lane_boundary_world_points_for(lane, waypoints),
                )
                for lane in lanes
                if (waypoints := self._lane_boundary_waypoints(lane, spacing)).size > 0
            )
            self._lane_boundary_cache_key = cache_key
        return self._lane_boundary_samples

    def _lane_spatial_bounds_for(
        self,
        lane: AbstractLane,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        lane_type = lane.__class__
        if lane_type is StraightLane:
            return self._straight_lane_spatial_bounds(lane)
        if lane_type is CircularLane:
            return self._circular_lane_spatial_bounds(lane)
        return None

    def _straight_lane_spatial_bounds(
        self,
        lane: StraightLane,
    ) -> tuple[np.ndarray, np.ndarray]:
        longitudinal_margin = float(lane.VEHICLE_LENGTH)
        half_width = 0.5 * float(lane.width)
        sample_points = np.stack(
            [
                lane.position(longitudinal, lateral)
                for longitudinal in (
                    -longitudinal_margin,
                    lane.length + longitudinal_margin,
                )
                for lateral in (-half_width, half_width)
            ],
            axis=0,
        ).astype(np.float32, copy=False)
        return (
            sample_points.min(axis=0),
            sample_points.max(axis=0),
        )

    def _circular_lane_spatial_bounds(
        self,
        lane: CircularLane,
    ) -> tuple[np.ndarray, np.ndarray]:
        longitudinal_margin = float(lane.VEHICLE_LENGTH)
        half_width = 0.5 * float(lane.width)
        inner_radius = max(0.0, float(lane.radius) - half_width)
        outer_radius = float(lane.radius) + half_width
        start_angle = lane.start_phase - lane.direction * longitudinal_margin / lane.radius
        end_angle = lane.end_phase + lane.direction * longitudinal_margin / lane.radius
        candidate_angles = (
            start_angle,
            end_angle,
            0.0,
            0.5 * np.pi,
            np.pi,
            -0.5 * np.pi,
        )
        points = []
        for angle in candidate_angles:
            unit = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            for radius in (inner_radius, outer_radius):
                point = lane.center + radius * unit
                longitudinal, lateral = lane.local_coordinates(point)
                if (
                    abs(lateral) <= half_width + 1e-6
                    and -longitudinal_margin - 1e-6
                    <= longitudinal
                    <= lane.length + longitudinal_margin + 1e-6
                ):
                    points.append(point)
        if not points:
            points = [
                lane.position(-longitudinal_margin, -lane.direction * half_width),
                lane.position(-longitudinal_margin, lane.direction * half_width),
                lane.position(
                    lane.length + longitudinal_margin,
                    -lane.direction * half_width,
                ),
                lane.position(
                    lane.length + longitudinal_margin,
                    lane.direction * half_width,
                ),
            ]
        sample_points = np.asarray(points, dtype=np.float32)
        return (
            sample_points.min(axis=0),
            sample_points.max(axis=0),
        )

    @staticmethod
    def _aabb_intersects(
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
        region_min: np.ndarray,
        region_max: np.ndarray,
    ) -> bool:
        return not (
            bounds_max[0] < region_min[0]
            or bounds_min[0] > region_max[0]
            or bounds_max[1] < region_min[1]
            or bounds_min[1] > region_max[1]
        )

    @staticmethod
    def _points_inside_aabb(
        points: np.ndarray,
        bounds_min: np.ndarray,
        bounds_max: np.ndarray,
    ) -> np.ndarray:
        return (
            (points[:, 0] >= bounds_min[0])
            & (points[:, 0] <= bounds_max[0])
            & (points[:, 1] >= bounds_min[1])
            & (points[:, 1] <= bounds_max[1])
        )

    def _lane_on_points_mask(
        self,
        lane: AbstractLane,
        world_points: np.ndarray,
        lane_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> np.ndarray:
        candidate_points = world_points
        coarse_mask = None
        if lane_bounds is not None:
            coarse_mask = self._points_inside_aabb(
                world_points,
                lane_bounds[0],
                lane_bounds[1],
            )
            if not np.any(coarse_mask):
                return np.zeros(world_points.shape[0], dtype=bool)
            candidate_points = world_points[coarse_mask]

        lane_type = lane.__class__
        if lane_type is StraightLane:
            delta = candidate_points - lane.start
            longitudinal = delta @ lane.direction
            lateral = delta @ lane.direction_lateral
            lane_mask = (
                (np.abs(lateral) <= 0.5 * lane.width)
                & (-lane.VEHICLE_LENGTH <= longitudinal)
                & (longitudinal < lane.length + lane.VEHICLE_LENGTH)
            )
        elif lane_type is CircularLane:
            delta = candidate_points - lane.center
            phi = np.arctan2(delta[:, 1], delta[:, 0])
            phi = lane.start_phase + utils.wrap_to_pi(phi - lane.start_phase)
            radius = np.linalg.norm(delta, axis=1)
            longitudinal = lane.direction * (phi - lane.start_phase) * lane.radius
            lateral = lane.direction * (lane.radius - radius)
            lane_mask = (
                (np.abs(lateral) <= 0.5 * lane.width)
                & (-lane.VEHICLE_LENGTH <= longitudinal)
                & (longitudinal < lane.length + lane.VEHICLE_LENGTH)
            )
        else:
            lane_mask = np.fromiter(
                (lane.on_lane(point) for point in candidate_points),
                dtype=bool,
                count=candidate_points.shape[0],
            )

        if coarse_mask is None:
            return lane_mask
        full_mask = np.zeros(world_points.shape[0], dtype=bool)
        full_mask[coarse_mask] = lane_mask
        return full_mask

    def _road_mask_for_local_points(self, local_points: np.ndarray) -> np.ndarray:
        world_points = self._grid_local_points_to_world(local_points.reshape(-1, 2))
        world_min = world_points.min(axis=0)
        world_max = world_points.max(axis=0)
        lanes, lane_spatial_bounds = self._ensure_lane_spatial_cache()
        on_road = np.zeros(world_points.shape[0], dtype=bool)
        for lane in lanes:
            lane_bounds = lane_spatial_bounds.get(id(lane))
            if lane_bounds is not None and not self._aabb_intersects(
                lane_bounds[0],
                lane_bounds[1],
                world_min,
                world_max,
            ):
                continue
            on_road |= self._lane_on_points_mask(
                lane,
                world_points,
                lane_bounds=lane_bounds,
            )
            if on_road.all():
                break
        return on_road.reshape(local_points.shape[:-1])

    def _vehicle_cell_coverage_patch(
        self,
        center_x: float,
        center_y: float,
        heading: float,
        length: float,
        width: float,
    ) -> tuple[int, int, np.ndarray] | None:
        if not self.vehicle_footprint:
            i, j = self._coords_to_index(center_x, center_y)
            if 0 <= i < self.grid.shape[-2] and 0 <= j < self.grid.shape[-1]:
                return i, j, np.ones((1, 1), dtype=np.float32)
            return None

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        half_l, half_w = 0.5 * length, 0.5 * width
        extent_x = abs(cos_h) * half_l + abs(sin_h) * half_w
        extent_y = abs(sin_h) * half_l + abs(cos_h) * half_w

        i_min, j_min = self._coords_to_index(center_x - extent_x, center_y - extent_y)
        i_max, j_max = self._coords_to_index(center_x + extent_x, center_y + extent_y)
        i_min = max(0, i_min)
        j_min = max(0, j_min)
        i_max = min(self.grid.shape[-2] - 1, i_max)
        j_max = min(self.grid.shape[-1] - 1, j_max)
        if i_min > i_max or j_min > j_max:
            return None

        cell_x = (
            np.arange(i_min, i_max + 1, dtype=np.float32) * self.grid_step[0]
            + self.grid_size[0, 0]
        )
        cell_y = (
            np.arange(j_min, j_max + 1, dtype=np.float32) * self.grid_step[1]
            + self.grid_size[1, 0]
        )
        sample_x = cell_x[:, None, None, None] + self._presence_sample_x[None, None, :, None]
        sample_y = cell_y[None, :, None, None] + self._presence_sample_y[None, None, None, :]

        dx = sample_x - center_x
        dy = sample_y - center_y
        local_long = cos_h * dx + sin_h * dy
        local_lat = -sin_h * dx + cos_h * dy
        inside = (np.abs(local_long) <= half_l) & (np.abs(local_lat) <= half_w)
        coverage = inside.mean(axis=(2, 3), dtype=np.float32)
        return i_min, j_min, coverage.astype(np.float32, copy=False)

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)
        if self.absolute:
            raise NotImplementedError()

        self.grid.fill(0.0)
        if "on_lane" in self.features:
            self.fill_lane_boundaries_layer(self.features.index("on_lane"))
        if "on_road" in self.features:
            on_road_layer = self.features.index("on_road")
            if self.on_road_mode == "area":
                if self.on_road_soft_mode:
                    self.fill_road_layer_by_cell_soft(on_road_layer)
                else:
                    self.fill_road_layer_by_cell(on_road_layer)
            else:
                self.fill_road_layer_by_lanes(on_road_layer)

        vehicles = (
            list(self.env.road.vehicles)
            if self.include_ego_vehicle
            else [v for v in self.env.road.vehicles if v is not self.observer_vehicle]
        )
        # Paint far vehicles first, near vehicles last, so nearby details dominate overlap.
        vehicles.sort(
            key=lambda vehicle: np.linalg.norm(vehicle.position - self.observer_vehicle.position),
            reverse=True,
        )

        controlled_vehicles = tuple(getattr(self.env, "controlled_vehicles", ()))
        controlled_vehicle_ids = {id(vehicle) for vehicle in controlled_vehicles}
        target_vehicle_ids = {
            id(vehicle)
            for vehicle in controlled_vehicles
            if vehicle is not self.observer_vehicle
        }

        for vehicle in vehicles:
            rel = vehicle.to_dict(self.observer_vehicle)
            center = self._to_grid_frame(np.array([rel["x"], rel["y"]], dtype=np.float32))
            heading = (
                vehicle.heading - self.observer_vehicle.heading
                if self.align_to_vehicle_axes
                else vehicle.heading
            )
            coverage_patch = self._vehicle_cell_coverage_patch(
                center_x=float(center[0]),
                center_y=float(center[1]),
                heading=heading,
                length=vehicle.LENGTH * self.footprint_margin,
                width=vehicle.WIDTH * self.footprint_margin,
            )
            if coverage_patch is None:
                continue
            i_min, j_min, coverage = coverage_patch
            active_mask = coverage > 0.0
            if not np.any(active_mask):
                continue
            i_max = i_min + coverage.shape[0]
            j_max = j_min + coverage.shape[1]
            vehicle_id = id(vehicle)
            is_target_vehicle = vehicle_id in target_vehicle_ids
            is_idm_vehicle = vehicle_id not in controlled_vehicle_ids
            values = self._feature_values(
                vehicle=vehicle,
                rel=rel,
                heading=heading,
                is_ego=vehicle is self.observer_vehicle,
            )
            for layer, feature in enumerate(self.features):
                if feature in {"on_road", "on_lane"}:
                    continue
                target = self.grid[layer, i_min:i_max, j_min:j_max]
                if feature == "presence":
                    np.maximum(target, coverage, out=target)
                elif feature == "presence_target":
                    if is_target_vehicle:
                        np.maximum(target, coverage, out=target)
                elif feature == "presence_idm":
                    if is_idm_vehicle:
                        np.maximum(target, coverage, out=target)
                else:
                    target[active_mask] = values.get(feature, 0.0)

        obs = self.grid
        if self.clip:
            obs = np.clip(obs, -1, 1)
        if self.as_image:
            obs = ((np.clip(obs, -1, 1) + 1) / 2 * 255).astype(np.uint8)
        return np.nan_to_num(obs).astype(self.space().dtype)

    def fill_lane_boundaries_layer(
        self, layer_index: int, lane_perception_distance: float = 100.0
    ) -> None:
        """
        Encode lane boundaries (left/right limits) into a dedicated grid layer.

        A value of 1 indicates that the grid cell belongs to a lane boundary sample.
        """
        observer = self.observer_vehicle
        observer_position = np.asarray(observer.position, dtype=np.float32)
        if self.align_to_vehicle_axes:
            c, s = np.cos(observer.heading), np.sin(observer.heading)
            rotation_t = np.array([[c, -s], [s, c]], dtype=np.float32)
        else:
            rotation_t = None

        grid_x_min, grid_x_max = self.grid_size[0]
        grid_y_min, grid_y_max = self.grid_size[1]
        step_x, step_y = float(self.grid_step[0]), float(self.grid_step[1])

        for lane, waypoints, boundary_points in self._ensure_lane_boundary_cache():
            origin, _ = lane.local_coordinates(observer_position)
            waypoint_mask = (
                (waypoints >= origin - lane_perception_distance)
                & (waypoints <= origin + lane_perception_distance)
            )
            if not np.any(waypoint_mask):
                continue

            local_points = (
                boundary_points[waypoint_mask].reshape(-1, 2) - observer_position
            )
            if rotation_t is not None:
                local_points = local_points @ rotation_t

            valid_mask = (
                (local_points[:, 0] >= grid_x_min)
                & (local_points[:, 0] < grid_x_max)
                & (local_points[:, 1] >= grid_y_min)
                & (local_points[:, 1] < grid_y_max)
            )
            if not np.any(valid_mask):
                continue

            valid_points = local_points[valid_mask]
            cell_x = np.floor((valid_points[:, 0] - grid_x_min) / step_x).astype(np.intp)
            cell_y = np.floor((valid_points[:, 1] - grid_y_min) / step_y).astype(np.intp)
            self.grid[layer_index, cell_x, cell_y] = 1.0

    def fill_road_layer_by_cell_soft(self, layer_index: int) -> None:
        """
        Soft road occupancy: each cell stores ratio of sampled points that are on-road.

        This produces fractional values in [0, 1] for partially covered cells.
        """
        self.grid[layer_index] = self._road_mask_for_local_points(
            self._soft_local_points
        ).mean(axis=2, dtype=np.float32)

    def fill_road_layer_by_cell(self, layer_index) -> None:
        self.grid[layer_index] = self._road_mask_for_local_points(
            self._cell_center_local_points
        )[:, :, 0].astype(np.float32, copy=False)

    def _to_grid_frame(self, position: np.ndarray) -> np.ndarray:
        """
        Convert a relative world-frame position to the grid frame.
        """
        if self.align_to_vehicle_axes:
            c, s = np.cos(self.observer_vehicle.heading), np.sin(
                self.observer_vehicle.heading
            )
            return np.array([[c, s], [-s, c]]) @ position
        return position

    def _grid_center(self, i: int, j: int) -> tuple[float, float]:
        return (
            (i + 0.5) * self.grid_step[0] + self.grid_size[0, 0],
            (j + 0.5) * self.grid_step[1] + self.grid_size[1, 0],
        )

    def _grid_to_world_frame(self, local_position: np.ndarray) -> np.ndarray:
        if self.align_to_vehicle_axes:
            c, s = np.cos(-self.observer_vehicle.heading), np.sin(
                -self.observer_vehicle.heading
            )
            local_position = np.array([[c, s], [-s, c]]) @ local_position
        return local_position + self.observer_vehicle.position

    def _coords_to_index(self, x: float, y: float) -> tuple[int, int]:
        return (
            int(np.floor((x - self.grid_size[0, 0]) / self.grid_step[0])),
            int(np.floor((y - self.grid_size[1, 0]) / self.grid_step[1])),
        )

    def _vehicle_cell_coverages(
        self,
        center_x: float,
        center_y: float,
        heading: float,
        length: float,
        width: float,
    ) -> dict[tuple[int, int], float]:
        if not self.vehicle_footprint:
            i, j = self._coords_to_index(center_x, center_y)
            if 0 <= i < self.grid.shape[-2] and 0 <= j < self.grid.shape[-1]:
                return {(i, j): 1.0}
            return {}

        cos_h, sin_h = np.cos(heading), np.sin(heading)
        half_l, half_w = 0.5 * length, 0.5 * width
        extent_x = abs(cos_h) * half_l + abs(sin_h) * half_w
        extent_y = abs(sin_h) * half_l + abs(cos_h) * half_w

        i_min, j_min = self._coords_to_index(center_x - extent_x, center_y - extent_y)
        i_max, j_max = self._coords_to_index(center_x + extent_x, center_y + extent_y)
        i_min = max(0, i_min)
        j_min = max(0, j_min)
        i_max = min(self.grid.shape[-2] - 1, i_max)
        j_max = min(self.grid.shape[-1] - 1, j_max)

        cell_coverages = {}
        for i in range(i_min, i_max + 1):
            for j in range(j_min, j_max + 1):
                coverage = self._cell_overlap_ratio(
                    center_x=center_x,
                    center_y=center_y,
                    cos_h=cos_h,
                    sin_h=sin_h,
                    half_l=half_l,
                    half_w=half_w,
                    i=i,
                    j=j,
                )
                if coverage > 0.0:
                    cell_coverages[(i, j)] = coverage
        return cell_coverages

    def _cell_overlap_ratio(
        self,
        center_x: float,
        center_y: float,
        cos_h: float,
        sin_h: float,
        half_l: float,
        half_w: float,
        i: int,
        j: int,
    ) -> float:
        """
        Approximate vehicle-cell overlap using stratified supersampling.

        A fully covered cell gets value 1, partial coverage gets value in (0, 1).
        """
        x0 = i * self.grid_step[0] + self.grid_size[0, 0]
        y0 = j * self.grid_step[1] + self.grid_size[1, 0]
        n = self.presence_subsamples
        sample_x = x0 + (np.arange(n, dtype=np.float32) + 0.5) / n * self.grid_step[0]
        sample_y = y0 + (np.arange(n, dtype=np.float32) + 0.5) / n * self.grid_step[1]
        xx, yy = np.meshgrid(sample_x, sample_y, indexing="ij")

        dx = xx - center_x
        dy = yy - center_y
        local_long = cos_h * dx + sin_h * dy
        local_lat = -sin_h * dx + cos_h * dy
        inside = (np.abs(local_long) <= half_l) & (np.abs(local_lat) <= half_w)
        return float(np.mean(inside))

    def _feature_values(
        self, vehicle: Vehicle, rel: dict, heading: float, is_ego: bool
    ) -> dict[str, float]:
        rel_pos = np.array([rel["x"], rel["y"]], dtype=np.float32)
        rel_vel = np.array([rel["vx"], rel["vy"]], dtype=np.float32)
        rel_speed = float(np.linalg.norm(rel_vel))
        distance = float(np.linalg.norm(rel_pos))
        ttc_risk = 0.0 if is_ego else self._ttc_risk(rel_pos, rel_vel)

        return {
            "presence": 1.0,
            "x": float(np.clip(rel["x"] / self.distance_normalization, -1, 1)),
            "y": float(np.clip(rel["y"] / self.distance_normalization, -1, 1)),
            "vx": float(np.clip(rel["vx"] / (2 * Vehicle.MAX_SPEED), -1, 1)),
            "vy": float(np.clip(rel["vy"] / (2 * Vehicle.MAX_SPEED), -1, 1)),
            "speed": float(np.clip(rel_speed / (2 * Vehicle.MAX_SPEED), 0, 1)),
            "heading": float(np.clip(heading / np.pi, -1, 1)),
            "cos_h": float(np.cos(heading)),
            "sin_h": float(np.sin(heading)),
            "long_off": float(np.clip(rel.get("long_off", 0.0) / self.distance_normalization, -1, 1)),
            "lat_off": float(np.clip(rel.get("lat_off", 0.0) / self.distance_normalization, -1, 1)),
            "distance": float(np.clip(distance / self.distance_normalization, 0, 1)),
            "ttc": ttc_risk,
        }

    def _ttc_risk(self, rel_pos: np.ndarray, rel_vel: np.ndarray) -> float:
        distance = float(np.linalg.norm(rel_pos))
        if distance < 1e-6:
            return 1.0
        closing_speed = -float(np.dot(rel_pos, rel_vel)) / distance
        if closing_speed <= 1e-3:
            return 0.0
        ttc = distance / closing_speed
        return float(np.clip(1.0 - ttc / self.ttc_horizon, 0.0, 1.0))


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: AbstractEnv, scales: list[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                dict(
                    desired_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["desired_goal"].shape,
                        dtype=np.float64,
                    ),
                    achieved_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["achieved_goal"].shape,
                        dtype=np.float64,
                    ),
                    observation=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["observation"].shape,
                        dtype=np.float64,
                    ),
                )
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return OrderedDict(
                [
                    ("observation", np.zeros((len(self.features),))),
                    ("achieved_goal", np.zeros((len(self.features),))),
                    ("desired_goal", np.zeros((len(self.features),))),
                ]
            )

        obs = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        )
        goal = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.goal.to_dict()])[
                self.features
            ]
        )
        obs = OrderedDict(
            [
                ("observation", obs / self.scales),
                ("achieved_goal", obs / self.scales),
                ("desired_goal", goal / self.scales),
            ]
        )
        return obs


class AttributesObservation(ObservationType):
    def __init__(self, env: AbstractEnv, attributes: list[str], **kwargs: dict) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                {
                    attribute: spaces.Box(
                        -np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float64
                    )
                    for attribute in self.attributes
                }
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> dict[str, np.ndarray]:
        return OrderedDict(
            [(attribute, getattr(self.env, attribute)) for attribute in self.attributes]
        )


class MultiAgentObservation(ObservationType):
    def __init__(self, env: AbstractEnv, observation_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [obs_type.space() for obs_type in self.agents_observation_types]
        )

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


class TupleObservation(ObservationType):
    def __init__(
        self, env: AbstractEnv, observation_configs: list[dict], **kwargs
    ) -> None:
        super().__init__(env)
        self.observation_types = [
            observation_factory(self.env, obs_config)
            for obs_config in observation_configs
        ]

    def space(self) -> spaces.Space:
        return spaces.Tuple([obs_type.space() for obs_type in self.observation_types])

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.observation_types)


class ExitObservation(KinematicObservation):
    """Specific to exit_env, observe the distance to the next exit lane as part of a KinematicObservation."""

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        # Add ego-vehicle
        ego_dict = self.observer_vehicle.to_dict()
        exit_lane = self.env.road.network.get_lane(("1", "2", -1))
        ego_dict["x"] = exit_lane.local_coordinates(self.observer_vehicle.position)[0]
        df = pd.DataFrame.from_records([ego_dict])[self.features]

        # Add nearby traffic
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )
        if close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None
            df = pd.concat(
                [
                    df,
                    pd.DataFrame.from_records(
                        [
                            v.to_dict(
                                origin, observe_intentions=self.observe_intentions
                            )
                            for v in close_vehicles[-self.vehicles_count + 1 :]
                        ]
                    )[self.features],
                ],
                ignore_index=True,
            )
        # Normalize and clip
        if self.normalize:
            df = self.normalize_obs(df)
        # Fill missing rows
        if df.shape[0] < self.vehicles_count:
            rows = np.zeros((self.vehicles_count - df.shape[0], len(self.features)))
            df = pd.concat(
                [df, pd.DataFrame(data=rows, columns=self.features)], ignore_index=True
            )
        # Reorder
        df = df[self.features]
        obs = df.values.copy()
        if self.order == "shuffled":
            self.env.np_random.shuffle(obs[1:])
        # Flatten
        return obs.astype(self.space().dtype)


class LidarObservation(ObservationType):
    DISTANCE = 0
    SPEED = 1

    def __init__(
        self,
        env,
        cells: int = 16,
        maximum_range: float = 60,
        normalize: bool = True,
        **kwargs,
    ):
        super().__init__(env, **kwargs)
        self.cells = cells
        self.maximum_range = maximum_range
        self.normalize = normalize
        self.angle = 2 * np.pi / self.cells
        self.grid = np.ones((self.cells, 1)) * float("inf")
        self.origin = None

    def space(self) -> spaces.Space:
        high = 1 if self.normalize else self.maximum_range
        return spaces.Box(shape=(self.cells, 2), low=-high, high=high, dtype=np.float32)

    def observe(self) -> np.ndarray:
        obs = self.trace(
            self.observer_vehicle.position, self.observer_vehicle.velocity
        ).copy()
        if self.normalize:
            obs /= self.maximum_range
        return obs

    def trace(self, origin: np.ndarray, origin_velocity: np.ndarray) -> np.ndarray:
        self.origin = origin.copy()
        self.grid = np.ones((self.cells, 2), dtype=np.float32) * self.maximum_range

        for obstacle in self.env.road.vehicles + self.env.road.objects:
            if obstacle is self.observer_vehicle or not obstacle.solid:
                continue
            center_distance = np.linalg.norm(obstacle.position - origin)
            if center_distance > self.maximum_range:
                continue
            center_angle = self.position_to_angle(obstacle.position, origin)
            center_index = self.angle_to_index(center_angle)
            distance = center_distance - obstacle.WIDTH / 2
            if distance <= self.grid[center_index, self.DISTANCE]:
                direction = self.index_to_direction(center_index)
                velocity = (obstacle.velocity - origin_velocity).dot(direction)
                self.grid[center_index, :] = [distance, velocity]

            # Angular sector covered by the obstacle
            corners = utils.rect_corners(
                obstacle.position, obstacle.LENGTH, obstacle.WIDTH, obstacle.heading
            )
            angles = [self.position_to_angle(corner, origin) for corner in corners]
            min_angle, max_angle = min(angles), max(angles)
            if (
                min_angle < -np.pi / 2 < np.pi / 2 < max_angle
            ):  # Object's corners are wrapping around +pi
                min_angle, max_angle = max_angle, min_angle + 2 * np.pi
            start, end = self.angle_to_index(min_angle), self.angle_to_index(max_angle)
            if start < end:
                indexes = np.arange(start, end + 1)
            else:  # Object's corners are wrapping around 0
                indexes = np.hstack(
                    [np.arange(start, self.cells), np.arange(0, end + 1)]
                )

            # Actual distance computation for these sections
            for index in indexes:
                direction = self.index_to_direction(index)
                ray = [origin, origin + self.maximum_range * direction]
                distance = utils.distance_to_rect(ray, corners)
                if distance <= self.grid[index, self.DISTANCE]:
                    velocity = (obstacle.velocity - origin_velocity).dot(direction)
                    self.grid[index, :] = [distance, velocity]
        return self.grid

    def position_to_angle(self, position: np.ndarray, origin: np.ndarray) -> float:
        return (
            np.arctan2(position[1] - origin[1], position[0] - origin[0])
            + self.angle / 2
        )

    def position_to_index(self, position: np.ndarray, origin: np.ndarray) -> int:
        return self.angle_to_index(self.position_to_angle(position, origin))

    def angle_to_index(self, angle: float) -> int:
        return int(np.floor(angle / self.angle)) % self.cells

    def index_to_direction(self, index: int) -> np.ndarray:
        return np.array([np.cos(index * self.angle), np.sin(index * self.angle)])


def observation_factory(env: AbstractEnv, config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "DetailedOccupancyGrid":
        return DetailedOccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, **config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    elif config["type"] == "TupleObservation":
        return TupleObservation(env, **config)
    elif config["type"] == "LidarObservation":
        return LidarObservation(env, **config)
    elif config["type"] == "ExitObservation":
        return ExitObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
