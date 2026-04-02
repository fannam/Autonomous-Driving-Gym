from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .game import get_agent_snapshots, get_progress_value, normalize_speed


@dataclass(frozen=True)
class AgentPose:
    position: tuple[float, float]
    heading: float


@dataclass(frozen=True)
class PerspectiveHistory:
    ego: tuple[AgentPose, ...]
    npc: tuple[AgentPose, ...]

    def swapped(self) -> "PerspectiveHistory":
        return PerspectiveHistory(ego=self.npc, npc=self.ego)


def _snapshot_to_pose(snapshot) -> AgentPose:
    return AgentPose(position=snapshot.position, heading=snapshot.heading)


def seed_history_from_env(env, history_length: int) -> PerspectiveHistory:
    ego_snapshot, npc_snapshot = get_agent_snapshots(env)
    ego_pose = _snapshot_to_pose(ego_snapshot)
    npc_pose = _snapshot_to_pose(npc_snapshot)
    return PerspectiveHistory(
        ego=tuple(ego_pose for _ in range(history_length)),
        npc=tuple(npc_pose for _ in range(history_length)),
    )


def advance_history(
    history: PerspectiveHistory,
    env,
    history_length: int,
) -> PerspectiveHistory:
    ego_snapshot, npc_snapshot = get_agent_snapshots(env)
    ego_pose = _snapshot_to_pose(ego_snapshot)
    npc_pose = _snapshot_to_pose(npc_snapshot)
    if history_length <= 1:
        return PerspectiveHistory(ego=(ego_pose,), npc=(npc_pose,))

    return PerspectiveHistory(
        ego=tuple((*history.ego[-(history_length - 1) :], ego_pose)),
        npc=tuple((*history.npc[-(history_length - 1) :], npc_pose)),
    )


class PerspectiveTensorBuilder:
    """Build `2N + k` tensors for both viewpoints in one pass."""

    def __init__(self, config: Any):
        self.config = config

    def build_batch(
        self,
        env,
        history: PerspectiveHistory,
    ) -> np.ndarray:
        observation_type = env.unwrapped.observation_type
        agents_observation_types = getattr(observation_type, "agents_observation_types", None)
        if agents_observation_types is None:
            raise RuntimeError(
                "Expected a MultiAgentObservation-based environment for adversarial tensor building."
            )
        observations = observation_type.observe()
        feature_names_by_agent = tuple(
            tuple(str(feature_name) for feature_name in getattr(obs_type, "features", ()))
            for obs_type in agents_observation_types
        )
        ego_tensor = self.build_agent_tensor(
            env=env,
            history=history,
            agent_index=0,
            observation=observations[0],
            feature_names=feature_names_by_agent[0],
        )
        npc_tensor = self.build_agent_tensor(
            env=env,
            history=history,
            agent_index=1,
            observation=observations[1],
            feature_names=feature_names_by_agent[1],
        )
        return np.stack((ego_tensor, npc_tensor), axis=0)

    def build_target_vector_batch(self, env) -> np.ndarray:
        if not self.config.use_target_vector:
            return np.zeros((2, 0), dtype=np.float32)
        return np.stack(
            (
                self.build_target_vector(env=env, agent_index=0),
                self.build_target_vector(env=env, agent_index=1),
            ),
            axis=0,
        )

    def build_agent_tensor(
        self,
        env,
        history: PerspectiveHistory,
        agent_index: int,
        observation: np.ndarray,
        feature_names: tuple[str, ...],
    ) -> np.ndarray:
        width, height = self.config.grid_shape
        tensor = np.zeros((self.config.plane_count, width, height), dtype=np.float32)
        mirror_y = bool(agent_index == 1 and self.config.flip_npc_perspective)

        ego_snapshot, npc_snapshot = get_agent_snapshots(env)
        if agent_index == 0:
            self_history = history.ego
            opponent_history = history.npc
            self_snapshot = ego_snapshot
        else:
            self_history = history.npc
            opponent_history = history.ego
            self_snapshot = npc_snapshot

        channel_offset = 0
        self._fill_history_block(
            tensor=tensor,
            channel_offset=channel_offset,
            self_snapshot=self_snapshot,
            poses=self_history,
            mirror_y=mirror_y,
        )
        channel_offset += self.config.history_length

        self._fill_history_block(
            tensor=tensor,
            channel_offset=channel_offset,
            self_snapshot=self_snapshot,
            poses=opponent_history,
            mirror_y=mirror_y,
        )
        channel_offset += self.config.history_length

        channel_offset = self._fill_static_features(
            tensor=tensor,
            channel_offset=channel_offset,
            observation=observation,
            feature_names=feature_names,
            mirror_y=mirror_y,
        )
        channel_offset = self._fill_scalar_planes(
            tensor=tensor,
            channel_offset=channel_offset,
            env=env,
            agent_index=agent_index,
            mirror_y=mirror_y,
        )

        if channel_offset != self.config.plane_count:
            raise RuntimeError(
                f"Expected {self.config.plane_count} channels, built {channel_offset}."
            )
        return tensor

    def build_target_vector(
        self,
        env,
        agent_index: int,
    ) -> np.ndarray:
        ego_vehicle, npc_vehicle = env.unwrapped.controlled_vehicles[:2]
        self_vehicle = ego_vehicle if agent_index == 0 else npc_vehicle
        mirror_y = bool(agent_index == 1 and self.config.flip_npc_perspective)

        if agent_index == 0:
            target_position, target_velocity = self._resolve_route_target(self_vehicle)
            role_bit = 1.0
            target_type_bit = 1.0
        else:
            target_position, target_velocity = self._resolve_intercept_target(
                self_vehicle=self_vehicle,
                target_vehicle=ego_vehicle,
            )
            role_bit = -1.0
            target_type_bit = -1.0

        local_dx, local_dy = self._world_delta_to_local(
            delta=target_position - np.asarray(self_vehicle.position, dtype=np.float32),
            observer_heading=float(getattr(self_vehicle, "heading", 0.0)),
            mirror_y=mirror_y,
        )
        local_dvx, local_dvy = self._world_delta_to_local(
            delta=target_velocity - self._velocity_vector(self_vehicle),
            observer_heading=float(getattr(self_vehicle, "heading", 0.0)),
            mirror_y=mirror_y,
        )
        bearing = float(np.arctan2(local_dy, local_dx)) if (local_dx or local_dy) else 0.0

        vector_values = [
            self._normalize_signed(local_dx, self.config.target_position_scale),
            self._normalize_signed(local_dy, self.config.target_position_scale),
            self._normalize_signed(local_dvx, self.config.target_velocity_scale),
            self._normalize_signed(local_dvy, self.config.target_velocity_scale),
            float(np.sin(bearing)),
            float(np.cos(bearing)),
        ]
        if self.config.include_role_bit:
            vector_values.append(role_bit)
        if self.config.include_target_type_bit:
            vector_values.append(target_type_bit)
        return np.asarray(vector_values, dtype=np.float32)

    def _fill_history_block(
        self,
        tensor: np.ndarray,
        channel_offset: int,
        self_snapshot,
        poses: tuple[AgentPose, ...],
        mirror_y: bool,
    ) -> None:
        for plane_offset, pose in enumerate(poses):
            plane = tensor[channel_offset + plane_offset]
            indices = self._world_to_grid_indices(
                target_position=np.asarray(pose.position, dtype=np.float32),
                observer_position=np.asarray(self_snapshot.position, dtype=np.float32),
                observer_heading=float(self_snapshot.heading),
                mirror_y=mirror_y,
            )
            if indices is None:
                continue
            plane[indices[0], indices[1]] = 1.0

    def _fill_static_features(
        self,
        tensor: np.ndarray,
        channel_offset: int,
        observation: np.ndarray,
        feature_names: tuple[str, ...],
        mirror_y: bool,
    ) -> int:
        for feature_name in self.config.static_feature_names:
            if feature_name in feature_names:
                source_index = feature_names.index(feature_name)
                feature_plane = observation[source_index]
                if mirror_y:
                    feature_plane = np.flip(feature_plane, axis=1)
                tensor[channel_offset] = feature_plane
            channel_offset += 1
        return channel_offset

    def _fill_scalar_planes(
        self,
        tensor: np.ndarray,
        channel_offset: int,
        env,
        agent_index: int,
        mirror_y: bool,
    ) -> int:
        ego_vehicle, npc_vehicle = env.unwrapped.controlled_vehicles[:2]
        self_vehicle = ego_vehicle if agent_index == 0 else npc_vehicle

        if self.config.include_self_speed_plane:
            tensor[channel_offset].fill(normalize_speed(self_vehicle))
            channel_offset += 1

        if self.config.include_heading_planes:
            heading = float(getattr(self_vehicle, "heading", 0.0))
            if mirror_y:
                heading = -heading
            tensor[channel_offset].fill(float(np.cos(heading)))
            tensor[channel_offset + 1].fill(float(np.sin(heading)))
            channel_offset += 2

        if self.config.include_progress_plane:
            tensor[channel_offset].fill(get_progress_value(env))
            channel_offset += 1

        return channel_offset

    @staticmethod
    def _normalize_signed(value: float, scale: float) -> float:
        return float(np.tanh(float(value) / max(float(scale), 1e-6)))

    @staticmethod
    def _velocity_vector(vehicle) -> np.ndarray:
        return np.asarray(getattr(vehicle, "velocity", np.zeros(2)), dtype=np.float32)

    def _resolve_route_target(self, vehicle) -> tuple[np.ndarray, np.ndarray]:
        lookahead_distance = float(
            np.clip(
                self.config.route_lookahead_base
                + self.config.route_lookahead_speed_gain * abs(float(getattr(vehicle, "speed", 0.0))),
                self.config.route_lookahead_min,
                self.config.route_lookahead_max,
            )
        )

        lane_index = getattr(vehicle, "target_lane_index", None) or getattr(vehicle, "lane_index", None)
        road = getattr(vehicle, "road", None)
        if road is None or lane_index is None:
            heading = float(getattr(vehicle, "heading", 0.0))
            target_position = np.asarray(vehicle.position, dtype=np.float32) + lookahead_distance * np.asarray(
                [np.cos(heading), np.sin(heading)],
                dtype=np.float32,
            )
            target_speed = float(getattr(vehicle, "target_speed", getattr(vehicle, "speed", 0.0)))
            target_velocity = target_speed * np.asarray([np.cos(heading), np.sin(heading)], dtype=np.float32)
            return target_position, target_velocity

        lane = road.network.get_lane(lane_index)
        longitudinal, _ = lane.local_coordinates(np.asarray(vehicle.position, dtype=np.float32))
        target_longitudinal = float(longitudinal) + lookahead_distance
        route = list(getattr(vehicle, "route", None) or [lane_index])
        if route:
            target_position, target_heading = road.network.position_heading_along_route(
                route,
                target_longitudinal,
                0.0,
                current_lane_index=lane_index,
            )
        else:
            target_position = lane.position(target_longitudinal, 0.0)
            target_heading = lane.heading_at(target_longitudinal)

        target_speed = float(getattr(vehicle, "target_speed", getattr(vehicle, "speed", 0.0)))
        target_velocity = target_speed * np.asarray(
            [np.cos(target_heading), np.sin(target_heading)],
            dtype=np.float32,
        )
        return np.asarray(target_position, dtype=np.float32), target_velocity

    def _resolve_intercept_target(
        self,
        *,
        self_vehicle,
        target_vehicle,
    ) -> tuple[np.ndarray, np.ndarray]:
        self_position = np.asarray(self_vehicle.position, dtype=np.float32)
        target_position = np.asarray(target_vehicle.position, dtype=np.float32)
        target_velocity = self._velocity_vector(target_vehicle)
        relative_position = target_position - self_position
        relative_velocity = target_velocity - self._velocity_vector(self_vehicle)

        distance = float(np.linalg.norm(relative_position))
        relative_speed = max(
            float(self.config.npc_intercept_speed_floor),
            float(np.linalg.norm(relative_velocity)),
        )
        tau = float(
            np.clip(
                distance / max(relative_speed, 1e-6),
                0.0,
                float(self.config.npc_intercept_tau_max),
            )
        )
        intercept_position = target_position + tau * target_velocity
        return intercept_position.astype(np.float32), target_velocity

    @staticmethod
    def _world_delta_to_local(
        delta: np.ndarray,
        observer_heading: float,
        mirror_y: bool,
    ) -> tuple[float, float]:
        cos_h = float(np.cos(observer_heading))
        sin_h = float(np.sin(observer_heading))
        local_x = cos_h * float(delta[0]) + sin_h * float(delta[1])
        local_y = -sin_h * float(delta[0]) + cos_h * float(delta[1])
        if mirror_y:
            local_y = -local_y
        return local_x, local_y

    def _world_to_grid_indices(
        self,
        target_position: np.ndarray,
        observer_position: np.ndarray,
        observer_heading: float,
        mirror_y: bool,
    ) -> tuple[int, int] | None:
        local_x, local_y = self._world_delta_to_local(
            delta=target_position - observer_position,
            observer_heading=observer_heading,
            mirror_y=mirror_y,
        )

        (x_min, x_max), (y_min, y_max) = self.config.grid_extent
        if not (x_min <= local_x < x_max and y_min <= local_y < y_max):
            return None

        ix = int(np.floor((local_x - x_min) / self.config.grid_step[0]))
        iy = int(np.floor((local_y - y_min) / self.config.grid_step[1]))
        width, height = self.config.grid_shape
        if not (0 <= ix < width and 0 <= iy < height):
            return None
        return ix, iy
