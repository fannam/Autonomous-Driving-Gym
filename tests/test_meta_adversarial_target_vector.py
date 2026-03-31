from __future__ import annotations

import numpy as np
import pytest
import torch

from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from AlphaZeroMetaAdversarial.core.perspective_stack import PerspectiveTensorBuilder
from AlphaZeroMetaAdversarial.core.settings import PerspectiveTensorConfig
from AlphaZeroMetaAdversarial.network.alphazero_network import AlphaZeroNetwork
from AlphaZeroMetaAdversarial.scripts.self_play_kaggle_dual_gpu import (
    _serialize_examples,
)


class DummyUnwrappedEnv:
    def __init__(self, controlled_vehicles) -> None:
        self.controlled_vehicles = controlled_vehicles


class DummyEnv:
    def __init__(self, controlled_vehicles) -> None:
        self.unwrapped = DummyUnwrappedEnv(controlled_vehicles)


def _make_tensor_config() -> PerspectiveTensorConfig:
    return PerspectiveTensorConfig(
        grid_shape=(4, 4),
        grid_extent=((-10.0, 10.0), (-10.0, 10.0)),
        grid_step=(5.0, 5.0),
        history_length=1,
        static_feature_names=(),
        include_self_speed_plane=False,
        include_heading_planes=False,
        include_progress_plane=False,
        flip_npc_perspective=True,
        use_target_vector=True,
        target_position_scale=50.0,
        target_velocity_scale=40.0,
        route_lookahead_base=20.0,
        route_lookahead_speed_gain=0.5,
        route_lookahead_min=10.0,
        route_lookahead_max=60.0,
        npc_intercept_speed_floor=5.0,
        npc_intercept_tau_max=3.0,
        include_role_bit=True,
        include_target_type_bit=True,
    )


def _make_dummy_env() -> DummyEnv:
    road = Road(RoadNetwork.straight_road_network(lanes=2, length=500.0))
    target_speeds = np.asarray([20.0, 25.0, 30.0], dtype=np.float32)
    ego = MDPVehicle(
        road,
        position=[50.0, 0.0],
        heading=0.0,
        speed=25.0,
        target_lane_index=("0", "1", 0),
        target_speed=25.0,
        target_speeds=target_speeds,
    )
    npc = MDPVehicle(
        road,
        position=[20.0, 4.0],
        heading=0.0,
        speed=20.0,
        target_lane_index=("0", "1", 1),
        target_speed=20.0,
        target_speeds=target_speeds,
    )
    road.vehicles.extend([ego, npc])
    return DummyEnv((ego, npc))


def test_target_vector_batch_has_expected_semantics() -> None:
    config = _make_tensor_config()
    builder = PerspectiveTensorBuilder(config)
    env = _make_dummy_env()

    vectors = builder.build_target_vector_batch(env)

    assert vectors.shape == (2, config.target_vector_dim)
    assert np.all(np.isfinite(vectors))
    assert np.all(np.abs(vectors) <= 1.0 + 1e-6)

    ego_vector = vectors[0]
    npc_vector = vectors[1]
    assert ego_vector[0] > 0.0
    assert npc_vector[0] > 0.0
    assert npc_vector[1] > 0.0
    assert ego_vector[6] == pytest.approx(1.0)
    assert ego_vector[7] == pytest.approx(1.0)
    assert npc_vector[6] == pytest.approx(-1.0)
    assert npc_vector[7] == pytest.approx(-1.0)


def test_network_accepts_target_vector_late_fusion() -> None:
    config = _make_tensor_config()
    network = AlphaZeroNetwork(
        input_shape=config.network_input_shape,
        n_residual_layers=1,
        n_actions=5,
        channels=16,
        dropout_p=0.0,
        target_vector_dim=config.target_vector_dim,
        target_hidden_dim=12,
    )
    state_batch = torch.randn(
        2,
        config.plane_count,
        config.grid_shape[0],
        config.grid_shape[1],
    )
    target_vector_batch = torch.randn(2, config.target_vector_dim)

    policy_logits, value = network(state_batch, target_vector_batch, return_logits=True)
    policy, value_again = network(state_batch, target_vector_batch)

    assert policy_logits.shape == (2, 5)
    assert value.shape == (2, 1)
    assert policy.shape == (2, 5)
    assert value_again.shape == (2, 1)
    assert torch.allclose(policy.sum(dim=1), torch.ones(2), atol=1e-5)


def test_kaggle_serialization_preserves_target_vectors() -> None:
    config = _make_tensor_config()
    state = np.zeros(
        (
            config.plane_count,
            config.grid_shape[0],
            config.grid_shape[1],
        ),
        dtype=np.float32,
    )
    target_vector = np.linspace(
        -0.75,
        0.75,
        num=config.target_vector_dim,
        dtype=np.float32,
    )
    policy = np.asarray([0.2, 0.3, 0.1, 0.25, 0.15], dtype=np.float32)
    value = -1.0

    state_tensor, target_vector_tensor, policy_tensor, value_tensor = _serialize_examples(
        [(state, target_vector, policy, value)]
    )

    assert state_tensor.shape == (1,) + state.shape
    assert target_vector_tensor.shape == (1, config.target_vector_dim)
    assert policy_tensor.shape == (1, policy.shape[0])
    assert value_tensor.shape == (1, 1)
    assert torch.allclose(target_vector_tensor[0], torch.from_numpy(target_vector))
