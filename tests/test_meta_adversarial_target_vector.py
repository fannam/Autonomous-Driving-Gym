from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.controller import MDPVehicle
from AlphaZeroMetaAdversarial.core.action_score_heuristics import (
    ActionScoreHeuristic,
    NPCClosingActionScoreHeuristic,
)
from AlphaZeroMetaAdversarial.core.game import classify_terminal_state
from AlphaZeroMetaAdversarial.core.mcts import SimultaneousMCTS, SimultaneousMCTSNode
from AlphaZeroMetaAdversarial.core.perspective_stack import (
    PerspectiveTensorBuilder,
    seed_history_from_env,
)
from AlphaZeroMetaAdversarial.core.settings import (
    PerspectiveTensorConfig,
    SELF_PLAY_CONFIG,
    ZeroSumConfig,
)
from AlphaZeroMetaAdversarial.environment.config import init_env as init_meta_env
from AlphaZeroMetaAdversarial.network.alphazero_network import AlphaZeroNetwork
from AlphaZeroMetaAdversarial.scripts.self_play_kaggle_dual_gpu import (
    _serialize_examples,
)
from AlphaZeroMetaAdversarial.training.trainer import AdversarialAlphaZeroTrainer
from autonomous_driving_shared.alphazero_adversarial.training.base import (
    EpisodeStepSample,
)


class DummyUnwrappedEnv:
    def __init__(self, controlled_vehicles) -> None:
        self.controlled_vehicles = controlled_vehicles
        self.time = 0.0
        self.steps = 0
        self.action_type = SimpleNamespace(
            agents_action_types=[
                DummyMetaActionType(controlled_vehicles[0]),
                DummyMetaActionType(controlled_vehicles[1]),
            ]
        )

    def _is_success(self) -> bool:
        return False

    def _is_terminated(self) -> bool:
        return False

    def _is_truncated(self) -> bool:
        return False


class DummyEnv:
    def __init__(self, controlled_vehicles) -> None:
        self.unwrapped = DummyUnwrappedEnv(controlled_vehicles)


class DummyMetaActionType:
    actions = {
        0: "LANE_LEFT",
        1: "IDLE",
        2: "LANE_RIGHT",
        3: "FASTER",
        4: "SLOWER",
    }
    actions_indexes = {label: action for action, label in actions.items()}

    def __init__(self, controlled_vehicle) -> None:
        self.controlled_vehicle = controlled_vehicle

    def get_available_actions(self) -> list[int]:
        return list(self.actions.keys())


def _make_tensor_config(*, history_length: int = 1) -> PerspectiveTensorConfig:
    return PerspectiveTensorConfig(
        grid_shape=(4, 4),
        grid_extent=((-10.0, 10.0), (-10.0, 10.0)),
        grid_step=(5.0, 5.0),
        history_length=history_length,
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


def _make_mcts_node(env: DummyEnv) -> SimultaneousMCTSNode:
    return SimultaneousMCTSNode(
        env=env,
        parent=None,
        parent_joint_action=None,
        tensor_config=_make_tensor_config(),
        zero_sum_config=ZeroSumConfig(minimum_safe_speed=5.0),
        history=None,
        copy_env=False,
    )


def _make_mcts(
    node: SimultaneousMCTSNode,
    *,
    config: PerspectiveTensorConfig,
    npc_closing_ucb_bonus: float,
    action_score_heuristic: ActionScoreHeuristic | None = None,
) -> SimultaneousMCTS:
    return SimultaneousMCTS(
        root=node,
        network=torch.nn.Linear(1, 1),
        tensor_builder=PerspectiveTensorBuilder(config),
        device="cpu",
        n_simulations=1,
        npc_closing_ucb_bonus=npc_closing_ucb_bonus,
        action_score_heuristic=action_score_heuristic,
    )


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


def test_npc_ucb_heuristic_prefers_faster_when_ego_is_ahead() -> None:
    config = _make_tensor_config()
    env = _make_dummy_env()
    ego_vehicle, npc_vehicle = env.unwrapped.controlled_vehicles
    npc_vehicle.position = np.asarray([20.0, 0.0], dtype=np.float32)
    npc_vehicle.target_lane_index = ("0", "1", 0)
    npc_vehicle.lane_index = ("0", "1", 0)
    ego_vehicle.position = np.asarray([60.0, 0.0], dtype=np.float32)
    ego_vehicle.target_lane_index = ("0", "1", 0)
    ego_vehicle.lane_index = ("0", "1", 0)

    node = _make_mcts_node(env)
    heuristic = NPCClosingActionScoreHeuristic(bonus_scale=1.0)
    _, npc_bonus_by_action = heuristic.action_score_bonuses(
        node=node,
        tensor_builder=PerspectiveTensorBuilder(config),
    )

    _, npc_action = node.select_joint_action(
        c_puct=0.0,
        npc_action_bonus_by_action=npc_bonus_by_action,
    )

    assert npc_action == 3


def test_npc_ucb_heuristic_prefers_lane_change_toward_ego() -> None:
    config = _make_tensor_config()
    env = _make_dummy_env()
    ego_vehicle, npc_vehicle = env.unwrapped.controlled_vehicles
    npc_vehicle.position = np.asarray([20.0, 4.0], dtype=np.float32)
    npc_vehicle.target_lane_index = ("0", "1", 1)
    npc_vehicle.lane_index = ("0", "1", 1)
    ego_vehicle.position = np.asarray([40.0, 0.0], dtype=np.float32)
    ego_vehicle.target_lane_index = ("0", "1", 0)
    ego_vehicle.lane_index = ("0", "1", 0)

    node = _make_mcts_node(env)
    heuristic = NPCClosingActionScoreHeuristic(bonus_scale=1.0)
    _, npc_bonus_by_action = heuristic.action_score_bonuses(
        node=node,
        tensor_builder=PerspectiveTensorBuilder(config),
    )

    _, npc_action = node.select_joint_action(
        c_puct=0.0,
        npc_action_bonus_by_action=npc_bonus_by_action,
    )

    assert npc_action == 0


def test_meta_mcts_accepts_custom_action_score_heuristic() -> None:
    class PreferSlowerHeuristic(ActionScoreHeuristic):
        def action_score_bonuses(
            self,
            *,
            node: SimultaneousMCTSNode,
            tensor_builder: PerspectiveTensorBuilder,
        ) -> tuple[dict[int, float], dict[int, float]]:
            del node
            del tensor_builder
            return {}, {4: 1.0}

    config = _make_tensor_config()
    env = _make_dummy_env()
    node = _make_mcts_node(env)
    mcts = _make_mcts(
        node,
        config=config,
        npc_closing_ucb_bonus=0.0,
        action_score_heuristic=PreferSlowerHeuristic(),
    )
    _, npc_bonus_by_action = mcts._action_score_bonuses(node)

    _, npc_action = node.select_joint_action(
        c_puct=0.0,
        npc_action_bonus_by_action=npc_bonus_by_action,
    )

    assert npc_action == 4


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


def test_inactive_npc_state_is_zero_padded_after_self_fault() -> None:
    config = _make_tensor_config(history_length=2)
    builder = PerspectiveTensorBuilder(config)
    env = _make_dummy_env()
    ego_vehicle, npc_vehicle = env.unwrapped.controlled_vehicles
    npc_vehicle.crashed = True

    outcome = classify_terminal_state(
        env,
        ZeroSumConfig(
            minimum_safe_speed=5.0,
            remove_npc_on_self_fault=True,
        ),
    )
    history = seed_history_from_env(env, config.history_length)
    empty_observation = np.zeros(
        (0, config.grid_shape[0], config.grid_shape[1]),
        dtype=np.float32,
    )

    ego_tensor = builder.build_agent_tensor(
        env=env,
        history=history,
        agent_index=0,
        observation=empty_observation,
        feature_names=(),
    )
    npc_tensor = builder.build_agent_tensor(
        env=env,
        history=history,
        agent_index=1,
        observation=None,
        feature_names=(),
    )

    assert outcome.terminal is False
    assert outcome.reason == "npc_self_collision"
    assert np.allclose(
        ego_tensor[config.history_length : 2 * config.history_length],
        0.0,
    )
    assert np.allclose(npc_tensor, 0.0)
    assert np.allclose(builder.build_target_vector(env, agent_index=1), 0.0)
    assert ego_vehicle in ego_vehicle.road.vehicles
    assert npc_vehicle not in ego_vehicle.road.vehicles


def test_trainer_builds_discounted_value_targets_per_trajectory() -> None:
    config = SimpleNamespace(
        tensor=_make_tensor_config(),
        learning_rate=1e-3,
        weight_decay=0.0,
        replay_buffer_size=16,
        discount_gamma=0.9,
    )
    trainer = AdversarialAlphaZeroTrainer(
        network=torch.nn.Linear(1, 1),
        config=config,
        env=None,
        device="cpu",
        verbose=False,
    )
    samples = [
        EpisodeStepSample(
            state=np.zeros((1, 1, 1), dtype=np.float32),
            target_vector=np.zeros((config.tensor.target_vector_dim,), dtype=np.float32),
            policy_targets=(np.asarray([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),),
            agent_index=0,
        )
        for _ in range(3)
    ]

    examples = trainer._make_replay_examples_for_trajectory(
        samples,
        final_value=-1.0,
    )

    assert len(examples) == 3
    assert [example[-1] for example in examples] == pytest.approx(
        [-0.81, -0.9, -1.0]
    )


def test_meta_self_play_summary_includes_joint_action_history() -> None:
    config = replace(
        SELF_PLAY_CONFIG,
        n_residual_layers=1,
        network_channels=16,
        target_hidden_dim=8,
        network_dropout_p=0.0,
        n_simulations=1,
        temperature=0.0,
        max_expand_actions_per_agent=2,
    )
    env = init_meta_env(
        seed=123,
        stage="self_play",
        env_config_overrides={
            "duration": 2,
            "vehicles_count": 4,
            "policy_frequency": 1,
            "simulation_frequency": 5,
        },
    )
    try:
        network = AlphaZeroNetwork(
            input_shape=config.input_shape,
            n_residual_layers=config.n_residual_layers,
            n_actions=config.n_actions,
            channels=config.network_channels,
            dropout_p=config.network_dropout_p,
            target_vector_dim=config.target_vector_dim,
            target_hidden_dim=config.target_hidden_dim,
        )
        trainer = AdversarialAlphaZeroTrainer(
            network=network,
            config=config,
            env=env,
            device="cpu",
            verbose=False,
            add_root_dirichlet_noise=False,
        )

        summary = trainer.run_episode(
            seed=123,
            env_seed=123,
            episode_index=0,
            max_steps=2,
            store_in_replay=False,
            add_root_dirichlet_noise=False,
            sample_actions=False,
        )
    finally:
        env.close()

    joint_actions = summary["joint_actions"]
    assert summary["steps"] >= 1
    assert isinstance(joint_actions, list)
    assert len(joint_actions) == summary["steps"]
    assert all(isinstance(joint_action, tuple) for joint_action in joint_actions)
    assert all(len(joint_action) == 2 for joint_action in joint_actions)
    assert all(
        isinstance(action, int)
        for joint_action in joint_actions
        for action in joint_action
    )
