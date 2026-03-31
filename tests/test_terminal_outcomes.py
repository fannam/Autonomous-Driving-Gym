from __future__ import annotations

import numpy as np
import pytest
import torch

from AlphaZeroAdversarial.core.game import classify_terminal_state as classify_adversarial
from AlphaZeroAdversarial.core.mcts import (
    SearchStats as AdversarialSearchStats,
    SimultaneousMCTS as AdversarialMCTS,
)
from AlphaZeroAdversarial.core.settings import ZeroSumConfig as AdversarialZeroSumConfig
from AlphaZeroMetaAdversarial.core.game import (
    classify_terminal_state as classify_meta_adversarial,
)
from AlphaZeroMetaAdversarial.core.mcts import (
    SearchStats as MetaAdversarialSearchStats,
    SimultaneousMCTS as MetaAdversarialMCTS,
)
from AlphaZeroMetaAdversarial.core.settings import (
    ZeroSumConfig as MetaAdversarialZeroSumConfig,
)


class DummyVehicle:
    def __init__(
        self,
        *,
        crashed: bool = False,
        on_road: bool = True,
        speed: float = 10.0,
    ) -> None:
        self.position = np.zeros(2, dtype=np.float32)
        self.heading = 0.0
        self.speed = float(speed)
        self.crashed = bool(crashed)
        self._on_road = bool(on_road)
        self.collision_partners: list[DummyVehicle] = []

    @property
    def on_road(self) -> bool:
        return self._on_road

    def _is_colliding(self, other, dt: float = 0.0):
        intersecting = any(partner is other for partner in self.collision_partners)
        return intersecting, intersecting, np.zeros(2, dtype=np.float32)


class DummyUnwrappedEnv:
    def __init__(
        self,
        ego_vehicle: DummyVehicle,
        npc_vehicle: DummyVehicle,
        *,
        success: bool = False,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        self.controlled_vehicles = (ego_vehicle, npc_vehicle)
        self._success = bool(success)
        self._terminated = bool(terminated)
        self._truncated = bool(truncated)

    def _is_success(self) -> bool:
        return self._success

    def _is_terminated(self) -> bool:
        return self._terminated

    def _is_truncated(self) -> bool:
        return self._truncated


class DummyEnv:
    def __init__(self, unwrapped: DummyUnwrappedEnv) -> None:
        self.unwrapped = unwrapped


class DummyPredictionNode:
    def __init__(self) -> None:
        self.cached_perspective_batch = np.zeros((2, 1, 1, 1), dtype=np.float32)
        self.cached_target_vector_batch = np.zeros((2, 0), dtype=np.float32)

    def ensure_perspective_batch(self, builder) -> None:
        del builder

    def ensure_model_inputs(self, builder) -> None:
        self.ensure_perspective_batch(builder)


class DummyFactorizedNetwork:
    training = False

    def __call__(self, batch, target_vector=None):
        del batch
        del target_vector
        accelerate = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)
        steering = torch.tensor([[0.8, 0.2], [0.1, 0.9]], dtype=torch.float32)
        value = torch.tensor([[0.25], [-0.75]], dtype=torch.float32)
        return accelerate, steering, value


class DummyFlatPolicyNetwork:
    training = False

    def __call__(self, batch, target_vector=None):
        del batch
        del target_vector
        policy = torch.tensor([[0.6, 0.4], [0.3, 0.7]], dtype=torch.float32)
        value = torch.tensor([[0.1], [-0.4]], dtype=torch.float32)
        return policy, value


@pytest.fixture(
    params=(
        (classify_adversarial, AdversarialZeroSumConfig),
        (classify_meta_adversarial, MetaAdversarialZeroSumConfig),
    ),
    ids=("adversarial", "meta_adversarial"),
)
def classifier_bundle(request):
    return request.param


def make_env(
    *,
    ego_vehicle: DummyVehicle,
    npc_vehicle: DummyVehicle,
    success: bool = False,
    terminated: bool = False,
    truncated: bool = False,
) -> DummyEnv:
    return DummyEnv(
        DummyUnwrappedEnv(
            ego_vehicle,
            npc_vehicle,
            success=success,
            terminated=terminated,
            truncated=truncated,
        )
    )


def test_direct_npc_hit_ego_rewards_npc(classifier_bundle) -> None:
    classify_terminal_state, config_cls = classifier_bundle
    ego_vehicle = DummyVehicle(crashed=True)
    npc_vehicle = DummyVehicle(crashed=True)
    ego_vehicle.collision_partners.append(npc_vehicle)
    npc_vehicle.collision_partners.append(ego_vehicle)
    env = make_env(ego_vehicle=ego_vehicle, npc_vehicle=npc_vehicle, terminated=True)

    outcome = classify_terminal_state(env, config_cls(minimum_safe_speed=5.0))

    assert outcome.terminal is True
    assert outcome.reason == "npc_hit_ego"
    assert outcome.ego_value == pytest.approx(-1.0)
    assert outcome.npc_value == pytest.approx(1.0)


def test_ego_self_collision_only_penalizes_ego(classifier_bundle) -> None:
    classify_terminal_state, config_cls = classifier_bundle
    ego_vehicle = DummyVehicle(crashed=True)
    npc_vehicle = DummyVehicle()
    env = make_env(ego_vehicle=ego_vehicle, npc_vehicle=npc_vehicle, terminated=True)

    outcome = classify_terminal_state(env, config_cls(minimum_safe_speed=5.0))

    assert outcome.reason == "ego_self_collision"
    assert outcome.ego_value == pytest.approx(-1.0)
    assert outcome.npc_value == pytest.approx(0.0)


def test_npc_self_collision_only_penalizes_npc(classifier_bundle) -> None:
    classify_terminal_state, config_cls = classifier_bundle
    ego_vehicle = DummyVehicle()
    npc_vehicle = DummyVehicle(crashed=True)
    env = make_env(ego_vehicle=ego_vehicle, npc_vehicle=npc_vehicle)

    outcome = classify_terminal_state(env, config_cls(minimum_safe_speed=5.0))

    assert outcome.reason == "npc_self_collision"
    assert outcome.ego_value == pytest.approx(0.0)
    assert outcome.npc_value == pytest.approx(-1.0)


def test_double_self_collision_penalizes_both_agents(classifier_bundle) -> None:
    classify_terminal_state, config_cls = classifier_bundle
    ego_vehicle = DummyVehicle(crashed=True)
    npc_vehicle = DummyVehicle(crashed=True)
    env = make_env(ego_vehicle=ego_vehicle, npc_vehicle=npc_vehicle, terminated=True)

    outcome = classify_terminal_state(env, config_cls(minimum_safe_speed=5.0))

    assert outcome.reason == "double_self_collision"
    assert outcome.ego_value == pytest.approx(-1.0)
    assert outcome.npc_value == pytest.approx(-1.0)


def test_npc_offroad_is_terminal_self_fault(classifier_bundle) -> None:
    classify_terminal_state, config_cls = classifier_bundle
    ego_vehicle = DummyVehicle()
    npc_vehicle = DummyVehicle(on_road=False)
    env = make_env(ego_vehicle=ego_vehicle, npc_vehicle=npc_vehicle, terminated=False)

    outcome = classify_terminal_state(env, config_cls(minimum_safe_speed=5.0))

    assert outcome.reason == "npc_offroad"
    assert outcome.ego_value == pytest.approx(0.0)
    assert outcome.npc_value == pytest.approx(-1.0)


def test_safe_timeout_rewards_ego(classifier_bundle) -> None:
    classify_terminal_state, config_cls = classifier_bundle
    ego_vehicle = DummyVehicle(speed=6.0)
    npc_vehicle = DummyVehicle()
    env = make_env(ego_vehicle=ego_vehicle, npc_vehicle=npc_vehicle, truncated=True)

    outcome = classify_terminal_state(env, config_cls(minimum_safe_speed=5.0))

    assert outcome.reason == "ego_timeout_safe"
    assert outcome.ego_value == pytest.approx(1.0)
    assert outcome.npc_value == pytest.approx(-1.0)


def test_adversarial_mcts_predict_keeps_agent_values_separate() -> None:
    mcts = AdversarialMCTS.__new__(AdversarialMCTS)
    mcts._builder = object()
    mcts._device = torch.device("cpu")
    mcts._network = DummyFactorizedNetwork()
    mcts._stats = AdversarialSearchStats()
    mcts.n_action_axis_0 = 2
    mcts.n_action_axis_1 = 2
    mcts.flip_npc_steering = False

    _, _, ego_value, npc_value = mcts._predict(DummyPredictionNode())

    assert ego_value == pytest.approx(0.25)
    assert npc_value == pytest.approx(-0.75)


def test_meta_adversarial_mcts_predict_keeps_agent_values_separate() -> None:
    mcts = MetaAdversarialMCTS.__new__(MetaAdversarialMCTS)
    mcts._builder = object()
    mcts._device = torch.device("cpu")
    mcts._network = DummyFlatPolicyNetwork()
    mcts._stats = MetaAdversarialSearchStats()

    _, _, ego_value, npc_value = mcts._predict(DummyPredictionNode())

    assert ego_value == pytest.approx(0.1)
    assert npc_value == pytest.approx(-0.4)
