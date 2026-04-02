from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from AlphaZeroAdversarial.core import runtime_config as adversarial_runtime_config
from AlphaZeroAdversarial.training.trainer import (
    AdversarialAlphaZeroTrainer as AdversarialTrainer,
)
from AlphaZeroMetaAdversarial.core import runtime_config as meta_runtime_config
from AlphaZeroMetaAdversarial.training.trainer import (
    AdversarialAlphaZeroTrainer as MetaAdversarialTrainer,
)


@pytest.mark.parametrize(
    ("runtime_module", "scenario_env_var", "declared_scenario"),
    (
        (
            adversarial_runtime_config,
            "ALPHAZERO_ADVERSARIAL_SCENARIO",
            "racetrack_adversarial",
        ),
        (
            meta_runtime_config,
            "ALPHAZERO_META_ADVERSARIAL_SCENARIO",
            "highway_meta_adversarial",
        ),
    ),
    ids=("adversarial", "meta_adversarial"),
)
def test_runtime_config_preserves_package_specific_env_validation(
    tmp_path,
    monkeypatch,
    runtime_module,
    scenario_env_var: str,
    declared_scenario: str,
) -> None:
    config_path = tmp_path / f"{declared_scenario}.yaml"
    config_path.write_text(
        json.dumps({"scenario_name": declared_scenario}),
        encoding="utf-8",
    )
    monkeypatch.setenv(scenario_env_var, "mismatched_scenario")

    with pytest.raises(ValueError) as exc_info:
        runtime_module.get_active_scenario_name(config_path=config_path)

    message = str(exc_info.value)
    assert scenario_env_var in message
    assert declared_scenario in message


def test_adversarial_trainer_build_policy_target_preserves_axis_factorization() -> None:
    trainer = object.__new__(AdversarialTrainer)
    trainer.config = SimpleNamespace(
        n_actions=9,
        n_action_axis_0=3,
        n_action_axis_1=3,
        tensor=SimpleNamespace(flip_npc_perspective=True),
        use_policy_target_smoothing=False,
        policy_target_smoothing_sigma=1.0,
    )

    accelerate_target, steering_target = trainer._build_policy_target(
        {
            0: 0.25,
            2: 0.25,
            6: 0.50,
        },
        agent_index=1,
    )

    assert accelerate_target.shape == (3,)
    assert steering_target.shape == (3,)
    assert np.allclose(accelerate_target, np.asarray([0.5, 0.0, 0.5], dtype=np.float32))
    assert np.allclose(steering_target, np.asarray([0.25, 0.0, 0.75], dtype=np.float32))


def test_meta_adversarial_trainer_build_policy_target_preserves_flat_policy() -> None:
    trainer = object.__new__(MetaAdversarialTrainer)
    trainer.config = SimpleNamespace(n_actions=5)

    (policy_target,) = trainer._build_policy_target(
        {
            1: 0.75,
            4: 0.25,
        },
        agent_index=0,
    )

    assert policy_target.shape == (5,)
    assert np.allclose(
        policy_target,
        np.asarray([0.0, 0.75, 0.0, 0.0, 0.25], dtype=np.float32),
    )
