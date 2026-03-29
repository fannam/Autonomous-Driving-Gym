from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .settings import ZeroSumConfig


@dataclass(frozen=True)
class AgentSnapshot:
    position: tuple[float, float]
    heading: float
    speed: float
    on_road: bool
    crashed: bool


@dataclass(frozen=True)
class TerminalOutcome:
    terminal: bool
    ego_value: float
    npc_value: float
    reason: str


def get_controlled_vehicles(env) -> tuple[Any, Any]:
    vehicles = tuple(getattr(env.unwrapped, "controlled_vehicles", ()))
    if len(vehicles) < 2:
        raise RuntimeError(
            "The adversarial implementation expects at least two controlled vehicles."
        )
    return vehicles[0], vehicles[1]


def get_agent_snapshots(env) -> tuple[AgentSnapshot, AgentSnapshot]:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    return _vehicle_to_snapshot(ego_vehicle), _vehicle_to_snapshot(npc_vehicle)


def _vehicle_to_snapshot(vehicle) -> AgentSnapshot:
    return AgentSnapshot(
        position=(float(vehicle.position[0]), float(vehicle.position[1])),
        heading=float(getattr(vehicle, "heading", 0.0)),
        speed=float(getattr(vehicle, "speed", 0.0)),
        on_road=bool(getattr(vehicle, "on_road", True)),
        crashed=bool(getattr(vehicle, "crashed", False)),
    )


def _fallback_discrete_actions(action_space) -> tuple[int, ...]:
    if hasattr(action_space, "n"):
        return tuple(range(int(action_space.n)))
    raise ValueError("Expected a discrete action space for the adversarial agents.")


def _get_agent_action_type(env, agent_index: int):
    action_type = getattr(env.unwrapped, "action_type", None)
    agents_action_types = getattr(action_type, "agents_action_types", None)
    if agents_action_types is None or len(agents_action_types) <= agent_index:
        raise RuntimeError("The environment is not using a compatible multi-agent action type.")
    return agents_action_types[agent_index]


def neutral_action_index(agent_action_type) -> int:
    actions_indexes = getattr(agent_action_type, "actions_indexes", None)
    if isinstance(actions_indexes, dict) and "IDLE" in actions_indexes:
        return int(actions_indexes["IDLE"])

    actions_per_axis = int(getattr(agent_action_type, "actions_per_axis", 0))
    size = int(getattr(agent_action_type, "size", 0))
    if actions_per_axis > 0 and size > 0:
        center = actions_per_axis // 2
        action_index = 0
        for _ in range(size):
            action_index = action_index * actions_per_axis + center
        return int(action_index)

    space = agent_action_type.space()
    discrete_actions = _fallback_discrete_actions(space)
    return discrete_actions[len(discrete_actions) // 2]


def resolve_scripted_action(env, agent_index: int, mode: str | None) -> int | None:
    if mode is None:
        return None

    normalized_mode = str(mode).strip().lower()
    action_type = _get_agent_action_type(env, agent_index)

    if normalized_mode in {"idle", "neutral"}:
        return neutral_action_index(action_type)

    raise ValueError(f"Unsupported scripted policy mode: {mode!r}")


def get_available_actions(
    env,
    policy_modes: tuple[str | None, str | None] = (None, None),
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    available_actions: list[tuple[int, ...]] = []
    action_space = getattr(env, "action_space", None)
    spaces = getattr(action_space, "spaces", None)

    for agent_index, mode in enumerate(policy_modes):
        scripted_action = resolve_scripted_action(env, agent_index, mode)
        if scripted_action is not None:
            available_actions.append((int(scripted_action),))
            continue

        agent_action_type = _get_agent_action_type(env, agent_index)
        try:
            actions = tuple(
                int(action) for action in agent_action_type.get_available_actions()
            )
        except (AttributeError, NotImplementedError, TypeError):
            if spaces is None or len(spaces) <= agent_index:
                raise
            actions = _fallback_discrete_actions(spaces[agent_index])
        available_actions.append(actions)

    return available_actions[0], available_actions[1]


def _refresh_progress_state(env) -> None:
    update_progress = getattr(env.unwrapped, "_update_progress_state", None)
    if callable(update_progress):
        update_progress()


def get_progress_value(env) -> float:
    _refresh_progress_state(env)
    finish_target = float(getattr(env.unwrapped, "_finish_target_progress", 0.0))
    progress = float(getattr(env.unwrapped, "_unwrapped_progress", 0.0))
    if finish_target <= 0.0:
        return 0.0
    normalized = float(np.clip(progress / finish_target, 0.0, 1.0))
    return 2.0 * normalized - 1.0


def resolve_speed_bounds(vehicle) -> tuple[float, float]:
    target_speeds = getattr(vehicle, "target_speeds", None)
    if target_speeds is not None and len(target_speeds) > 0:
        min_speed = float(np.min(target_speeds))
        max_speed = float(np.max(target_speeds))
    else:
        min_speed = float(getattr(vehicle, "MIN_SPEED", -40.0))
        max_speed = float(getattr(vehicle, "MAX_SPEED", 40.0))
    if min_speed > max_speed:
        min_speed, max_speed = max_speed, min_speed
    if np.isclose(min_speed, max_speed):
        max_speed = min_speed + 1e-6
    return min_speed, max_speed


def normalize_speed(vehicle) -> float:
    min_speed, max_speed = resolve_speed_bounds(vehicle)
    span = max(max_speed - min_speed, 1e-6)
    normalized = 2.0 * (float(getattr(vehicle, "speed", 0.0)) - min_speed) / span - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def classify_terminal_state(env, zero_sum_config: ZeroSumConfig) -> TerminalOutcome:
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)

    success_fn = getattr(env.unwrapped, "_is_success", None)
    ego_success = bool(success_fn()) if callable(success_fn) else False
    truncated = bool(env.unwrapped._is_truncated())
    terminated = bool(env.unwrapped._is_terminated())

    ego_crashed = bool(getattr(ego_vehicle, "crashed", False))
    npc_crashed = bool(getattr(npc_vehicle, "crashed", False))
    ego_on_road = bool(getattr(ego_vehicle, "on_road", True))
    npc_on_road = bool(getattr(npc_vehicle, "on_road", True))

    if ego_success and ego_on_road and not ego_crashed:
        return TerminalOutcome(True, 1.0, -1.0, "ego_finished")

    if ego_crashed:
        return TerminalOutcome(True, -1.0, 1.0, "ego_collision")

    if npc_crashed:
        return TerminalOutcome(True, 0.0, 0.0, "npc_self_collision")

    if terminated and not ego_on_road:
        return TerminalOutcome(True, 0.0, 0.0, "ego_offroad_draw")

    if terminated and not npc_on_road:
        return TerminalOutcome(True, 0.0, 0.0, "npc_offroad_draw")

    if truncated:
        ego_speed = float(getattr(ego_vehicle, "speed", 0.0))
        if ego_on_road and ego_speed >= float(zero_sum_config.minimum_safe_speed):
            return TerminalOutcome(True, 1.0, -1.0, "ego_timeout_safe")
        return TerminalOutcome(True, 0.0, 0.0, "timeout_draw")

    if terminated:
        return TerminalOutcome(True, 0.0, 0.0, "terminated_draw")

    return TerminalOutcome(False, 0.0, 0.0, "ongoing")
