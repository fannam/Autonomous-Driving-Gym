from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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


_AGENT_ACTIVE_MASK_ATTR = "_alphazero_agent_active_mask"
_AGENT_TERMINAL_VALUES_ATTR = "_alphazero_agent_terminal_values"
_AGENT_TERMINAL_REASONS_ATTR = "_alphazero_agent_terminal_reasons"


def _runtime_container(env):
    return getattr(env, "unwrapped", env)


def reset_agent_runtime_state(env) -> None:
    env_unwrapped = _runtime_container(env)
    vehicles = tuple(getattr(env_unwrapped, "controlled_vehicles", ()))
    setattr(
        env_unwrapped,
        _AGENT_ACTIVE_MASK_ATTR,
        [True for _ in vehicles],
    )
    setattr(
        env_unwrapped,
        _AGENT_TERMINAL_VALUES_ATTR,
        [None for _ in vehicles],
    )
    setattr(
        env_unwrapped,
        _AGENT_TERMINAL_REASONS_ATTR,
        [None for _ in vehicles],
    )


def _ensure_agent_runtime_state(env):
    env_unwrapped = _runtime_container(env)
    vehicles = tuple(getattr(env_unwrapped, "controlled_vehicles", ()))
    active_mask = getattr(env_unwrapped, _AGENT_ACTIVE_MASK_ATTR, None)
    terminal_values = getattr(env_unwrapped, _AGENT_TERMINAL_VALUES_ATTR, None)
    terminal_reasons = getattr(env_unwrapped, _AGENT_TERMINAL_REASONS_ATTR, None)

    needs_reset = (
        active_mask is None
        or terminal_values is None
        or terminal_reasons is None
        or len(active_mask) != len(vehicles)
        or len(terminal_values) != len(vehicles)
        or len(terminal_reasons) != len(vehicles)
    )

    if needs_reset:
        reset_agent_runtime_state(env)
    return env_unwrapped


def get_agent_active_mask(env) -> tuple[bool, bool]:
    env_unwrapped = _ensure_agent_runtime_state(env)
    active_mask = tuple(
        bool(active)
        for active in getattr(env_unwrapped, _AGENT_ACTIVE_MASK_ATTR, ())
    )
    if len(active_mask) < 2:
        raise RuntimeError(
            "The adversarial implementation expects at least two controlled vehicles."
        )
    return active_mask[0], active_mask[1]


def get_agent_terminal_values(env) -> tuple[float | None, float | None]:
    env_unwrapped = _ensure_agent_runtime_state(env)
    terminal_values = tuple(
        None if value is None else float(value)
        for value in getattr(env_unwrapped, _AGENT_TERMINAL_VALUES_ATTR, ())
    )
    if len(terminal_values) < 2:
        raise RuntimeError(
            "The adversarial implementation expects at least two controlled vehicles."
        )
    return terminal_values[0], terminal_values[1]


def _current_agent_values(env) -> tuple[float, float]:
    ego_value, npc_value = get_agent_terminal_values(env)
    return (
        0.0 if ego_value is None else float(ego_value),
        0.0 if npc_value is None else float(npc_value),
    )


def apply_terminal_value_overrides(
    env,
    ego_value: float,
    npc_value: float,
) -> tuple[float, float]:
    if env is None:
        return float(ego_value), float(npc_value)

    fixed_ego_value, fixed_npc_value = get_agent_terminal_values(env)
    if fixed_ego_value is not None:
        ego_value = fixed_ego_value
    if fixed_npc_value is not None:
        npc_value = fixed_npc_value
    return float(ego_value), float(npc_value)


def _deactivate_agent(env, agent_index: int) -> None:
    env_unwrapped = _ensure_agent_runtime_state(env)
    active_mask = getattr(env_unwrapped, _AGENT_ACTIVE_MASK_ATTR)
    if not active_mask[agent_index]:
        return
    active_mask[agent_index] = False

    vehicles = tuple(getattr(env_unwrapped, "controlled_vehicles", ()))
    if agent_index >= len(vehicles):
        return
    vehicle = vehicles[agent_index]
    road = getattr(vehicle, "road", None)
    road_vehicles = getattr(road, "vehicles", None)
    if road is not None and road_vehicles is not None:
        road.vehicles = [candidate for candidate in road_vehicles if candidate is not vehicle]


def _record_agent_terminal_value(
    env,
    agent_index: int,
    value: float,
    *,
    reason: str,
    deactivate: bool = False,
) -> None:
    env_unwrapped = _ensure_agent_runtime_state(env)
    terminal_values = getattr(env_unwrapped, _AGENT_TERMINAL_VALUES_ATTR)
    terminal_reasons = getattr(env_unwrapped, _AGENT_TERMINAL_REASONS_ATTR)

    if terminal_values[agent_index] is None:
        terminal_values[agent_index] = float(np.clip(value, -1.0, 1.0))
    if terminal_reasons[agent_index] is None:
        terminal_reasons[agent_index] = str(reason)
    if deactivate:
        _deactivate_agent(env, agent_index)


def _outcome_from_runtime(
    env,
    *,
    terminal: bool,
    reason: str,
) -> TerminalOutcome:
    ego_value, npc_value = _current_agent_values(env)
    return TerminalOutcome(
        terminal=bool(terminal),
        ego_value=float(ego_value),
        npc_value=float(npc_value),
        reason=str(reason),
    )


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


def get_scripted_action(env, agent_index: int, mode: str | None) -> int | None:
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
    active_mask = get_agent_active_mask(env)

    for agent_index, mode in enumerate(policy_modes):
        if not active_mask[agent_index]:
            action_type = _get_agent_action_type(env, agent_index)
            available_actions.append((neutral_action_index(action_type),))
            continue

        scripted_action = get_scripted_action(env, agent_index, mode)
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
    if finish_target > 0.0:
        normalized = float(np.clip(progress / finish_target, 0.0, 1.0))
        return 2.0 * normalized - 1.0

    distance_fn = getattr(env.unwrapped, "_distance_travelled", None)
    required_distance = getattr(env.unwrapped, "_required_success_distance", None)
    if callable(distance_fn) and required_distance is not None:
        required_distance = float(required_distance)
        if required_distance > 0.0:
            normalized = float(
                np.clip(float(distance_fn()) / required_distance, 0.0, 1.0)
            )
            return 2.0 * normalized - 1.0
    return 0.0


def get_speed_bounds(vehicle) -> tuple[float, float]:
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
    min_speed, max_speed = get_speed_bounds(vehicle)
    span = max(max_speed - min_speed, 1e-6)
    normalized = 2.0 * (float(getattr(vehicle, "speed", 0.0)) - min_speed) / span - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def _collision_partners(vehicle) -> tuple[Any, ...]:
    partners = getattr(vehicle, "collision_partners", ())
    if partners is None:
        return ()
    return tuple(partners)


def _collided_with(vehicle, other) -> bool:
    return any(partner is other for partner in _collision_partners(vehicle))


def _is_intersecting(vehicle, other) -> bool:
    is_colliding = getattr(vehicle, "_is_colliding", None)
    if not callable(is_colliding):
        return False

    try:
        intersecting, _, _ = is_colliding(other, 0.0)
    except TypeError:
        try:
            intersecting, _, _ = is_colliding(other)
        except Exception:
            return False
    except Exception:
        return False
    return bool(intersecting)


def _vehicles_collided_together(ego_vehicle, npc_vehicle) -> bool:
    return (
        _collided_with(ego_vehicle, npc_vehicle)
        or _collided_with(npc_vehicle, ego_vehicle)
        or _is_intersecting(ego_vehicle, npc_vehicle)
    )


def _classify_collision_outcome(
    ego_vehicle,
    npc_vehicle,
    *,
    ego_active: bool,
    npc_active: bool,
) -> TerminalOutcome | None:
    ego_crashed = bool(getattr(ego_vehicle, "crashed", False)) if ego_active else False
    npc_crashed = bool(getattr(npc_vehicle, "crashed", False)) if npc_active else False

    if not ego_crashed and not npc_crashed:
        return None

    if ego_crashed and npc_crashed and _vehicles_collided_together(ego_vehicle, npc_vehicle):
        return TerminalOutcome(True, -1.0, 1.0, "npc_hit_ego")

    if ego_crashed and npc_crashed:
        return TerminalOutcome(True, -1.0, -1.0, "double_self_collision")

    if ego_crashed:
        return TerminalOutcome(True, -1.0, 0.0, "ego_self_collision")

    return TerminalOutcome(True, 0.0, -1.0, "npc_self_collision")


def _classify_offroad_outcome(
    ego_vehicle,
    npc_vehicle,
    *,
    ego_active: bool,
    npc_active: bool,
) -> TerminalOutcome | None:
    ego_on_road = bool(getattr(ego_vehicle, "on_road", True)) if ego_active else True
    npc_on_road = bool(getattr(npc_vehicle, "on_road", True)) if npc_active else True

    if ego_on_road and npc_on_road:
        return None

    if not ego_on_road and not npc_on_road:
        return TerminalOutcome(True, -1.0, -1.0, "double_offroad")

    if not ego_on_road:
        return TerminalOutcome(True, -1.0, 0.0, "ego_offroad")

    return TerminalOutcome(True, 0.0, -1.0, "npc_offroad")


def classify_terminal_state(env, zero_sum_config) -> TerminalOutcome:
    _ensure_agent_runtime_state(env)
    ego_vehicle, npc_vehicle = get_controlled_vehicles(env)
    ego_active, npc_active = get_agent_active_mask(env)
    remove_npc_on_self_fault = bool(
        getattr(zero_sum_config, "remove_npc_on_self_fault", False)
    )

    success_fn = getattr(env.unwrapped, "_is_success", None)
    ego_success = bool(success_fn()) if callable(success_fn) and ego_active else False
    truncated = bool(env.unwrapped._is_truncated())
    terminated = bool(env.unwrapped._is_terminated())

    ego_on_road = bool(getattr(ego_vehicle, "on_road", True)) if ego_active else True
    ego_crashed = bool(getattr(ego_vehicle, "crashed", False)) if ego_active else False

    if ego_success and ego_on_road and not ego_crashed:
        _record_agent_terminal_value(env, 0, 1.0, reason="ego_finished")
        if npc_active:
            _record_agent_terminal_value(env, 1, -1.0, reason="ego_finished")
        return _outcome_from_runtime(env, terminal=True, reason="ego_finished")

    collision_outcome = _classify_collision_outcome(
        ego_vehicle,
        npc_vehicle,
        ego_active=ego_active,
        npc_active=npc_active,
    )
    if collision_outcome is not None:
        if collision_outcome.reason == "npc_hit_ego":
            _record_agent_terminal_value(env, 0, -1.0, reason=collision_outcome.reason)
            _record_agent_terminal_value(env, 1, 1.0, reason=collision_outcome.reason)
            return _outcome_from_runtime(env, terminal=True, reason=collision_outcome.reason)

        if collision_outcome.reason == "double_self_collision":
            if ego_active:
                _record_agent_terminal_value(env, 0, -1.0, reason=collision_outcome.reason)
            if npc_active:
                _record_agent_terminal_value(env, 1, -1.0, reason=collision_outcome.reason)
            return _outcome_from_runtime(env, terminal=True, reason=collision_outcome.reason)

        if collision_outcome.reason == "ego_self_collision":
            _record_agent_terminal_value(env, 0, -1.0, reason=collision_outcome.reason)
            return _outcome_from_runtime(env, terminal=True, reason=collision_outcome.reason)

        if collision_outcome.reason == "npc_self_collision":
            _record_agent_terminal_value(
                env,
                1,
                -1.0,
                reason=collision_outcome.reason,
                deactivate=remove_npc_on_self_fault,
            )
            return _outcome_from_runtime(
                env,
                terminal=not remove_npc_on_self_fault,
                reason=collision_outcome.reason,
            )

    offroad_outcome = _classify_offroad_outcome(
        ego_vehicle,
        npc_vehicle,
        ego_active=ego_active,
        npc_active=npc_active,
    )
    if offroad_outcome is not None:
        if offroad_outcome.reason == "double_offroad":
            if ego_active:
                _record_agent_terminal_value(env, 0, -1.0, reason=offroad_outcome.reason)
            if npc_active:
                _record_agent_terminal_value(env, 1, -1.0, reason=offroad_outcome.reason)
            return _outcome_from_runtime(env, terminal=True, reason=offroad_outcome.reason)

        if offroad_outcome.reason == "ego_offroad":
            _record_agent_terminal_value(env, 0, -1.0, reason=offroad_outcome.reason)
            return _outcome_from_runtime(env, terminal=True, reason=offroad_outcome.reason)

        if offroad_outcome.reason == "npc_offroad":
            _record_agent_terminal_value(
                env,
                1,
                -1.0,
                reason=offroad_outcome.reason,
                deactivate=remove_npc_on_self_fault,
            )
            return _outcome_from_runtime(
                env,
                terminal=not remove_npc_on_self_fault,
                reason=offroad_outcome.reason,
            )

    if truncated:
        ego_speed = float(getattr(ego_vehicle, "speed", 0.0))
        if ego_on_road and ego_speed >= float(zero_sum_config.minimum_safe_speed):
            _record_agent_terminal_value(env, 0, 1.0, reason="ego_timeout_safe")
            if npc_active:
                _record_agent_terminal_value(env, 1, -1.0, reason="ego_timeout_safe")
            return _outcome_from_runtime(env, terminal=True, reason="ego_timeout_safe")
        return _outcome_from_runtime(env, terminal=True, reason="timeout_draw")

    if terminated:
        return _outcome_from_runtime(env, terminal=True, reason="terminated_draw")

    return _outcome_from_runtime(env, terminal=False, reason="ongoing")
