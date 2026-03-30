from __future__ import annotations

import numpy as np


def action_index_to_axes(
    action: int,
    *,
    n_action_axis_0: int,
    n_action_axis_1: int,
) -> tuple[int, int]:
    action_index = int(action)
    action_count = int(n_action_axis_0) * int(n_action_axis_1)
    if not 0 <= action_index < action_count:
        raise ValueError(
            f"Action index {action_index} is out of range for "
            f"shape=({n_action_axis_0}, {n_action_axis_1})."
        )
    return divmod(action_index, int(n_action_axis_1))


def axes_to_action_index(axis_0: int, axis_1: int, *, n_action_axis_1: int) -> int:
    return int(axis_0) * int(n_action_axis_1) + int(axis_1)


def reverse_steering_distribution(distribution) -> np.ndarray:
    values = np.asarray(distribution, dtype=np.float32).reshape(-1)
    return np.ascontiguousarray(values[::-1])


def outer_product_policy(
    accelerate_policy,
    steering_policy,
    *,
    n_action_axis_0: int,
    n_action_axis_1: int,
    flip_steering: bool = False,
) -> np.ndarray:
    accelerate = np.asarray(accelerate_policy, dtype=np.float32).reshape(-1)
    steering = np.asarray(steering_policy, dtype=np.float32).reshape(-1)
    if accelerate.shape[0] != int(n_action_axis_0):
        raise ValueError(
            f"Expected {n_action_axis_0} accelerate bins, got {accelerate.shape[0]}."
        )
    if steering.shape[0] != int(n_action_axis_1):
        raise ValueError(
            f"Expected {n_action_axis_1} steering bins, got {steering.shape[0]}."
        )
    if flip_steering:
        steering = reverse_steering_distribution(steering)
    return np.outer(accelerate, steering).reshape(-1).astype(np.float32, copy=False)


def policy_dict_to_array(policy_dict: dict[int, float], n_actions: int) -> np.ndarray:
    vector = np.zeros(int(n_actions), dtype=np.float32)
    for action, prob in policy_dict.items():
        action_index = int(action)
        if 0 <= action_index < int(n_actions):
            vector[action_index] = float(prob)
    return vector


def marginalize_action_policy(
    policy,
    *,
    n_action_axis_0: int,
    n_action_axis_1: int,
    flip_steering: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    flat_policy = np.asarray(policy, dtype=np.float32).reshape(-1)
    expected_size = int(n_action_axis_0) * int(n_action_axis_1)
    if flat_policy.shape[0] != expected_size:
        raise ValueError(
            f"Expected flat policy of size {expected_size}, got {flat_policy.shape[0]}."
        )

    joint_policy = flat_policy.reshape(int(n_action_axis_0), int(n_action_axis_1))
    accelerate = np.sum(joint_policy, axis=1).astype(np.float32, copy=False)
    steering = np.sum(joint_policy, axis=0).astype(np.float32, copy=False)
    if flip_steering:
        steering = reverse_steering_distribution(steering)
    return accelerate, steering


def normalize_policy(
    policy: dict[int, float],
    available_actions: tuple[int, ...],
) -> dict[int, float]:
    """Normalize prior values over the available actions only."""
    normalized = {int(action): 0.0 for action in policy}
    if not available_actions:
        return normalized

    available_probs = np.asarray(
        [float(policy.get(action, 0.0)) for action in available_actions],
        dtype=np.float64,
    )
    available_probs = np.where(
        np.isfinite(available_probs) & (available_probs > 0.0),
        available_probs,
        0.0,
    )
    prob_sum = float(np.sum(available_probs))
    if prob_sum <= 0.0 or not np.isfinite(prob_sum):
        uniform_prob = 1.0 / len(available_actions)
        for action in available_actions:
            normalized[int(action)] = uniform_prob
        return normalized

    available_probs /= prob_sum
    if not np.all(np.isfinite(available_probs)):
        uniform_prob = 1.0 / len(available_actions)
        for action in available_actions:
            normalized[int(action)] = uniform_prob
        return normalized
    for action, prob in zip(available_actions, available_probs):
        normalized[int(action)] = float(prob)
    return normalized
