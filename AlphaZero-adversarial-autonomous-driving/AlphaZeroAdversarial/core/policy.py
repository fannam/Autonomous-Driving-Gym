from __future__ import annotations

import numpy as np


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
        dtype=np.float32,
    )
    prob_sum = float(np.sum(available_probs))
    if prob_sum <= 0.0 or not np.isfinite(prob_sum):
        uniform_prob = 1.0 / len(available_actions)
        for action in available_actions:
            normalized[int(action)] = uniform_prob
        return normalized

    available_probs /= prob_sum
    for action, prob in zip(available_actions, available_probs):
        normalized[int(action)] = float(prob)
    return normalized
