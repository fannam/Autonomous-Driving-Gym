import numpy as np


def softmax_policy(policy, available_actions):
    """
    Normalize policy values over available actions.

    `policy` is expected to contain all action ids as keys.
    Unavailable actions are forced to 0.0.
    """
    updated_policy = {action: 0.0 for action in policy}
    if not available_actions:
        return updated_policy

    available_probs = np.asarray([policy[action] for action in available_actions], dtype=np.float32)
    prob_sum = float(np.sum(available_probs))
    if not np.isfinite(prob_sum) or prob_sum <= 0.0:
        uniform_prob = 1.0 / len(available_actions)
        for action in available_actions:
            updated_policy[action] = uniform_prob
        return updated_policy

    softmax_probs = available_probs / prob_sum
    for action, prob in zip(available_actions, softmax_probs):
        updated_policy[action] = prob
    return updated_policy
