import numpy as np


def softmax_policy(policy, available_actions):
    """
    Normalize policy values over available actions.

    `policy` is expected to contain all action ids as keys.
    Unavailable actions are forced to 0.0.
    """
    available_probs = np.array([policy[action] for action in available_actions])
    softmax_probs = available_probs / np.sum(available_probs)

    updated_policy = {action: 0.0 for action in policy}
    for action, prob in zip(available_actions, softmax_probs):
        updated_policy[action] = prob
    return updated_policy
