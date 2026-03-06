import numpy as np

def softmax_policy(policy, available_actions):
    """
    Áp dụng softmax cho các xác suất trong policy dựa trên available_actions.

    :param policy: Dictionary chứa 5 hành động [0, 1, 2, 3, 4] với các xác suất tương ứng.
    :param available_actions: Danh sách các hành động có thể thực hiện (subset của [0, 1, 2, 3, 4]).
    :return: Dictionary chứa xác suất mới cho từng hành động (softmax áp dụng với các hành động khả dụng).
    """
    # Lấy các giá trị xác suất tương ứng với available_actions
    available_probs = np.array([policy[action] for action in available_actions])

    # Áp dụng softmax chỉ trên available_probs
    softmax_probs = available_probs / np.sum(available_probs)

    # Cập nhật xác suất mới
    updated_policy = {action: 0.0 for action in policy}  # Khởi tạo tất cả xác suất bằng 0
    for action, prob in zip(available_actions, softmax_probs):
        updated_policy[action] = prob

    return updated_policy