import numpy as np
import highway_env

# Buffer để lưu lịch sử Occupancy Grid
history_length = 5  # Số lượng occupancy grid cần lưu
ego_position = (4, 2)
# Kích thước của occupancy grid từ môi trường
grid_size = (21, 5)  # Ví dụ: W x H từ highway-env

def init_stack_of_grid(grid_size, ego_position, history_length=5):
    """
    Khởi tạo một mảng numpy chứa các occupancy grid và thông tin về tốc độ.

    Args:
        grid_size (tuple): Kích thước của grid (W, H).
        ego_position (tuple): Tọa độ của ego vehicle trong grid (x, y).
        max_speed (float): Tốc độ tối đa của xe.
        min_speed (float): Tốc độ tối thiểu của xe.
        history_length (int): Số lượng grid cần lưu cho lịch sử.

    Returns:
        numpy.ndarray: Một mảng numpy chứa các grid và thông tin tốc độ (shape: (history_length + 3, W, H)).
    """
    W, H = grid_size

    # Khởi tạo history stack với ego_position
    init_grid = np.zeros((W, H), dtype=np.float32)
    init_grid[ego_position] = 1.0
    history_stack = np.stack([init_grid.copy() for _ in range(history_length)], axis=0)

    # Tạo grid cho thông tin làn đường và tốc độ (toàn số 0 ban đầu)
    lane_info_grid = np.zeros((W, H), dtype=np.float32)
    speed_max_grid = np.full((W, H), 0.0, dtype=np.float32)
    speed_min_grid = np.full((W, H), 0.0, dtype=np.float32)
    ego_absolute_speed = np.full((W, H), 0.0, dtype=np.float32)
    # Ghép thành stack đầy đủ
    stack = np.concatenate([history_stack,
                            lane_info_grid[np.newaxis, :, :],
                            speed_max_grid[np.newaxis, :, :],
                            speed_min_grid[np.newaxis, :, :],
                            ego_absolute_speed[np.newaxis, :, :]], axis=0)
    return stack

def get_stack_of_grid(env, stack, new_grid, history_length=5):
    """
    Cập nhật stack với occupancy grid mới, thông tin làn đường và tốc độ tương đối.

    Args:
        stack (numpy.ndarray): Stack chứa lịch sử grid và thông tin tốc độ.
        new_grid (list of numpy.ndarray):
            - new_grid[0]: Occupancy grid mới.
            - new_grid[1]: Thông tin làn đường.
        history_length (int): Số lượng grid lịch sử.

    Returns:
        numpy.ndarray: Stack đã được cập nhật.
    """
    # Cập nhật history stack
    stack[:history_length - 1] = stack[1:history_length]
    stack[history_length - 1] = new_grid[0]
    stack[history_length] = new_grid[1]
    ego_speed = env.unwrapped.road.vehicles[0].speed
    max_speed = env.unwrapped.road.vehicles[0].target_speeds[-1]
    min_speed = env.unwrapped.road.vehicles[0].target_speeds[0]
    relative_speed_max = (ego_speed-max_speed)/(max_speed-min_speed)
    relative_speed_min = (ego_speed-min_speed)/(max_speed-min_speed)
    # Cập nhật stack tốc độ
    speed_max_grid = np.full(stack.shape[1:], relative_speed_max, dtype=np.float32)
    speed_min_grid = np.full(stack.shape[1:], relative_speed_min, dtype=np.float32)
    ego_absolute_speed = np.full(stack.shape[1:], ego_speed, dtype=np.float32)
    # Cập nhật stack tốc độ
    stack[history_length + 1] = speed_max_grid
    stack[history_length + 2] = speed_min_grid
    stack[history_length + 3] = ego_absolute_speed

    return stack
