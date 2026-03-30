# 📝 Plan: AlphaZero MCTS Improvements for Highway-Env

Bản kế hoạch này hệ thống hóa các cải tiến về kiến trúc Mạng Nơ-ron và thuật toán MCTS nhằm tối ưu hóa khả năng điều khiển liên tục trong môi trường đối kháng.

---

## 1. 🧠 Kiến trúc Mạng Nơ-ron (Factorized Multi-Head)
Thay đổi từ kiến trúc 25-index phẳng sang kiến trúc phân tách đặc trưng (Factorized) để tối ưu hóa khả năng học và tổng quát hóa vật lý.

### 1.1. Backbone (Shared Features)
* **Input:** Occupancy Grid $(2N+k, H, W)$.
* **Shared Weights:** Ego và NPC dùng chung một bộ trọng số duy nhất. 
* **Perspective Flip:** Sử dụng toán học để lật ngược góc nhìn (Y-axis, Steering angle) khi tính toán cho NPC, giúp mạng dùng chung "tư duy" lái xe.

### 1.2. Policy & Value Heads
* **Accelerate Head:** `Linear(1024, n_actions_axis_0)` -> `Softmax`. Các mức: `[accelerate_min, ..., accelerate_max]`.
* **Steering Head:** `Linear(1024, n_action_axis_1)` -> `Softmax`. Các mức: `[steering_min, ..., steering_max]`.
* **Value Head:** `Linear(1024, 1)` -> `Tanh`. Dự đoán độ an toàn/chiến thắng trong khoảng $[-1, 1]$.

---

## 2. 🌳 Lõi MCTS Đối kháng (Adversarial & Decoupled)
Nâng cấp thuật toán tìm kiếm để xử lý tổ hợp hành động của 2 tác tử mà không làm nông cây MCTS.

### 2.1. Cơ chế Duyệt nhánh (Selection & Expansion)
* **Joint Action Probability:** Tại bước Expand, tái cấu trúc vector 25 xác suất bằng phép nhân chéo (Outer Product): 
  $$P(i, j) = p_{accel}[i] \times p_{steer}[j]$$
* **Decoupled UCT:** Lưu trữ độc lập bảng thống kê $\{N, W, Q\}$ cho Ego và NPC tại mỗi Node. Điều này giúp tối ưu hóa hiệu suất lấy mẫu (Sample Efficiency).
* **Relative Threshold Pruning:** Cắt tỉa các nhánh có xác suất thấp hơn ngưỡng $\gamma \cdot \max(P)$ để tập trung ngân sách `n_simulations` vào các nhánh tiềm năng.

---

## 3. 📊 Luồng Dữ liệu & Huấn luyện (Data Pipeline)
Quy trình "phiên dịch" dữ liệu giữa cây MCTS và Mạng Nơ-ron.

### 3.1. Tạo Nhãn Mục tiêu (Target Policy)
* **Marginalization (Cộng gộp biên):** Chuyển đổi 25 visit counts của MCTS thành 2 mảng mục tiêu 1D (size 5) cho Ga và Lái bằng cách cộng dồn theo trục.
* **Gaussian Label Smoothing (Experimental):** * Flag: `use_smoothing`. Cho phép bật tắt feature này
    * Áp dụng làm mịn Gaussian 1D để dạy mạng khái niệm "khoảng cách vật lý", giúp hội tụ nhanh trong giai đoạn đầu.

### 3.2. Loss Function
Sử dụng **KL-Divergence** cho Policy và **MSE** cho Value:
$$Loss = Loss_{Value} + Loss_{Accel} + Loss_{Steer}$$
---

## 4. 🛠 Checklist Thực nghiệm (A/B Testing)

| Tham số/Tính năng | Giá trị đề xuất | Mục tiêu kiểm chứng |
| :--- | :--- | :--- |
| **Factorized Heads** | On (Default) | Tốc độ hội tụ so với kiến trúc 25-index cũ. |
| **Relative Gamma** ($\gamma$) | $0.1$ | Độ sâu của cây (Depth) vs. Độ phủ nhánh (Breadth). |
| **Smoothing Flag** | True/False | Độ mượt của quỹ đạo vs. Khả năng né vật cản đa đỉnh. |
| **N_simulations** | $250$ | Hiệu suất thời gian thực trên CPU. |

---

## 🚀 Roadmap Triển khai
1. **Phase 1 (Model):** Refactor `model.py` để tách 3 Heads. Implement hàm `outer_product` trong class MCTS.
2. **Phase 2 (MCTS Logic):** Cập nhật `MCTSNode` hỗ trợ Decoupled Stats và cơ chế lật tọa độ NPC.
3. **Phase 3 (Buffer):** Viết logic Marginalization để lưu dữ liệu vào Replay Buffer dưới dạng 2 mảng Policy riêng biệt.