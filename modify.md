# THIẾT KẾ KIẾN TRÚC HỆ THỐNG ĐIỂM SỐ & HUẤN LUYỆN
**Dự án:** Trí tuệ nhân tạo đối kháng cho xe tự lái (AlphaZero POC trên Highway-Env)
**Phiên bản:** 1.0 - Discrete Meta-Action (5 Actions)

---

## 1. Ma trận Kết quả Cuối cùng (Terminal Outcome Matrix - z)

Giá trị tuyệt đối `z` thuộc tập {-1.0, 0.0, 1.0} được sinh ra tại thời điểm một trong hai tác tử kết thúc ván đấu. Đây là hạt giống (seed) để tính toán giá trị học cho mạng nơ-ron.

| Kịch bản kết thúc (Terminal Condition) | Outcome Ego (z_e) | Outcome NPC (z_n) | Hành động của Simulator |
| :--- | :---: | :---: | :--- |
| **1. NPC đâm trúng Ego** | `-1.0` | `+1.0` | Kết thúc hoàn toàn Episode. NPC chiến thắng tuyệt đối. |
| **2. Hết giờ (Timeout) - Ego còn sống**| `+1.0` | `-1.0` | Kết thúc hoàn toàn Episode. Ego trốn thoát thành công. |
| **3. Ego tự đâm (vào IDM/Tường)** | `-1.0` | `0.0` | Kết thúc hoàn toàn Episode. NPC không được điểm thưởng để tránh sinh ra hành vi "lười biếng rình mồi". |
| **4. NPC tự đâm (vào IDM/Tường)** | (Chưa có) | `-1.0` | Simulator **XÓA NPC**, Ego tiếp tục chạy. Chuỗi dữ liệu của NPC kết thúc tại đây và được gán nhãn -1.0. |

---

## 2. Cơ chế Truyền ngược trong MCTS (MCTS Value Backpropagation)

Khi MCTS chạy mô phỏng (Simulation), hệ thống sử dụng hệ số chiết khấu `gamma` (Discount Factor, VD: gamma = 0.99) để đánh giá các nhánh. 

**Quy tắc cập nhật Node:**
Khi duyệt từ dưới lên, giá trị của Node cha được cập nhật bằng công thức:
> `V_parent = gamma * V_child`

**Xử lý tại Node lá (Leaf Nodes):**
* **Nếu Node lá là Terminal State (Đâm xe / Timeout):** `V_child` chính là giá trị `z` lấy từ Ma trận Outcome ở Phần 1.
* **Nếu Node lá chưa kết thúc (Non-terminal):** MCTS gọi mạng Nơ-ron để lấy dự đoán: `V_child = V_network(s_leaf)`.

**Hệ quả:** MCTS sẽ tự động ưu tiên các nhánh có khả năng sinh tồn lâu nhất (giá trị ít âm nhất) để tránh hiện tượng tự sát (Pessimistic Agent).

---

## 3. Giá trị Mục tiêu của Mạng Nơ-ron (v_target)

Trong Replay Buffer, giá trị `z` không được gán đồng loạt cho mọi frame. Hệ thống áp dụng cơ chế **Suy giảm giá trị theo thời gian (Time-decayed Value)**.

Cho một chuỗi độ dài `T`, tại mỗi bước `t`, giá trị mục tiêu mà mạng Nơ-ron cần dự đoán là:
> `v_target(t) = z * (gamma ^ (T - t))`

**Hàm Loss của Value Head:**
Sử dụng Mean Squared Error (MSE) giữa dự đoán của Value Head và v_target:
> `Loss_value = MSE(V_predict(t), v_target(t))`

**Phân tích Tâm lý học thuật toán:**
* **Ego (z_e = -1):** Càng gần thời điểm bị đâm, `v_target` càng tiến về -1. Mạng học được "sự sợ hãi" và rủi ro va chạm.
* **NPC (z_n = +1):** Càng gần thời điểm bắt được Ego, `v_target` càng tiến về +1. Mạng học được "sự hưng phấn" và động lực tăng tốc dứt điểm.
* **NPC (z_n = -1 khi Timeout):** Càng về cuối, giá trị càng tiến về -1. Mạng học được áp lực về mặt thời gian (Urgency).

---

## 4. Đặc tả Kỹ thuật Triển khai (Implementation Notes)

### 4.1. Xử lý độ dài chuỗi bất đối xứng (Asymmetric Trajectories)
Do NPC có thể tự đâm và kết thúc ở frame 50, trong khi Ego tiếp tục sống đến frame 200 (giả sử là số như này, có thể linh động tính toán dựa trên độ dài trajectory):
* Chiều dài chuỗi NPC: `T_npc = 50`. Outcome: `z_n = -1.0`.
* Chiều dài chuỗi Ego: `T_ego = 200`. Outcome: `z_e = 1.0`.
* **Cảnh báo:** Phải sử dụng đúng chiều dài `T` tương ứng của từng tác tử để tính `v_target(t)`.

### 4.2. Xử lý Trạng thái Ego sau khi NPC chết (Zero-Padding)
Từ frame thứ 51 đến 200 (khi NPC đã bị xóa):
* Tensor đầu vào của Ego (Occupancy Grid) phải **lấp đầy Kênh Số 2** (kênh chứa tọa độ Hunter) bằng ma trận toàn số 0 (Zero Matrix).
* Điều này cho phép Ego rèn luyện khả năng lái xe "hòa bình" ngay trong cùng một ván đấu.

### 4.3. Cấu hình Value Head
* Do giá trị `v_target` luôn nằm trong khoảng `[-1.0, 1.0]`.
* Lớp cuối cùng (Output Layer) của Value Head **bắt buộc sử dụng hàm kích hoạt Tanh**. Tuyệt đối không dùng ReLU hay Linear.

### 4.4. Tinh chỉnh Hệ số Gamma
* `gamma = 0.99`: Tầm nhìn xa, phù hợp đường cao tốc dài, tốc độ ổn định.
* `gamma = 0.95`: Tầm nhìn ngắn, phản xạ cực nhanh, phù hợp đoạn đường chật chội, cần lách lách liên tục.