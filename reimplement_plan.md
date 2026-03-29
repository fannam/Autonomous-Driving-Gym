# Kế hoạch Triển khai: Hệ thống Tự lái Đa tác tử Đối kháng (Multi-Agent Adversarial MCTS)

Bản kế hoạch này phác thảo cấu trúc lý thuyết và kỹ thuật để xây dựng một hệ thống AI tự lái, trong đó xe chính (Ego) được huấn luyện thông qua việc sinh tồn trước một xe đối thủ (Adversarial NPC) luôn tìm cách gây tai nạn. Hệ thống sử dụng thuật toán Tìm kiếm cây Monte Carlo Đồng thời (Simultaneous MCTS) kết hợp với Mạng Nơ-ron dùng chung trọng số.

---

## PHẦN 1: CƠ SỞ LÝ THUYẾT CỦA MULTI-AGENT MCTS ĐỐI KHÁNG

### 1.1. Bản chất Môi trường: Trò chơi Hành động Đồng thời
Khác với các board game đánh theo lượt (Turn-based) như Cờ vua hay Cờ vây, môi trường xe tự lái (`highway-env`) là một **trò chơi hành động đồng thời (Simultaneous-move game)**.
* Tại mỗi bước thời gian $t$, cả Ego và NPC đều phải ra quyết định mà không biết trước lựa chọn của đối phương.
* Cây MCTS không được phép phân nhánh theo lượt (ví dụ: Ego đi trước, NPC đi sau) vì điều đó vi phạm tính vật lý của môi trường.

### 1.2. MCTS Đồng thời (Decoupled UCT)
Để giải quyết tính đồng thời, một nút (Node) trong cây MCTS đại diện cho một khoảnh khắc thời gian bị đóng băng. Tại nút này:
* **Lưu trữ độc lập:** Node chứa 2 bộ thống kê song song: Khối lượng ghé thăm và Giá trị cho các hành động của Ego, và tương tự cho các hành động của NPC.
* **Định tuyến (Traverse):** Hàm PUCT được gọi 2 lần độc lập. Ego chọn hành động tối đa hóa lợi thế của Ego. NPC chọn hành động tối đa hóa lợi thế của NPC. Cây MCTS đi xuống nhánh đại diện cho cặp hành động chung `(action_ego, action_npc)`.
* **Truyền ngược (Backpropagation):** Khi nhận được điểm số dự đoán tại nút lá, giá trị này được truyền thẳng về gốc. Không có sự "đảo dấu theo độ sâu", mà giá trị được cập nhật trực tiếp vào bộ thống kê tương ứng của mỗi xe ở từng tầng.

### 1.3. Thiết lập Đối kháng Tổng bằng không (Zero-Sum Formulation)
Hệ thống được ép vào mô hình **Zero-sum game** để tạo động lực tiến hóa tối đa cho cả hai tác tử. Tổng điểm của Ego và NPC luôn bằng 0.

* **Mục tiêu của Ego (Phòng thủ):** Sống sót, đi đúng luật, tránh va chạm.
* **Mục tiêu của NPC (Tấn công):** Tìm mọi cách ép Ego phải va chạm (tạt đầu, phanh gấp).
* **Quy tắc tính điểm (Terminal Values):**
    * **NPC Thắng (Ego: -1, NPC: +1):** Xảy ra va chạm mà Ego là nạn nhân hoặc thủ phạm.
    * **Ego Thắng (Ego: +1, NPC: -1):** Hết thời gian mô phỏng, Ego tẩu thoát an toàn và đạt vận tốc yêu cầu.
    * **Hòa (Cả hai: 0):** Ego sống sót nhưng vi phạm luật (lao ra bãi cỏ, đi ngược chiều), hoặc NPC tự hủy (đâm vào xe khác không phải Ego). Trạng thái Hòa giúp chống lại hiện tượng "hack phần thưởng" (ví dụ: Ego đứng im để không bao giờ đâm).

---

## PHẦN 2: KIẾN TRÚC DỮ LIỆU & MẠNG NƠ-RON

Hệ thống sử dụng **1 Mạng Nơ-ron duy nhất** điều khiển cả Ego và NPC. Để làm được điều này, dữ liệu đầu vào phải tuân thủ nghiêm ngặt triết lý **Góc nhìn Bất biến (Perspective-Agnostic)**: Mạng không phân biệt danh tính xe Ego hay NPC, nó chỉ phân biệt "Bản thân (Self)" và "Đối thủ (Opponent)".

### 2.1. Cấu trúc Tensor Đầu vào ($2N + k$ Channels)
Đầu vào của mạng là một Occupancy Grid (Ma trận không gian lưới) kích thước `(C, H, W)`, với $C = 2N + k$. Khối Tensor này **luôn được đặt trung tâm tại chiếc xe đang cần được dự đoán hành động (Self)**.

* **Các kênh từ 0 đến N-1 (Lịch sử Bản thân):** Chứa lưới tọa độ của "Self" trong $N$ bước thời gian gần nhất. Chiếc xe này luôn nằm ở tâm bức ảnh.
* **Các kênh từ N đến 2N-1 (Lịch sử Đối thủ):** Chứa lưới tọa độ của "Opponent" trong $N$ bước thời gian gần nhất. Việc xếp lịch sử chồng lên nhau giúp mạng nơ-ron tự đạo hàm được vận tốc và quỹ đạo lao tới của đối thủ.
* **Các kênh từ 2N đến 2N+k-1 (Thuộc tính Cục bộ & Hạ tầng):** Chứa $k$ mặt phẳng biểu diễn các thông tin đặc thù của **riêng chiếc xe đang làm tâm (Self)** và môi trường vật lý xung quanh nó. Bao gồm:
    * Môi trường tĩnh được cắt theo góc nhìn của Self (Làn đường, vạch kẻ, lề đường).
    * Lộ trình mục tiêu (Target Route/Waypoint) mà Self muốn hướng tới.
    * Tốc độ hiện tại, gia tốc, hoặc góc đánh lái của riêng Self (được mã hóa thành các mặt phẳng vô hướng).

### 2.2. Kỹ thuật Inference Batching (Tráo đổi Góc nhìn)
Khi MCTS đứng tại một nút và cần xin dự đoán cho cả Ego và NPC, hệ thống không gọi hàm mạng nơ-ron 2 lần riêng biệt. Nó đóng gói 1 Batch kích thước `(2, 2N+k, H, W)`:

* **Phần tử số 0 (Góc nhìn Ego):**
    * Self: Lịch sử Ego (Ego ở tâm).
    * Opponent: Lịch sử NPC.
    * $k$ channels: Môi trường và Thuộc tính nội tại của Ego.
* **Phần tử số 1 (Góc nhìn NPC):**
    * Self: Lịch sử NPC (NPC ở tâm).
    * Opponent: Lịch sử Ego.
    * $k$ channels: Môi trường và Thuộc tính nội tại của NPC.

Bằng cách đưa khối Batch này qua mạng Nơ-ron 1 lần duy nhất, mạng sẽ xử lý song song 2 tình huống độc lập và trả về kết quả cho cả 2 xe cùng lúc, tối ưu hóa triệt để thời gian chạy.

### 2.3. Cấu trúc Đầu ra của Mô hình (Model Heads)
Mạng Nơ-ron bao gồm một phần thân (Backbone) trích xuất đặc trưng chung, và chia làm 2 nhánh đầu ra (Heads) ở cuối. Không đi sâu vào chi tiết số lớp mạng, đầu ra cần đảm bảo 2 thuộc tính:

1.  **Policy Head (Chính sách):** Trả về vector xác suất phân phối qua tất cả các hành động hợp lệ. Tổng các xác suất bằng 1.
2.  **Value Head (Giá trị):** Trả về một số thực vô hướng nằm trong khoảng [-1, 1]. Giá trị này dự đoán tỷ lệ chiến thắng của chiếc xe đang làm tâm (Self). Càng gần 1, Self càng an toàn; càng gần -1, Self càng dễ gặp tai nạn.

---

## PHẦN 3: LỘ TRÌNH THỰC THI (IMPLEMENTATION STEPS)

* **Bước 1: Trình tạo Grid (Grid Generator Pipeline)**
    * Viết hàm cấu trúc dữ liệu để duy trì hàng đợi lịch sử tọa độ $N$ bước.
    * Viết hàm trích xuất Tensor $2N+k$ linh hoạt bằng toán học/NumPy, tuyệt đối không dùng hàm render đồ họa của môi trường để đảm bảo tốc độ.
* **Bước 2: Nâng cấp MCTS Đồng thời**
    * Sửa đổi lớp Node để chứa 2 bộ thống kê độc lập cho Self và Opponent.
    * Viết lại logic Traverse để lấy cặp hành động chung dựa trên điểm PUCT độc lập của 2 xe.
    * Điều chỉnh hàm Backpropagate để truyền điểm không lật dấu theo tầng.
* **Bước 3: Tích hợp Model & Batching**
    * Xây dựng kiến trúc mạng Nơ-ron với Input $2N+k$ và Output (Policy, Value).
    * Cài đặt luồng ghép Batch 2 góc nhìn tại nút lá trước khi gọi `model.forward()`.
* **Bước 4: Huấn luyện (Self-Play & Learner)**
    * Cài đặt thuật toán Curriculum Learning: Huấn luyện Ego chạy an toàn trước, sau đó mới cho NPC tham gia đối kháng để tránh hiện tượng mất phương hướng ở giai đoạn đầu.
    * Viết vòng lặp Learner tối ưu hóa hàm Loss (kết hợp Policy Loss và Value Loss).