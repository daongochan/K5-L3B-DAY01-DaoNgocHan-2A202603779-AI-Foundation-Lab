# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** `2A202603779`
- **Họ và tên:** Đào Ngọc Hân
- **SEED** (in ở ô đầu notebook): `5095`

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô Báo cáo cuối sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

Bảng so sánh thực tế trên tập dữ liệu riêng theo `SEED = 5095` (`train = 3.621`, `val = 775`, `test = 775` gồm 103 tin rác, `SUB = 40` cân bằng 20 spam / 20 ham; đo trên CPU):

| Phương án | Precision | Recall | F1 (test) | F1 (SUB 40) | Train (s) | Tin/s (cả lô) | p50 ms/tin | p95 ms/tin | USD / 1k tin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **1. BoW + LogReg (tự code)** | 1.000 | 0.767 | **0.868** | 0.788 | 2.594 | 40,965.8 | 0.043 | **0.079** | 0.000 |
| **2. BoW + MLP (deep learning)** | 0.968 | 0.893 | **0.929** | 0.919 | 4.040 | 47,792.9 | 0.296 | **0.343** | 0.000 |
| **3a. Encoder (MiniLM) + kNN** | 0.878 | 0.981 | **0.927** | 1.000 | 33.573 | 32.7 | 44.353 | **55.078** | 0.000 |
| **3b. Encoder (MiniLM) + LogReg** | 0.942 | 0.951 | **0.947** | 0.974 | 33.915 | 63.9 | 20.201 | **26.150** | 0.000 |
| **3c. Encoder (MiniLM) + MLP head** | 0.951 | 0.942 | **0.946** | 0.947 | 35.495 | 72.3 | 18.367 | **24.703** | 0.000 |
| **4a–4c. LLM (`gpt-6-luna` / `gpt-6.1-terra`)** *(ước tính tham chiếu; chạy local không gắn API key trả phí, đã đạt kiểm tra tự động 4.1–4.4)* | — | — | — | ~0.95 | 0.000 | ~1–3 | >200 | **>300** | ~0.011 (zero-shot luna) |

## 3. Dự đoán và thực tế

Ba dự đoán `predict(...)` có điểm khác biệt đáng chú ý nhất so với kết quả thực nghiệm:

1. **`dl_relu` và `dl_curve` (Vai trò của hàm kích hoạt ReLU so với thuật toán tối ưu):**
   - **Dự đoán:** Mạng `BoW + MLP` chỉ nhỉnh hơn `BoW + LogReg` rất nhẹ và khi bỏ `ReLU` điểm F1 sẽ sụt về gần mức của `LogReg` ở Phần 1 (`0.868`).
   - **Thực tế:** `BoW + MLP` tăng mạnh lên `F1 = 0.929` so với `0.868` của Phần 1, nhưng khi xóa `nn.ReLU()` ở Thí nghiệm 2.B thì mạng 2 lớp tuyến tính liên tiếp vẫn giữ nguyên `F1 = 0.929`.
   - **Vì sao lệch:** Trên biểu diễn Bag-of-Words thưa 2.000 chiều, ranh giới phân chia tuyến tính vốn đã đủ tốt; mức chênh lệch từ `0.868` lên `0.929` không đến từ tính phi tuyến của `ReLU` mà đến từ bộ tối ưu Adam theo mini-batch hội tụ hiệu quả hơn trên các từ ít xuất hiện so với Gradient Descent toàn tập 300 vòng lặp ở Phần 1.

2. **`emb_tricky` (Thí nghiệm 3.B trên 6 câu ngụy trang):**
   - **Dự đoán:** Mô hình `3b. Encoder + LogReg` sẽ thắng cách biệt `BoW + LogReg` trên cả 6 câu ngụy trang.
   - **Thực tế:** Hai mô hình hòa nhau `5/6` nhưng đoán sai ở hai câu hoàn toàn khác nhau: `BoW` bỏ lọt câu lừa đảo thông báo điểm thưởng tài khoản sắp hết hạn vào tối nay vì câu này né sạch các từ khóa quen thuộc như `free`, `win`, `call`; ngược lại `Encoder + LogReg` lại bỏ lọt câu thông báo thuê bao được chọn nhận kỳ nghỉ miễn phí kèm yêu cầu nhắn xác nhận, trong khi `BoW` bắt được nhờ trọng số dương của cụm từ `mobile`, `reply`, `your`.
   - **Vì sao lệch:** Tập 6 câu quá nhỏ và câu chứa thuật ngữ giao thức tin nhắn (`mobile`, `reply`) vẫn giúp `BoW` ghi điểm; chỉ khi đo trên toàn tập test 775 tin và bộ 50 case khó, lợi thế hiểu ngữ cảnh của `Encoder` mới thể hiện rõ rệt (`0.947` so với `0.868` trên tập test; `0.640` so với `0.378` trên 50 case khó).

3. **`tradeoff` (Đường Pareto trên `SUB 40` so với toàn tập `test`):**
   - **Dự đoán:** Phương án `3a. Encoder + kNN` sẽ bị `3b` và `3c` lấn át hoàn toàn trên biểu đồ đánh đổi.
   - **Thực tế:** Khi vẽ theo cột lỗi trên `SUB 40`, phương án `3a` đạt `F1_SUB = 1.000` (tỷ lệ lỗi `0%`) nên vẫn nằm trên đường Pareto dù vi phạm ngưỡng SLA (`p95 = 55.08 ms`). Tuy nhiên khi xét trên toàn tập test 775 tin, `3a` chỉ đạt `F1 = 0.927`, vừa thua `3b` (`0.947`) vừa chậm hơn gấp đôi (`55.08 ms` so với `26.15 ms`), nghĩa là thực chất bị `3b` và `3c` lấn át.
   - **Vì sao lệch:** Trên tập nhỏ 40 tin, chỉ cần chênh lệch 1 tin nhắn đã làm điểm F1 dịch chuyển khoảng 2,5 điểm phần trăm, tạo ra nhiễu thứ hạng.

## 4. Phân tích 50 case khó

- **Mức độ sụt giảm:** Mọi mô hình đều giảm mạnh hiệu năng khi bước ra khỏi phân phối quen thuộc của kho dữ liệu SMS năm 2011:
  - **`2. BoW + MLP` tụt mạnh nhất (`-0.443` điểm F1:** từ `0.919` trên SUB xuống **`0.476`**, độ chính xác `56%`), kế đến là **`1. BoW + LogReg` (`-0.410`:** từ `0.788` xuống **`0.378`**, độ chính xác `54%`).
  - Nhóm Embedding bền vững hơn rõ rệt: `3a. Emb + kNN` giữ được **`F1 = 0.759`** (`72%`), `3c. Emb + MLP head` đạt **`0.653`** (`66%`), và `3b. Emb + LogReg` đạt **`0.640`** (`64%`).
- **Nhóm case khó nhất:**
  - Với **BoW**: Nhóm né từ khóa (`obfuscation`, trong đó `BoW + MLP` đúng `0%`, `BoW + LogReg` đúng `17%`) và nhóm thao túng tâm lý (`manipulation`, `BoW + LogReg` chỉ đúng `12%`).
  - Với **Embedding**: Nhóm tin rác viết giọng tự nhiên (`natural_spam`, `3b` và `3c` chỉ đúng `30%`) và nhóm kể chuyện cảnh báo tin rác (`talks_about_spam`, `3a` đúng `25%`, `3b` và `3c` đúng `50%`, thấp hơn mức `100%` của `BoW`).
- **Hai case cụ thể và cơ chế gây sai:**
  1. **Case `#31` (`obfuscation`, nhãn thật `spam`):** Nội dung thông báo nhận thưởng tiền mặt nhưng xen dấu chấm vào giữa từng chữ cái như `s.e.l.e.c.t.e.d`, `c.a.s.h`, `r.e.w.a.r.d`, `c.a.l.l` kèm đầu số tính phí cao. Cả hai mô hình `BoW` lẫn `3c. Emb + MLP head` đều đoán nhầm thành `ham`. Cơ chế: hàm tách từ theo biểu thức chính quy biến `c.a.s.h` thành 4 chữ cái đơn lẻ `c`, `a`, `s`, `h` nên không khớp từ `cash` trong từ điển; đồng thời bộ tách từ con của MiniLM cũng bị cắt vụn thành các ký tự rời làm nhạt tín hiệu ngữ nghĩa nhận thưởng.
  2. **Case `#47` (`talks_about_spam`, nhãn thật `ham`):** Người gửi kể lại với giọng châm biếm rằng vừa nhận được tin báo trúng 1.000 bảng Anh và yêu cầu gọi vào đầu số lạ rồi hỏi sao vẫn có người tin. Mô hình `3c. Emb + MLP head` đoán nhầm thành `spam` trong khi `BoW` đoán đúng `ham`. Cơ chế: Encoder gom trung bình ngữ nghĩa toàn câu (mean pooling), trong đó phần nội dung trích dẫn trúng thưởng chiếm trọng tâm chủ đề quá mạnh và lấn át sắc thái bình luận ở đầu và cuối câu; trái lại `BoW` có các đại từ giao tiếp cá nhân (`i`, `that`, `got`) mang trọng số âm đủ lớn để giữ tổng điểm ở phía `ham`.
- **Nhận xét thêm 6 case tự thiết kế (`MY_CASES` `#100..#105`):** Cả hai mô hình `BoW` đều bị lừa ở `#101` (mời làm việc nhập liệu từ xa qua Telegram), `#102` (giả danh con bị khóa ứng dụng ngân hàng ở trạm xăng nhờ chuyển gấp 85 bảng) và `#103` (dùng gạch dưới và số thay chữ trong từ `claim`, `voucher`), trong khi cả 3 mô hình Embedding bắt đúng toàn bộ 3 case này. Ngược lại, cả 3 mô hình Embedding và `1. BoW + LogReg` bị đánh lừa ở `#100` (bình chọn bánh nội bộ văn phòng có các chữ `URGENT`, `FREE`, `Winner`) và `#105` (kể chuyện ngăn bà ngoại nhắn tin nhận iPhone miễn phí), riêng `2. BoW + MLP` giữ được dự đoán đúng ở hai case này.

## 5. Thử thách cá nhân

- **Đề theo `SEED = 5095` (`5095 % 6 = 1` → Đề số 2):**
  Giảm kích thước từ điển `vocab` xuống 100 từ và tăng lên 5.000 từ thông qua `build_vocab(..., max_size)`, huấn luyện lại Logistic Regression, báo cáo `F1 test` và giải thích xu hướng.
- **Cách làm:** Duyệt `max_size` lần lượt qua ba mốc `100`, `2000`, `5000`, xây lại từ điển trên `texts_tr`, vector hóa `texts_tr` và `texts_te`, huấn luyện lại bằng `train_logreg(..., lr=0.5, epochs=300)` và ghi nhận các chỉ số đánh giá cùng thời gian chạy.
- **Kết quả thực nghiệm:**

| `max_size` | Số từ thực tế | Train F1 | Test Precision | Test Recall | Test F1 | Train (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **100** | 100 | 0.829 | 0.971 | 0.641 | **0.772** | 0.17s |
| **2000** | 2000 | 0.911 | 1.000 | 0.767 | **0.868** | 2.69s |
| **5000** | 5000 | 0.916 | 1.000 | 0.767 | **0.868** | 6.55s |

- **Giải thích xu hướng:**
  - Khi giới hạn từ điển ở **100 từ**, phần lớn vị trí bị chiếm bởi các hư từ tiếng Anh xuất hiện dày ở cả hai nhãn (`i`, `to`, `you`, `a`, `the`), làm rớt mất nhiều từ khóa đặc trưng của tin rác (`ringtone`, `prize`, `urgent`, `guaranteed`, `award`). Mô hình thiếu đặc trưng phân loại (underfitting) nên `test recall` sụt từ `0.767` xuống `0.641` và `test F1` giảm còn `0.772`.
  - Khi tăng từ **2.000 lên 5.000 từ**, nhóm từ xếp hạng 2.001 đến 5.000 là các từ hiếm chỉ xuất hiện 1–2 lần trong tập train (mã thưởng cá biệt, từ sai chính tả, tên riêng). Nhóm từ này chỉ giúp khớp thêm vài mẫu trên tập huấn luyện (`train F1` nhích nhẹ từ `0.911` lên `0.916`) nhưng hầu như không gặp lại trên tập test nên **`test F1` đi ngang ở `0.868`**, trong khi thời gian huấn luyện tăng gấp **2,4 lần** (`2.69s` lên `6.55s`). Vì vậy mốc `2000` từ là điểm cân bằng hợp lý.

## 6. Phương án cho sản phẩm thật

Với quy mô lọc **10 triệu SMS/ngày** (~116 tin/giây trung bình, giờ cao điểm có thể gấp 3–5 lần) và yêu cầu cứng **`p95 < 50 ms/tin`**:

- **Nếu chọn một mô hình duy nhất:** Chọn **`3b. Encoder (MiniLM) + LogReg`**.
  - *Dẫn chứng số liệu:* Đạt điểm cao nhất trên tập test (`F1 = 0.947`, `Precision = 0.942`, `Recall = 0.951`), chống chịu né từ khóa tốt hơn hẳn BoW (`83%` so với `17%`), không bị ảnh hưởng bởi Prompt Injection (`100%`), chi phí gọi API bằng `0 USD`, và độ trễ đơn chiếc **`p95 = 26.15 ms < 50 ms`** (triển khai 3–4 tiến trình CPU song song hoặc 1 GPU nhỏ là đáp ứng dư thông lượng giờ cao điểm).
  - *Lý do loại phương án khác:* `3a. kNN` vi phạm ngưỡng độ trễ (`p95 = 55.08 ms > 50 ms`) và chi phí quét tăng tuyến tính theo kích thước kho mẫu; gọi trực tiếp `LLM` qua mạng vừa vượt xa 50 ms vừa tiêu tốn khoảng **`3.150 USD/tháng`** với `gpt-6-luna` (`10.50 USD / 1 triệu tin` × 300 triệu tin/tháng).
- **Kiến trúc lai đề xuất (Cascaded Router đa tầng) và đánh đổi chấp nhận:**
  - **Tầng 1 (`2. BoW + MLP`, `p95 = 0.34 ms`, `47.793 tin/s`):** Chuẩn hóa ký tự thay thế cơ bản rồi phân loại tức thì khoảng 80% lưu lượng tin nhắn có xác suất rất rõ ràng ($p < 0.05$ hoặc $p > 0.98$).
  - **Tầng 2 (`3b. Encoder + LogReg`, `p95 = 26.15 ms`):** Chỉ xử lý khoảng 20% tin có xác suất lưỡng lự $p \in [0.05, 0.98]$, tiết kiệm 80% tải tính toán cho Encoder mà vẫn giữ `F1` xấp xỉ `0.95` và tổng độ trễ `p95 < 27 ms`.
  - **Tầng 3 (`LLM` kèm prompt phòng thủ thẻ `<sms>` chạy bất đồng bộ ngoài luồng chính):** Mỗi ngày lấy mẫu khoảng 0,5% ca khó nhất ($p \in [0.35, 0.65]$) để phát hiện sớm các kịch bản mạo danh tự nhiên mới và bổ sung nhãn huấn luyện lại Tầng 1–2 định kỳ, tốn chỉ khoảng `15–30 USD/tháng`.
  - *Đánh đổi chấp nhận:* Chấp nhận vận hành hệ thống nhiều tầng phức tạp hơn một mô hình đơn lẻ và chấp nhận một tỷ lệ nhỏ mẫu lừa đảo hoàn toàn mới có thể lọt qua ở thời điểm đầu trước khi luồng học chủ động kịp cập nhật, đổi lại giữ vững cam kết `p95 < 50 ms`, bảo vệ tin nhắn giao dịch hợp lệ (`Precision > 94%`) và kiểm soát chi phí sát mức 0.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì (prompt) | AI trả lời / gợi ý gì | Đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 0–1 | Hỏi giải thích khái niệm data leakage, dấu của gradient hàm Binary Cross-Entropy và cách tránh tràn số học khi tính hàm mũ trong numpy | Giải thích vì sao dữ liệu trùng giữa train và test làm sai lệch đánh giá, ý nghĩa đại lượng `p - y`, và gợi ý giới hạn miền đầu vào bằng `np.clip` trước `np.exp` | Tự viết các hàm `1.1` đến `1.4`, thêm kiểm tra mẫu số bằng 0 trong `metrics`, chạy bộ kiểm tra tự động đạt và tự đọc 10 ca đoán sai trên `SEED = 5095` |
| Phần 2–3 | Nhờ đọc traceback lỗi `NameError: name 'predict_mlp' is not defined` khi chạy ô MLP và hỏi ý tưởng đại số tuyến tính để tính cosine hàng loạt không dùng vòng lặp | Chỉ ra việc vô tình xóa mất hàm `predict_mlp` có sẵn ở cuối ô code trước đó và giải thích nguyên lý chia từng dòng cho chuẩn L2 để đưa cosine về phép nhân ma trận | Khôi phục hàm `predict_mlp` có sẵn, tự hoàn thiện `build_mlp`, `train_mlp`, `cosine_similarity`, `top_k_similar`, `knn_predict` và kiểm chứng toàn bộ kiểm tra `2.1` đến `3.3` đạt  |
| Phần 4 & Side Lab | Nhờ gợi ý khung hàm `call_openai`, `build_classification_prompt`, `parse_label`, `llm_classify_batch` và side lab theo đúng chữ ký hàm yêu cầu | Gợi ý cách truyền `**provider_kwargs(max_tokens)`, cấu trúc few-shot prompt và lưu ý thứ tự kiểm tra chuỗi `not spam` trước `spam` trong `parse_label` | Tự rà soát và bổ sung xử lý chuỗi rỗng, ký tự dấu câu, tiền tố `Label:`, chạy toàn bộ kiểm tra tự động `4.1–4.4` cùng 27 unit test của side lab đạt  |

**Điều hiểu ra hôm nay:**
Trước buổi thực hành, em từng nghĩ mô hình thuộc thế hệ mới hơn và nhiều tham số hơn (mạng MLP sâu hơn Logistic Regression, hay Embedding và LLM so với Bag-of-Words) sẽ mặc nhiên vượt trội ở mọi tình huống. Sau khi trực tiếp đo đạc từng con số trên `SEED = 5095` và kiểm thử trên 50 case đối kháng, em hiểu rằng chất lượng biểu diễn đầu vào quan trọng hơn nhiều so với việc chồng thêm tầng ẩn (bỏ `ReLU` trên BoW không làm giảm điểm `0.929`, trong khi chuyển sang biểu diễn ngữ nghĩa MiniLM kéo tỷ lệ bắt đúng nhóm né từ khóa từ `0–17%` lên `83%`). Đồng thời, mỗi kiến trúc đều có góc khuất riêng như việc Embedding lại dễ bắt nhầm tin kể chuyện cảnh báo lừa đảo hơn cả BoW, nên bài toán kỹ thuật thực tế luôn là sự phối hợp đa tầng dựa trên số đo cụ thể giữa độ chính xác, độ trễ `p95` và chi phí vận hành.
