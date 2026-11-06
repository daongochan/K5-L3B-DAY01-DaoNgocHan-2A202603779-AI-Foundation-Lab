# Ngày 1 — Bài Tập & Phản Ánh (side lab TUỲ CHỌN)

> Side lab này **không** thuộc điểm chính của Day 01 (xem [RUBRIC.md](../RUBRIC.md) mục Bonus). Model: `gpt-6.1-terra` (lớn) và `gpt-6-luna` (nhỏ). `gpt-4o` / `gpt-4o-mini` bị gỡ khỏi API ngày 14/01/2027 ([nguồn](https://developers.openai.com/api/docs/deprecations)). Tên khoá `gpt4o_*` / `mini_*` trong kết quả là hợp đồng của test, giữ nguyên để tương thích: đọc là "model lớn" / "model nhỏ".
## Nền Tảng LLM API | Phiếu Thực Hành

**Thời lượng:** 1:30 giờ  
**Cấu trúc:** Lập trình cốt lõi (60 phút) → Bài tập mở rộng (30 phút)

---

## Phần 1 — Lập Trình Cốt Lõi (0:00–1:00)

Chạy các ví dụ trong Google Colab tại: https://colab.research.google.com/drive/172zCiXpLr1FEXMRCAbmZoqTrKiSkUERm?usp=sharing

Copy `src/optional_llm_api/template.py` thành `src/optional_llm_api/solution/solution.py`, triển khai mọi TODO, rồi chạy từ gốc repo: `pytest tests/test_solution.py -v`. Chưa có `solution/solution.py` thì test tự bỏ qua.

**Điểm kiểm tra:** Sau khi hoàn thành 4 nhiệm vụ, chạy:
```bash
python src/optional_llm_api/solution/solution.py
```
Bạn sẽ thấy output so sánh phản hồi của `gpt-6.1-terra` và `gpt-6-luna`.

---

## Phần 2 — Bài Tập Mở Rộng (1:00–1:30)

### Bài tập 2.1 — Độ Nhạy Của Temperature
Gọi `call_openai` với các giá trị temperature 0.0, 0.5, 1.0 và 1.5 sử dụng prompt **"Hãy kể cho tôi một sự thật thú vị về Việt Nam."** (nhớ gửi `reasoning_effort="none"`: model GPT-6 chỉ nhận `temperature` khi tắt suy luận)

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở `temperature = 0.0`, mô hình luôn trả về cùng một sự thật quen thuộc (ví dụ về hang Sơn Đoòng hoặc sản lượng cà phê) với cách diễn đạt chuẩn mực và lặp lại gần như y hệt giữa các lần gọi. Khi tăng dần lên `0.5` và `1.0`, câu trả lời đa dạng hơn về chủ đề và cách dùng từ sinh động hơn nhưng vẫn mạch lạc. Đến `1.5`, câu chữ trở nên phóng khoáng, đôi khi lan man hoặc pha trộn chi tiết kém chính xác hơn do xác suất chọn các token ít phổ biến tăng cao.

**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ đặt `temperature` thấp trong khoảng **0.0 – 0.2** (ưu tiên `0.0` hoặc `0.1`). Chatbot hỗ trợ khách hàng cần cung cấp thông tin chính sách, giá cả, quy trình đổi trả một cách **nhất quán, chính xác và bám sát tài liệu**, tránh việc tự suy diễn hoặc bịa thêm thông tin sai lệch (hallucination).

---

### Bài tập 2.2 — Đánh Đổi Chi Phí
Xem xét kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người thực hiện 3 lần gọi API, mỗi lần trung bình ~350 token.

**Ước tính xem `gpt-6.1-terra` đắt hơn `gpt-6-luna` bao nhiêu lần cho workload này** (giá 11/2026: terra $1.00 / $5.00, luna $0.10 / $0.50 mỗi 1M token input / output):
> Tổng khối lượng mỗi ngày là $10.000 \times 3 \times 350 = 10.500.000$ token (10,5M token/ngày). Vì đơn giá của `gpt-6.1-terra` (1,00 USD / 1M input và 5,00 USD / 1M output) cao gấp đúng **10 lần** `gpt-6-luna` (0,10 USD / 1M input và 0,50 USD / 1M output) ở cả chiều input lẫn output, nên bất kể tỷ lệ chia input/output trong 350 token là bao nhiêu, `gpt-6.1-terra` luôn đắt gấp **10 lần** `gpt-6-luna` (ví dụ nếu 250 input + 100 output mỗi lần gọi: `luna` tốn 0,75 USD + 1,50 USD = **2,25 USD/ngày**, còn `terra` tốn 7,50 USD + 15,00 USD = **22,50 USD/ngày**).

**Mô tả một trường hợp mà chi phí cao hơn của `gpt-6.1-terra` là xứng đáng, và một trường hợp `gpt-6-luna` là lựa chọn tốt hơn:**
> - **Khi `gpt-6.1-terra` xứng đáng:** Phân tích hợp đồng pháp lý phức tạp, chẩn đoán lỗi kiến trúc phần mềm nhiều file, hoặc xử lý khiếu nại tài chính nhạy cảm — nơi một sai sót suy luận gây thiệt hại lớn hơn nhiều so với vài USD tiền API.
> - **Khi `gpt-6-luna` tốt hơn:** Phân loại nhãn hàng loạt (như lọc SMS spam/ham, gắn tag ticket hỗ trợ), tóm tắt ngắn hoặc trích xuất trường dữ liệu đơn giản ở quy mô hàng triệu lượt gọi/ngày cần độ trễ thấp và chi phí tối ưu.

---

### Bài tập 2.3 — Trải Nghiệm Người Dùng với Streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì non-streaming lại phù hợp hơn?** (1 đoạn văn)
> **Streaming** quan trọng nhất trong các giao diện tương tác trực tiếp với con người (chatbot, trợ lý viết lách, giải thích code dài) vì người dùng nhìn thấy chữ đầu tiên hiện ra chỉ sau vài trăm mili-giây (*Time To First Token* thấp), tạo cảm giác phản hồi tức thì thay vì phải nhìn màn hình chờ 5–10 giây cho đến khi toàn bộ câu trả lời sinh xong. Ngược lại, **non-streaming** phù hợp hơn cho các tác vụ xử lý ngầm hoặc pipeline tự động giữa máy với máy (như phân loại `spam`/`ham` bằng 1 từ, trích xuất JSON có cấu trúc để parse ngay, gọi hàm batch job), nơi chương trình phía sau cần nhận trọn vẹn chuỗi kết quả hợp lệ một lần để kiểm tra và xử lý tiếp mà không cần hiển thị từng mảnh token trung gian.


## Danh Sách Kiểm Tra Nộp Bài
- [x] Tất cả tests pass: `pytest tests/test_solution.py -v`
- [x] `call_openai` đã triển khai và kiểm thử
- [x] `call_openai_mini` đã triển khai và kiểm thử
- [x] `compare_models` đã triển khai và kiểm thử
- [x] `streaming_chatbot` đã triển khai và kiểm thử
- [x] `retry_with_backoff` đã triển khai và kiểm thử
- [x] `batch_compare` đã triển khai và kiểm thử
- [x] `format_comparison_table` đã triển khai và kiểm thử
- [x] `exercises.md` đã điền đầy đủ
- [x] Sao chép bài làm vào folder `solution` và đặt tên theo quy định 
