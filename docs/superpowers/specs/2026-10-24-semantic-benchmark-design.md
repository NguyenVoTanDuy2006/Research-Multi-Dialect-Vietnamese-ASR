# Thiết kế: Semantic metrics trong ASR benchmark

**Trạng thái:** Đã được người dùng duyệt; chưa triển khai code.

## Mục tiêu

Benchmark hiện chạy ASR cho từng model, ghi predictions và tính WER/CER. Tích hợp thêm SemSim-E5 và BERTScore để mỗi lần chạy benchmark có thể đánh giá cả độ khớp chuỗi lẫn mức tương đồng ngữ nghĩa giữa transcript tham chiếu và dự đoán.

## Phương án

1. **Chấm ngữ nghĩa trong vòng lặp inference:** không chọn vì semantic encoders làm tăng áp lực VRAM khi model ASR còn trên GPU và có thể chấm lặp các cặp transcript giống nhau.
2. **Chấm sau ASR trong cùng workflow:** chọn. Giữ nguyên inference và WER/CER; sau đó chấm các cặp thành công theo batch bằng SemSim-E5 và BERTScore, giải phóng từng encoder trước khi nạp encoder kế tiếp.
3. **Giữ semantic scoring ở notebook hậu xử lý riêng:** không chọn vì người dùng muốn benchmark tạo luôn bộ kết quả có semantic metrics.

## Thiết kế đề xuất

- Dùng lại `SemSimE5` và `BERTScoreMetric` hiện có; không viết lại công thức metric.
- Thêm cấu hình semantic có thể bật/tắt, chọn metric, thiết bị, batch size và chính sách transcript dài. Benchmark notebook bật cả SemSim-E5 lẫn BERTScore; API workflow tổng quát giữ trạng thái tắt nếu không được cấu hình để tránh phụ thuộc/tải model ngoài ý muốn.
- Thêm bước semantic sau khi `BenchmarkRunner` đã hoàn tất và prediction artifacts đã được lưu. Chỉ chấm kết quả inference không có lỗi; prediction rỗng nhận score theo quy ước hiện có của metric. Bỏ các cặp `(reference, prediction)` trùng chính xác khi encode rồi ánh xạ score về từng model/sample.
- Chạy hai encoder tuần tự, báo tiến độ theo batch, kiểm tra smoke sample trước lượt chấm toàn bộ và ghi metadata về model revision, batch size, token limit, long-text policy, thời gian chạy và số cặp duy nhất.
- Lưu score cache bền vững sau mỗi batch đã hoàn tất. Cache key kết hợp metric, cấu hình/model revision thực tế và hash của cặp transcript; khi resume chỉ chấm các cặp còn thiếu. Nếu cấu hình hoặc model revision đổi, score cũ không được tái sử dụng. Dòng JSONL lỗi ở cuối file do process bị dừng giữa lúc ghi sẽ được bỏ qua an toàn.
- Giữ nguyên các file WER/CER hiện tại. Tạo các file riêng: `semantic_predictions.csv`, `semantic_metrics_overall.csv`, `semantic_metrics_by_region.csv`, `semantic_metrics_by_province.csv` và `semantic_metrics_config.json`. Score theo mẫu là SemSim-E5/distance và BERTScore precision/recall/F1; summary dùng mean/median/p05 theo mẫu, cùng số mẫu được chấm.
- `bert-score` là dependency tùy chọn, còn transformers và E5 weights cần có thể tải từ Hugging Face. Notebook sẽ kiểm tra/cài dependency và báo rõ yêu cầu Internet/token trước khi benchmark tốn thời gian.

## Phạm vi không bao gồm

- Thay đổi mô hình ASR, cách decode, chuẩn hóa transcript, hay định nghĩa WER/CER.
- Xóa hoặc hợp nhất notebook `metrics_predictions.py`; notebook đó vẫn hữu ích để tính metrics trên file prediction đã có.
- Thay đổi prediction schema hiện tại hoặc trộn semantic scores vào bảng WER/CER cũ.

## Kiểm thử dự kiến

- Unit test với metric test doubles: dedup cặp text, ánh xạ score về đúng model/sample, xử lý prediction lỗi/rỗng, và tạo summary theo model/miền/tỉnh.
- Unit test mô phỏng dừng sau một batch rồi resume: score đã cache không bị chấm lại, cặp thiếu được chấm tiếp; fingerprint đổi không sử dụng cache cũ; trailing JSONL record dở dang được bỏ qua.
- Kiểm tra tương thích: artifacts WER/CER cũ vẫn được tạo như trước; semantic config tắt không tải metric model/dependency.
- Smoke test trên một batch thật trước khi chạy toàn bộ; xác minh score hữu hạn, số hàng và metadata của output.

## Quyết định cho bản triển khai

- Cấu hình trong notebook sẽ bật semantic metrics mặc định; API benchmark không có cấu hình semantic sẽ tiếp tục chỉ chạy WER/CER.
- Semantic score cache sẽ được lưu theo batch và tái sử dụng khi resume; prediction cache ASR hoạt động độc lập.
