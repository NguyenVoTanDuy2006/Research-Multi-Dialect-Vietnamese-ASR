# Kiểm tra dataset ViMD trong cache Hugging Face

**Ngày kiểm tra:** 12/09/2026  
**Mục đích giả định:** dùng ViMD để fine-tune và đánh giá ASR tiếng Việt đa vùng miền.  
**Đơn vị dữ liệu:** một dòng tương ứng một đoạn audio và transcript.

## Kết luận

Dataset **đã tải đầy đủ** và phần cache đã xử lý có thể mở được: đủ 18.949 dòng và 108 Arrow shard, khớp hoàn toàn với `dataset_info.json`. Không có shard tải dở, liên kết cache hỏng, trường metadata bắt buộc bị thiếu, filename trùng hay audio payload rỗng.

Tuy nhiên, **không nên dùng nguyên split để công bố kết quả cuối** trước khi xử lý hai `speakerID` xuất hiện đồng thời trong `valid` và `test`. Ngoài ra, môi trường Python hiện chưa có `torchcodec`/PyTorch nên truy cập `row["audio"]` theo cơ chế decode mặc định của `datasets` sẽ lỗi dù dữ liệu không hỏng.

## Tổng quan dữ liệu

| Split | Số mẫu | Speaker ID phân biệt | Số tỉnh/thành | Số vùng |
|---|---:|---:|---:|---:|
| train | 15.023 | 10.291 | 63 | 3 |
| valid | 1.900 | 1.320 | 63 | 3 |
| test | 2.026 | 1.344 | 63 | 3 |
| **Tổng** | **18.949** | **12.953 unique toàn cục** | **63** | **3** |

Tổng thời lượng tính trực tiếp từ header WAV của **toàn bộ 18.949 audio** là **369.228,489 giây = 102 giờ 33 phút 48,49 giây (102,563469 giờ)**:

| Split | Thời lượng |
|---|---:|
| train | 81,429121 giờ |
| valid | 10,261993 giờ |
| test | 10,872355 giờ |
| **Tổng** | **102,563469 giờ** |

Phép tính đọc `fmt/data` header của mọi payload, không copy hoặc decode toàn bộ waveform; cả 18.949 header đều hợp lệ. Kết quả này khớp với thống kê 102,56 giờ trong dataset card/paper sau khi làm tròn.

Phân bố vùng theo số mẫu:

| Vùng | Số mẫu | Tỷ lệ |
|---|---:|---:|
| North | 7.448 | 39,31% |
| Central | 5.930 | 31,29% |
| South | 5.571 | 29,40% |

Mỗi tỉnh có từ 263 đến 363 mẫu, trung vị 303. Phân bố tỉnh tương đối cân bằng; vùng Bắc có tỷ trọng lớn hơn hai vùng còn lại.

Trường `gender` có 5.345 dòng mang giá trị `0` và 13.604 dòng mang giá trị `1`, tương ứng 28,21% và 71,79%. Theo README của dataset, `0 = female`, `1 = male`; vì vậy dữ liệu nghiêng về nam khoảng 2,55 lần.

## Các kiểm tra đã chạy

- Mở toàn bộ 108 Arrow shard và đối chiếu số shard/số dòng với `dataset_info.json`.
- Kiểm tra null và chuỗi rỗng cho `region`, `province_code`, `province_name`, `filename`, `text`, `speakerID`, `gender`.
- Kiểm tra audio struct, audio bytes và audio path không null trên toàn bộ 18.949 dòng.
- Đọc RIFF/WAV header của một mẫu trên mỗi shard: 108/108 mẫu đọc thành công.
- Kiểm tra filename duy nhất, tiền tố filename khớp mã tỉnh, ánh xạ mã–tên tỉnh, miền và giá trị gender.
- Kiểm tra `speakerID` trùng giữa các split và metadata của cùng speaker.
- Kiểm tra transcript trùng; băm SHA-256 audio của 25 dòng đáng nghi gồm speaker overlap và transcript trùng xuyên split.
- Kiểm tra snapshot Hugging Face: 131 symlink hợp lệ, không có broken symlink và không có `.incomplete`/`.part`.

## Phát hiện

### 1. Hai speaker ID bị rò rỉ giữa valid và test — mức High, độ tin cậy cao

Không có overlap train–valid hoặc train–test, nhưng có hai ID ở cả valid và test:

| Speaker ID | Valid | Test | Bất nhất đi kèm |
|---|---|---|---|
| `spk_73_0186` | `73_0309.wav` | `73_0307.wav`, `73_0308.wav` | gender 1 ở valid, 0 ở test |
| `spk_76_0219` | `76_0295.wav` | `76_0294.wav` | gender 0 ở valid, 1 ở test |

Tên file liền nhau, cùng tỉnh và cùng speaker ID làm tăng khả năng đây là cùng nguồn/người nói; chưa nghe đối chiếu giọng nên không khẳng định tuyệt đối danh tính người nói.

**Rủi ro:** lựa chọn checkpoint/hyperparameter trên valid đã nghe cùng speaker với test, khiến test không còn hoàn toàn speaker-independent.

**Cách xử lý nhỏ nhất:** giữ các đoạn test, loại hai dòng valid `73_0309.wav` và `76_0295.wav` khỏi valid dùng cho tuning; ghi rõ filtered-valid có 1.898 dòng. Nếu cần so sánh trực tiếp với baseline paper, báo cả kết quả split gốc và split đã lọc.

### 2. Môi trường chưa giải mã được audio qua `datasets` — mức High đối với việc huấn luyện

Project hiện có `datasets==5.0.1` và `pyarrow==25.0.1`, nhưng không có `torch`, `torchcodec`, `torchaudio` hoặc `soundfile`. Vì vậy truy cập trực tiếp:

```python
dataset[0]["audio"]
```

phát sinh `ImportError: To support decoding audio data, please install 'torchcodec'.`

Đây là lỗi dependency, **không phải dataset bị hỏng**. Trước khi train, cần cài PyTorch phù hợp CPU/CUDA và TorchCodec tương thích, rồi kiểm tra decode/resample bằng một batch nhỏ.

### 3. Audio không đồng nhất định dạng — mức Medium, độ tin cậy cao trên mẫu shard

Trong 108 mẫu, phát hiện sample rate 32 kHz, 44,1 kHz và 48 kHz; có cả mono/stereo; PCM 16/32-bit và IEEE-float 32-bit. Thời lượng mẫu từ 4,908 đến 29,932 giây, trung vị 19,820 giây.

**Rủi ro:** batch có số kênh/sample rate khác nhau; pipeline feature extraction có thể lỗi hoặc xử lý không nhất quán.

**Khắc phục:** decode an toàn, downmix mono và resample 16 kHz trong preprocessing; không ghi đè audio gốc. Thêm test xác nhận mọi waveform sau chuẩn hóa có shape một chiều, 16 kHz, finite và thời lượng hợp lệ.

### 4. Có 17 speaker ID mang hai nhãn gender — mức Medium cho phân tích fairness

Toàn bộ 17 bất nhất speaker đều nằm ở trường gender; không có speaker thay đổi region hoặc province. Hai speaker overlap phía trên cũng thuộc nhóm này.

**Rủi ro:** sai số khi đánh giá WER theo giới; không ảnh hưởng trực tiếp đến loss ASR nếu gender không được dùng làm feature.

**Khắc phục:** không tự suy đoán gender theo đa số khi chưa nghe/đối chiếu nguồn. Gắn cờ 17 ID là `gender_unknown` trong báo cáo subgroup hoặc phân xử thủ công trước khi phân tích fairness.

### 5. Transcript trùng với tỷ lệ thấp — mức Low/Medium

- 117/18.949 dòng (0,617%) thuộc 57 chuỗi transcript xuất hiện nhiều lần.
- 10 chuỗi xuất hiện xuyên split, ảnh hưởng 20 dòng (0,106%).
- Không có SHA-256 audio giống hệt nhau trong 25 dòng đáng nghi đã kiểm tra.

Transcript giống nhau nhưng audio khác byte có thể là nhiều người đọc/nói cùng nội dung, bản mã hóa khác nhau hoặc dữ liệu lặp ở nguồn. Chưa đủ bằng chứng để gọi là audio leakage.

**Khắc phục:** nghe hoặc dùng acoustic fingerprint cho 20 dòng xuyên split trước khi khóa benchmark; không nên loại chỉ vì text giống nhau.

### 6. Schema thực tế lệch nhẹ so với README — mức Low

README mô tả các thuộc tính `set`, `length` và `province`; schema tải về dùng split ngoài bảng và các cột `province_code`, `province_name`, đồng thời không có `length`. `dataset_info.json` để trống trường license trong khi README front matter ghi `cc-by-nc-nd-4.0`.

**Khắc phục:** lấy schema runtime làm chuẩn kỹ thuật, tự tính duration từ audio header, và giữ README/dataset card làm nguồn kiểm tra giấy phép. Không suy luận rằng license trống nghĩa là không có hạn chế.

## Dung lượng và vị trí

Dataset được lưu ở cache mặc định, không nằm trong thư mục project:

```text
/home/fernir/.cache/huggingface/
```

`dataset_info.json` ghi khoảng 59,83 GB dữ liệu tải và 64,02 GB cache Arrow đã xử lý, tổng cộng 123,85 GB theo đơn vị thập phân (xấp xỉ 117 GiB trên `du`). Việc gần gấp đôi dung lượng là do Hugging Face giữ cả source Parquet và cache Arrow.

Không nên xóa một phần shard bằng tay. Nếu cần giải phóng dung lượng, phải quyết định giữ source hub hay processed cache và dùng công cụ quản lý cache của Hugging Face sau khi đã xác nhận pipeline có thể tải/tạo lại.

## Thứ tự xử lý đề xuất

1. Cài dependency audio tương thích và xác nhận decode được một batch chứa đủ PCM/IEEE-float, mono/stereo.
2. Tạo filtered-valid không chứa `73_0309.wav` và `76_0295.wav`.
3. Chuẩn hóa audio runtime về mono 16 kHz; giữ audio gốc trong cache.
4. Nghe/so fingerprint 20 dòng có transcript trùng xuyên split.
5. Gắn cờ 17 speaker có gender không nhất quán trước mọi phân tích theo giới.
6. Chạy audit này trong CI hoặc trước mỗi lần đổi phiên bản dataset.

## Phạm vi và giới hạn

- Metadata và null audio payload được kiểm tra trên toàn bộ 18.949 dòng.
- Thời lượng được tính trên toàn bộ 18.949 WAV header; không cần giải mã waveform.
- Kiểm tra định dạng/header mở rộng (channels, sample rate, format) đại diện trên 108 mẫu, mỗi Arrow shard một mẫu; phép tính duration vẫn đã đọc header `fmt/data` của toàn bộ 18.949 audio; chưa giải mã/nghe toàn bộ waveform.
- SHA-256 chỉ được tính cho 25 dòng đáng nghi, không phải toàn bộ 64 GB processed audio.
- Không có trường thời gian nên kiểm tra freshness/trend không áp dụng.
- Chưa nghe trực tiếp để xác nhận hai ID overlap thực sự là cùng người hay lỗi gán ID.

## Nguồn kiểm tra

- Script tải: `src/research/download.py`.
- Cache metadata: `dataset_info.json` trong processed ViMD cache, SHA-256 `90a3137a8a6c98b4445cec2ec8d7379097238bbcbb24b16e1c07781820a0a365`.
- Dataset card đã cache: snapshot `3a5b30157034e7eadd5c75fae1a820c6f9383398/README.md`.
- Toàn bộ 108 Arrow shard và snapshot Parquet trong cache Hugging Face.
- Audit tái lập: `src/research/audit_vimd.py`; kết quả máy đọc: `docs/vimd_dataset_audit.json`.
