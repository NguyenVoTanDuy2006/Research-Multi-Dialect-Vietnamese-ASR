# Review source code trước khi tổ chức lại dự án

Ngày kiểm tra: 09/10/2026. Đánh giá dựa trên source hiện có trong workspace, không giả định bản đã chạy trên Kaggle giống hệt bản local.

## Phạm vi và cách kiểm tra

- Rà soát 21 file Python tự viết, khoảng 10.091 dòng, trong `nb/`, `scripts/`, `demo/` và `norm_volumn.py`; một file trong số đó rỗng. Hai worker snapshot trong output được kiểm tra thêm, không tính vào 21 file này.
- Kiểm tra notebook `.ipynb` hiện có, hai worker được lưu cùng output và cấu hình dự án.
- Đọc các entrypoint CBA-Whisper và đường train/callback liên quan; không xem toàn bộ thư viện vendored `pyannote`, `faster_whisper`, `whisperx` là code tự viết, và không audit từng dòng của các thư viện đó.
- Kiểm tra cú pháp bằng AST sau khi thay dòng magic `%pip` bằng `pass` trong bộ nhớ; kiểm tra riêng worker nằm trong chuỗi. Đây không phải kiểm tra import hoặc chạy end-to-end.
- Chạy kiểm tra cô lập các hàm bằng dữ liệu giả lập; đọc header của 5 adapter SafeTensors để đối chiếu shape với `rank_pattern`.
- Local có Torch nhưng không có GPU khả dụng và chưa cài PEFT. Chưa chạy lại training, inference model, tensor round-trip hoặc render video trong lần review này.
- Không thay đổi source thực thi, dữ liệu hay checkpoint. File này là báo cáo review.

## Kết luận

Dự án đã có các thành phần phục vụ nghiên cứu: đọc ViMD, lấy mẫu, đóng gói, DI, ASR, AdaLoRA, benchmark và demo. Vấn đề lớn nhất là các thành phần đang nằm trong nhiều notebook có code trùng, trạng thái global và cấu hình riêng. Một số sửa lỗi chỉ xuất hiện trong một bản sao; một số kết quả cũng dùng định nghĩa metric khác nhau.

Nên tổ chức lại thành thư viện dùng chung, các lệnh chạy độc lập và notebook percent-format mỏng. Không cần viết lại thuật toán từ đầu. Trước khi dùng cho một thí nghiệm mới, cần xử lý lỗi Run All, tính đại diện của sample và khả năng truy xuất nguồn của benchmark.

## Phát hiện theo mức ưu tiên

### 1. P1 — Cả hai cấu hình fine-tuning thiếu `num_workers`

Vị trí: `nb/qwen3_asr_adalora_kaggle_t4x2.py:91`, `:412`; cấu hình tương tự trong `nb/fine-tuning.py`.

`NUM_WORKERS = 2` được khai báo nhưng không thêm vào `CONFIG`. Worker dùng `CFG["num_workers"]`, nên fresh run tới `make_loader()` sẽ gặp `KeyError: 'num_workers'`. Kiểm tra cô lập đã tái hiện lỗi này. Smoke gọi dataset trực tiếp, không qua DataLoader, nên có thể thành công mà full training vẫn lỗi.

Hướng sửa: dùng một schema cấu hình được xác thực trước khi launch; bổ sung khóa này và kiểm tra chính DataLoader trong smoke.

### 2. P1 — Cell tạo manifest dùng `io` trước khi import

Vị trí: `nb/qwen3_asr_adalora_kaggle_t4x2.py:171`; `import io` nằm ở dòng 246.

ViMD có audio bytes, nên `sf.info(io.BytesIO(raw))` được gọi trong cell tạo manifest trước cell import `io`. Fresh Run All gặp `NameError: name 'io' is not defined`. Đã tái hiện với namespace có `sf` nhưng chưa có `io`. Kernel từng chạy cell sau có thể che lỗi.

Hướng sửa: tập trung imports đầu script hoặc đóng chức năng đọc audio vào module độc lập.

### 3. P1 — `fine-tuning.py` vẫn còn lỗi `env` chưa được tạo

Vị trí: `nb/fine-tuning.py:892`, `:920`.

Toàn bộ cell smoke, kể cả lệnh tạo `env`, đã bị comment. Cell train/test vẫn truyền `env=env`. Không có assignment `env` đang hoạt động trong file. Bản `qwen3_asr_adalora_kaggle_t4x2.py` đã sửa độc lập ở các cell launch, nhưng bản sao này chưa sửa.

File còn gán token placeholder thay cho Secrets và báo đã đọc Secret dù thực tế không đọc. Placeholder không phải token hoạt động.

Hướng sửa: chỉ giữ một entrypoint train; dựng môi trường subprocess trong hàm launch chung; dùng env/Secrets thật, không gán placeholder vào request.

### 4. P1 — Hai script phân tích tham chiếu biến đã bị comment

- `nb/extract_vimd_by_region.py:489`: dùng `summary_province`, nhưng cell tạo bảng ở dòng 472 đã bị comment.
- `nb/benchmark_SR_models/analyze_quick_predictions_metrics.py:117`: merge `success_model_counts`, nhưng assignment ở dòng 114 đã bị comment.

Cả hai `NameError` đã được tái hiện bằng chính biểu thức trong source. Điều này chặn biểu đồ/manifest cuối của extraction và phần xuất báo cáo metric tương ứng.

### 5. P1 — Lấy mẫu valid/test ưu tiên clip dài, không đảm bảo giữ phân bố tự nhiên

Vị trí: `scripts/package_vimd_adalora_kaggle.py:220`.

Sau khi shuffle, thuật toán luôn chọn clip dài nhất còn vừa quota. Shuffle chỉ ảnh hưởng tie; đây không phải random sampling thông thường. Thuật toán bảo toàn gần đúng quota thời lượng từng tỉnh nhưng có thể thay đổi phân bố độ dài, số clip, speaker và giới tính.

Kiểm tra giả lập một tỉnh với 20 clip nữ 5 giây và 20 clip nam 30 giây: nguồn có 50% clip nam, nhưng quota 120 giây chọn 4 clip nam, tức 100%. Đây là bằng chứng về khả năng lệch của thuật toán, không phải khẳng định sample thực tế lệch đúng mức đó.

Hướng sửa: xác định rõ mục tiêu cần bảo toàn; chọn ngẫu nhiên theo strata đã chốt, rồi điều chỉnh quota có giới hạn và xuất chênh lệch phân bố trước/sau. Không mô tả valid/test là giữ nguyên phân bố giới tính nếu chưa đo được sai lệch.

### 6. P2 — Gender đã được coi là unknown khi chọn mẫu nhưng không được lưu nhất quán

Vị trí: `scripts/package_vimd_adalora_kaggle.py:363`, `:477`, `:490`.

`speaker_gender_groups()` phát hiện speaker có metadata mâu thuẫn và gán giới tính hiệu dụng là `None`. Tuy nhiên record được chọn vẫn giữ `gender_code` gốc; manifest xuất `record.gender`, và summary suy lại từ các clip đã chọn. Nếu chỉ chọn một clip của speaker mâu thuẫn, summary có thể lại ghi speaker đó là nam/nữ đã xác định.

Đã tái hiện với một speaker có hai clip mang gender 0 và 1: sampler tính unknown nhưng manifest và summary tính male khi clip gender 1 được chọn.

Hướng sửa: lưu riêng `source_gender` và `resolved_speaker_gender`, dùng trường resolved cho cân bằng và báo cáo. Speaker thiếu giới tính hoàn toàn cũng cần được đếm vào unknown.

### 7. P2 — Benchmark full test giữ audio bytes của toàn bộ test trong RAM

Vị trí: `nb/benchmark_vimd_test_three_models.py:250`.

`load_test_records()` đặt `audio_ds[local_index]["audio"]` vào từng record. Với bytes nhúng trong Arrow, việc này materialize và giữ payload của cả tập test, trái với chú thích chỉ đọc một audio tại một thời điểm. Memory-map của Arrow không loại bỏ các Python bytes đã được giữ trong list.

Hướng sửa: record chỉ giữ shard/index và metadata; đọc audio khi `__getitem__`/inference cần, cache handle dataset theo shard.

### 8. P2 — Resume có thể dùng prediction cũ hoặc bỏ qua vĩnh viễn mẫu lỗi

Vị trí: `nb/benchmark_vimd_test_three_models.py:393`, `:423`; vòng resume trong hai benchmark Transformers/NeMo.

Cache chỉ nhận diện `model` và sample ID/index, chưa kiểm tra revision base, hash adapter, fingerprint dataset, decode settings hoặc metric version. Đổi adapter nhưng giữ tên `small_vimd` có thể khiến kết quả cũ được tái sử dụng. `load_completed_rows()` cũng nhận cả hàng có inference error; kiểm tra cô lập xác nhận hàng lỗi vẫn được coi là đã hoàn tất.

Hướng sửa: run manifest có fingerprint; từ chối resume khi không khớp, cho phép retry lỗi. Kiểm tra tập ID khớp chính xác, không chỉ so số lượng unique.

### 9. P2 — Một số so sánh metric có thể dùng những tập audio thành công khác nhau

Vị trí: `nb/benchmark_vimd_test_three_models.py:481`; phần tổng hợp benchmark Transformers/NeMo và benchmark ba audio.

Các nhóm tự loại hàng lỗi của từng model rồi tính CER/WER riêng. Nếu một model lỗi ở audio khó, metric còn lại có thể nhìn tốt hơn. Có báo số lỗi nhưng chưa bắt buộc so sánh trên cùng tập thành công.

Hướng sửa: benchmark chính phải đủ cùng test IDs hoặc công bố rõ metric trên giao các sample thành công cùng coverage. Giữ prediction rỗng không có exception như một lỗi nhận dạng hợp lệ.

### 10. P2 — Quy tắc CER/WER khác nhau giữa các script; prediction rỗng có thể thành chữ `nan`

Vị trí: `nb/benchmark_SR_models/benchmark_transformers_only.py:469`, `nb/benchmark_SR_models/run/benchmark_nemo_only.py` phần aggregate, `nb/qwen3_asr_adalora_vimd.py` phần evaluation, `nb/qwen3_asr_adalora_kaggle_t4x2.py:958`.

Các benchmark cũ lấy trung bình rate từng clip; benchmark Qwen mới dùng tổng lỗi chia tổng đơn vị tham chiếu. Hai cách trả lời hai câu hỏi khác nhau và không nên dùng cùng tên CER/WER tổng thể mà thiếu mô tả. Chuẩn hóa dấu câu cũng khác giữa Qwen local, các benchmark và script phân tích.

Cell metric cuối của fine-tuning đọc CSV bằng pandas mặc định rồi dùng `str(text or "")`. Prediction rỗng được pandas đọc thành NaN, và NaN được hàm biến thành chuỗi `nan`; kiểm tra cô lập đã tái hiện. Đây làm sai CER/WER của những hàng rỗng không có inference exception.

Hướng sửa: dùng module metric duy nhất, xử lý missing rõ ràng, corpus CER/WER là metric chính; đặt tên riêng cho sentence-macro. Với tiếng Việt, WER tách whitespace là rate theo đơn vị cách nhau bằng khoảng trắng, không phải kết quả của bộ phân đoạn từ tiếng Việt.

### 11. P2 — Revision do người dùng đặt bị ghi đè ở notebook verify

Vị trí: `nb/verify_qwen_adalora_checkpoints.py:265`.

Script cho đặt `MODEL_REVISION` và trước đó đã chọn revision từ config. Cell inference gán lại `training_config.get(...) if CHECKPOINT_ROOT else None`. Khi dùng đường dẫn Kaggle Model riêng, `CHECKPOINT_ROOT` có thể là `None`, khiến revision người dùng đặt bị mất.

Hướng sửa: giải quyết revision một lần với thứ tự ưu tiên rõ ràng; processor và mọi base model phải dùng cùng revision.

### 12. P2 — Smoke chưa kiểm tra những rủi ro chính của training/checkpoint

Vị trí: `nb/qwen3_asr_adalora_kaggle_t4x2.py:749`.

Smoke không dùng DataLoader, không chạy gradient accumulation/GradScaler như full train và chỉ gọi allocation tại step 1. Với `total_steps=20`, `tinit=1`, step này vẫn nằm trong pha budget ban đầu; gọi hàm chưa chứng minh có giảm rank. Không có kiểm tra mask/rank giữa các rank, save/load adapter hoặc prediction sau reload. Labels chỉ được kiểm tra có token khác -100, chưa xác minh toàn bộ user/audio prompt đã được mask.

Hướng sửa: smoke dùng cùng đường train, đi qua ít nhất một mốc đổi budget, assert labels và mask, sau đó round-trip checkpoint nhỏ. [Tài liệu PEFT](https://huggingface.co/docs/peft/package_reference/adalora) mô tả các pha allocation và yêu cầu `update_and_allocate()` trong quá trình train.

### 13. P2 — Checkpoint chưa đủ để resume training chính xác

Vị trí: `nb/qwen3_asr_adalora_kaggle_t4x2.py:715`; các worker output tương ứng.

Chỉ lưu adapter/processor khi valid CER cải thiện. Không lưu optimizer, scheduler, scaler, RNG và state của rank allocator. Các folder `epoch_N_adapter` là artifact cho inference, không phải full training checkpoint. Nếu epoch sau không tốt hơn, không nhất thiết có adapter của epoch đó.

Hướng sửa: tách best inference adapter khỏi resumable training checkpoint; ghi rõ loại artifact và trạng thái training. Bản worker small cũ trong `outputs/fine-tuning/.../worker/train_worker.py:408` vẫn assert successful steps bằng planned steps và còn có thể gây lỗi 877/879; bản full đã có counters. Không dùng các bản worker output làm source chính.

### 14. P2 — Pipeline benchmark → demo chưa dùng định danh nhất quán

Vị trí: `demo/asr_performance.py:53`, `:113`; `nb/benchmark_qwen_adalora_three_test_audio.py` cell super-hard.

Manifest super-hard ghi `filename="super_hard.wav"` và `sample_id="test_super_hard_000000"`. Benchmark mới xuất hai trường này riêng. Scene demo lại tìm ID trong cột `filename`. CSV demo hiện có đã được chỉnh để phù hợp nên chạy được, nhưng thay trực tiếp bằng output benchmark mới sẽ không tìm thấy super-hard.

Hướng sửa: join bằng sample ID ổn định, không biến filename thành ID. Tỉnh/vùng nên đến từ manifest có provenance; scene hiện hard-code cả Quảng Ngãi trong khi manifest nguồn ngoài để tỉnh là unknown.

Phần reveal chữ dùng tỷ lệ thời gian đồng đều theo số từ; đây là visualization transcript lưu sẵn, không phải timestamp từ audio hoặc tốc độ streaming inference. README demo đã nói đúng giới hạn này. Cách dùng một paragraph Pango để giữ baseline nên được giữ lại.

### 15. P2 — Kết quả khai phá từ mới là ứng viên, chưa xác lập từ địa phương

Vị trí: `nb/extract_central_province.py:155`, `:216`, `:292`, `:307`.

- Tokenizer regex tách các đơn vị chữ theo whitespace/dấu câu, chưa phân đoạn từ nhiều âm tiết hoặc xét ngữ cảnh.
- Tập đối chiếu chỉ gồm miền Trung; kết quả mô tả khác biệt giữa các tỉnh trong miền Trung, chưa chứng minh đặc trưng so với cả nước.
- Nhiều token/clip từ cùng speaker không phải bằng chứng độc lập. Thay speaker thiếu bằng một ID mới cho mỗi row có thể phóng đại speaker coverage.
- Chi-square chạy cả bảng có tần suất thấp nhưng không kiểm tra expected counts; FDR không sửa được p-value đầu vào thiếu chính xác. Có thể giữ yêu cầu không lọc frequency và đổi cách kiểm định khi bảng thưa. [SciPy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html) nêu giới hạn của xấp xỉ này và phương án resampling.
- Xếp hạng dùng `ascending=True` cho tần suất, nên top 250 đang là phía ít gặp nhất trong danh sách đã đạt kiểm định. Nếu muốn ưu tiên từ phổ biến, cần đảo chiều cột tần suất. Nếu cố ý ưu tiên từ hiếm, nên ghi rõ.
- Các đường dẫn mặc định còn trỏ `vimd_central_region` trong khi extraction mới xuất `vimd_central`.

Hướng sửa: giữ tên “ứng viên”, dùng kiểm định theo speaker/nhóm phù hợp, tách discovery và confirmation, nghe và gán nhãn thủ công. Không tự thêm lại các ngưỡng frequency mà người dùng đã yêu cầu bỏ.

### 16. P2 — Qwen local prototype cập nhật lịch ngay cả khi AMP skip optimizer

Vị trí: `nb/qwen3_asr_adalora_vimd.py`, hàm `train_adalora()`.

Sau `scaler.step()/update()`, script luôn tăng global step, scheduler và gọi AdaLoRA. Nếu GradScaler skip, các lịch vẫn tiến và allocator có thể gặp gradient không hữu hạn. Nhóm accumulation cuối vẫn chia theo 8 dù có ít hơn 8 clip. Bản DDP mới đã xử lý hai điểm này tốt hơn.

Hướng sửa: dùng chung training engine thay cho duy trì vòng train prototype riêng; test cả bước thành công, overflow và nhóm cuối thiếu batch.

## Kiểm tra riêng các checkpoint hiện có

| Checkpoint | Tensor trong file | Tensor zero-shape | Module có mask rank 0 | Shape so với rank pattern |
|---|---:|---:|---:|---|
| Small epoch 1 | 312 | 0 | 0 | Khớp toàn bộ |
| Small epoch 2 | 312 | 93 | 31 | Khớp toàn bộ |
| Small epoch 3 | 312 | 81 | 27 | Khớp toàn bộ |
| Full epoch 1 | 312 | 3 | 1 | Khớp toàn bộ |
| Full epoch 2 | 312 | 60 | 20 | Khớp toàn bộ |

Mỗi module có A/B/E; zero-shape của các file này phù hợp với rank 0 trong metadata. Đã đối chiếu chiều rank của 312 tensor mỗi adapter, không có mismatch. Đây là kiểm tra cấu trúc **trong artifact**, chưa phải so sánh giá trị tensor trong file với tensor sau khi PEFT nạp. Không kết luận checkpoint hỏng chỉ vì có tensor rỗng, và cũng chưa kết luận loader đã khôi phục chính xác mọi trọng số.

## Danh mục source tự viết

| File | Chức năng và nhận xét |
|---|---|
| `scripts/package_vimd_adalora_kaggle.py` | Package sample; staging/ZIP CRC và chống ghi đè tốt. Cần sửa sampling và resolved gender. |
| `scripts/package_super_hard_instance.py` | Instance ngoài ViMD; giữ transcript nguyên văn, mono 16 kHz, staging và kiểm tra ZIP. Chưa thấy lỗi chặn chạy qua review. |
| `scripts/extract_vimd_named_samples.py` | Tìm tên trên các split, kiểm tra trùng/thiếu. Path cache hard-code, path audio tương đối chưa resolve theo shard. |
| `nb/read_dataset.py` | Đọc/preview dataset; đơn giản. `.to_pandas().head(10)` chuyển metadata cả train rồi mới lấy 10 dòng. |
| `nb/extract_vimd_by_region.py` | Tra cứu audio/Qwen đã tách khỏi `region_df`. Lỗi summary; Run All vẫn load model dù chỉ cần extraction. |
| `nb/extract_central_province.py` | Thống kê/FDR; cần sửa đường dẫn mặc định, cách xếp hạng và giới hạn kiểm định. |
| `nb/vimd_local_di_sr.py` | Sample, SR, train/eval DI. Nhiều trách nhiệm; resample bằng nội suy; AMP scheduler chưa phân biệt skip. |
| `nb/vimd_kaggle_local_di_sr.py` | Bản sao local đổi nguồn Kaggle; cùng vấn đề và token placeholder. Nên dùng cùng engine/config. |
| `nb/vimd_experiment_analysis.py` | Phân tích có provenance, kiểm tra duplicate và corpus metric. Default `vimd_results.csv` không khớp tên `vimd_model_results.csv` do script local xuất. Quy tắc bỏ dấu câu chưa hoàn toàn giống benchmark. |
| `nb/qwen3_asr_adalora_vimd.py` | Prototype local; mặc định giới hạn clip/sample, khác cấu hình thí nghiệm chính. Có vấn đề AMP/accumulation và metric. |
| `nb/qwen3_asr_adalora_kaggle_t4x2.py` | Fine-tuning full ViMD 2 epoch, DDP. Có lỗi config/import; worker là chuỗi lớn khó bảo trì. |
| `nb/fine-tuning.py` | Bản sao lệch; còn lỗi env và một số fix chưa được đồng bộ. |
| `nb/verify_qwen_adalora_checkpoints.py` | Chạy base + 3 epoch sample; bắt lỗi/coverage hữu ích. Revision bị ghi đè, chưa audit tensor sau load. |
| `nb/benchmark_vimd_test_three_models.py` | Full test base/small/full; corpus metric theo tỉnh/vùng. Cần sửa RAM, resume và so sánh cùng sample. |
| `nb/benchmark_qwen_adalora_three_test_audio.py` | Ba audio ViMD + một external, tách metric hai nguồn đúng. Output cần schema định danh chung với demo. |
| `nb/benchmark_SR_models/benchmark_transformers_only.py` | Nhiều backend; cài môi trường và inference trộn nhau, macro metric/resume cần thống nhất. Có code backend NeMo không chạy. |
| `nb/benchmark_SR_models/run/benchmark_nemo_only.py` | NeMo từ metadata; cần xác nhận metadata chỉ chứa test, đối chiếu filename/ref với original index. Có nhiều code backend Transformers không chạy. |
| `nb/benchmark_SR_models/analyze_quick_predictions_metrics.py` | Bị lỗi biến chưa tạo; macro metric có ghi chú nhưng không đồng nhất metric chính. |
| `demo/asr_performance.py` | Layout/reveal/alignment; cần hợp nhất schema benchmark/demo. Không đo streaming latency. |
| `demo/render_demos.py` | CLI render bốn scene; đường dẫn theo file, subprocess có kiểm tra exit code. Chưa render lại trong review. |
| `norm_volumn.py` | File rỗng, hiện chưa có chức năng. |

`nb/benchmark_qwen_adalora_three_test_audio.ipynb`: 16 cell; cú pháp code hợp lệ sau xử lý magic. Các định nghĩa hàm khớp bản `.py` trong lần kiểm tra; chưa thiết lập cơ chế giữ hai bản đồng bộ tự động.

Hai worker trong output là snapshot của lần chạy, không nên sửa để thay thế source. Worker small còn assertion 877/879; worker full có counters nhưng chưa có guard lỗi validation và cleanup `finally` như bản DDP mới.

## Cấu hình dự án và code tham khảo

- `README.md` ở root rỗng; chưa có hướng dẫn chọn entrypoint, schema và quy trình tái lập.
- `pyproject.toml` có `research = "research:main"` và build backend nhưng không có module `src/research`. `tool.uv.package=false` tránh build khi uv sync thông thường, không làm entrypoint/build package trở nên hợp lệ.
- Dependency local chưa bao gồm PEFT và Accelerate; không có test suite tự viết được tìm thấy. Các môi trường training, NeMo và render cần dependency groups/lock riêng thay cho các cell pip khác nhau.
- Các file percent-format còn `%pip`; dùng được như notebook/IPython nhưng chưa phải mọi file đều chạy bằng `python file.py`. Cần ghi rõ hai chế độ hoặc chuyển cài dependency ra ngoài runtime.
- Các đường cache `/home/fernir/...`, Kaggle root và output đang được lặp lại nhiều nơi. Một số root discovery còn yêu cầu thư mục `src` hiện không tồn tại.
- CBA-Whisper có entrypoint chứa API credential literal; không chép giá trị đó sang source mới. Đường inference còn dùng normalizer tiếng Anh; không chuyển nguyên trạng sang ViMD. Một entrypoint gọi postprocess với file handle đã đóng thay vì path, cần sửa trước khi dùng.
- CBA-Whisper fine-tune tạo AdaLoRA nhưng không thấy gọi `update_and_allocate()` trong đường Trainer/callback đã đọc và thiếu `total_step` trong config. Không coi đó là implementation AdaLoRA hoàn chỉnh để bê nguyên sang Qwen. Xem thêm `docs/cba_whisper_code_audit.md`.

## Thành phần nên giữ

- Đọc Arrow với `Audio(decode=False)`, mono 16 kHz và `resample_poly` trong pipeline Qwen mới.
- Audio-only input; tỉnh/vùng dùng phân tích, không đưa nhãn reference vào inference.
- Dò target Linear theo module thực tế, kiểm tra có cả audio encoder và decoder.
- DDP mỗi GPU một model, đếm attempted/successful/skipped và update AdaLoRA trước zero-grad trong bản mới.
- Chọn best bằng valid CER, so base/adapter trên test riêng.
- Lưu prediction cùng error counts và thông tin tỉnh/vùng; corpus metric trong benchmark mới.
- Staging trước publish package local, từ chối overwrite và kiểm tra ZIP.
- Demo dùng paragraph layout chung và nói rõ giới hạn visualization.

## Cấu trúc đề xuất cho lần viết lại

```text
src/mdv_asr/
    config.py
    data/          # schema, source index, audio reader, sampling, packaging
    models/        # Qwen loader, AdaLoRA, checkpoint audit
    training/      # DDP loop, AMP, allocator state, resumable checkpoint
    evaluation/    # normalization, edit counts, inference, resume, aggregation
    analysis/      # tỉnh/vùng, lexical candidates, báo cáo
configs/           # small/full, train/eval và model revisions
scripts/           # CLI mỏng: prepare, train, benchmark, verify
nb/                # percent-format, import thư viện và chạy từng cell
demo/              # đọc schema prediction chung, chỉ xử lý visualization
tests/             # reader, sampler, metric, resume, training-state và checkpoint
```

Thứ tự triển khai: chốt schema/config và metric → reader/sampling → checkpoint audit → train engine → benchmark/resume → notebook và demo. DI có thể giữ như module/thí nghiệm riêng, tránh trộn vào pipeline ASR. Cần kiểm tra tương thích output cũ trước khi chuyển đổi; không xóa artifacts hoặc code tham khảo trong quá trình viết lại.
