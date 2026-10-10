# Kế hoạch tổ chức lại source code ViMD/ASR

> Phạm vi đã được thu hẹp: active `src/mdv_asr` chỉ chứa fine-tuning ASR và benchmark. Sampling/đóng gói dataset, phân tích từ vựng và phân tích khám phá không thuộc package đang chạy; các implementation cũ được lưu ngoài package trong `legacy/`.

## 1. Mục tiêu và nguyên tắc

Viết lại toàn bộ code tự phát triển theo từng giai đoạn, dùng chung trên local và Kaggle.

- Python package là nguồn code chính; notebook chỉ cấu hình, gọi class và hiển thị kết quả.
- Notebook tiếp tục lưu dạng `.py` với cell `# %%`, có thể xuất `.ipynb` khi cần.
- Gom code cũ vào `legacy/`, giữ cấu trúc tương đối để đối chiếu.
- Không di chuyển hoặc sửa dataset, checkpoint, output hiện có hay `third_party`.
- Sửa các lỗi đã review và chuẩn hóa sampler/metric cho thí nghiệm mới; không ghi đè kết quả cũ.
- Không bổ sung phương pháp nghiên cứu mới như phonetic, CBA hoặc MoE trong đợt này.

Class dùng cho thành phần quản lý dữ liệu, model, trạng thái train và artifact. Các phép chuẩn hóa văn bản, edit distance và thống kê thuần vẫn là hàm.

## 2. Cây thư mục đề xuất

```text
Research/
├── pyproject.toml
├── README.md
├── src/mdv_asr/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── types.py
│   │   └── loader.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── errors.py
│   │   └── artifacts.py
│   ├── runtime/
│   │   ├── __init__.py
│   │   ├── environment.py
│   │   ├── distributed.py
│   │   └── launcher.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── records.py
│   │   ├── sources.py
│   │   ├── index.py
│   │   ├── manifest.py
│   │   ├── audio.py
│   │   ├── sampling.py
│   │   ├── packaging.py
│   │   └── inspection.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── factory.py
│   │   ├── qwen.py
│   │   ├── adalora.py
│   │   ├── audit.py
│   │   ├── ctc.py
│   │   ├── whisper.py
│   │   ├── ark.py
│   │   ├── cohere.py
│   │   ├── granite.py
│   │   └── nemo.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── collator.py
│   │   ├── optimization.py
│   │   ├── checkpoints.py
│   │   ├── ada_lora.py
│   │   └── worker.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── records.py
│   │   ├── normalization.py
│   │   ├── metrics.py
│   │   ├── alignment.py
│   │   ├── store.py
│   │   ├── evaluator.py
│   │   ├── benchmark.py
│   │   └── reports.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── lexicon.py
│   │   └── experiments.py
│   └── workflows/
│       ├── __init__.py
│       ├── dataset.py
│       ├── training.py
│       └── benchmark.py
├── configs/
│   ├── datasets/
│   ├── sampling/
│   ├── training/
│   ├── benchmark/
│   └── analysis/
├── demo/                     # Code, config, input và output chỉ dành cho video demo
│   ├── settings.py
│   ├── theme.py
│   ├── layout.py
│   ├── scenes.py
│   ├── renderer.py
│   └── configs/
├── nb/                       # Notebook Python dạng # %%
├── scripts/                  # CLI/build mỏng
├── tests/
├── legacy/                   # Code cũ được giữ nguyên
├── docs/
├── outputs/                  # Artifact hiện có
├── models/                   # Checkpoint hiện có
└── third_party/              # Code tham khảo
```

## 3. Các class dữ liệu

### `data/records.py`

`AudioLocator` chỉ vị trí audio (shard, row index, ZIP member), không giữ bytes. `SampleRecord` đại diện một instance, gồm dataset ID, sample ID, split, filename, transcript, thời lượng, speaker, giới tính, tỉnh, vùng và locator. `AudioSample` chứa waveform đã decode. `SelectionResult` chứa record được chọn, quota, thống kê và cảnh báo.

Method chính:

- `AudioLocator.validate()`, `to_dict()`, `from_dict()`.
- `SampleRecord.key`, `to_dict()`, `from_dict()`.
- `AudioSample.validate()`, `duration_seconds`.
- `SelectionResult.records_for()`, `summary()`, `validate()`.

`filename` chỉ dùng tra cứu/hiển thị; join dữ liệu bằng `(dataset_id, sample_id)`. Transcript gốc không bị sửa.

### `data/sources.py`

`DatasetSource` là interface chung với `build_index()`, `read_bytes()`, `fingerprint()` và `close()`.

- `ArrowDatasetSource`: đọc Arrow cache theo shard/row.
- `ParquetDatasetSource`: đọc Parquet và ánh xạ row.
- `PackagedDatasetSource`: đọc manifest mini-dataset, ZIP hoặc thư mục đã giải nén.
- `DatasetSourceFactory`: chọn implementation và resolve root.

### `data/index.py` và `data/manifest.py`

`DatasetIndex` có `get()`, `for_split()`, `filter()`, `find_by_filename()`, `find_many()`, `to_frame()` và `validate()`.

`ManifestRepository` có `save()`, `load()`, `import_legacy()`, `validate_source()` và `is_reusable()`. Manifest mới có `schema_version`, fingerprint nguồn và locator. Không tự lấy phần tử đầu nếu filename khớp nhiều record.

### `data/audio.py`

`AudioReader`:

- `inspect(record, source)`.
- `read(record, source)`.
- `export(record, source, destination)`.
- `validate_duration(record, audio, tolerance)`.

Chuẩn là mono, `float32`, 16 kHz, resample bằng `resample_poly`, không tự cắt clip trên 30 giây.

`AudioPreview.play()` và `play_by_filename()` chỉ import IPython khi được gọi, không phụ thuộc dataframe lọc vùng và không in raw bytes.

### `data/sampling.py`

`SpeakerGenderResolver` có `resolve()`, `gender_for()`, `conflicts()` và `report()`.

`StratifiedSampler` có:

- `allocate_quotas()`.
- `select_train()`.
- `select_evaluation()`.
- `exclude_overlapping_speakers()`.
- `sample()`.
- `compare_distribution()`.

Sampling có seed, giữ nguyên clip và quota thời lượng; 50–50 là cân bằng speaker nam/nữ ở train theo tỉnh, không phải 50–50 số clip. Valid/test giữ phân bố tự nhiên. Gender mâu thuẫn được lưu là unknown.

### `data/packaging.py` và `data/inspection.py`

`DatasetPackager` có `estimate()`, `write_audio()`, `write_metadata()`, `write_documentation()`, `validate()` và `package()`.

`ExternalInstanceBuilder` xử lý video super-hard: `extract_audio()`, `build_record()`, `package_video()` và `source_checksum()`.

`DatasetInspector` có `summarize()`, `speaker_overlap()`, `validate_records()`, `check_audio()`, `longest_records()` và `export_report()`.

## 4. Cấu hình, runtime và artifact

### `config/types.py`

Các dataclass:

- `DatasetConfig`.
- `SamplingConfig`.
- `ModelSpec`.
- `AdaLoraSettings`.
- `DecodeConfig`.
- `TrainConfig`.
- `BenchmarkConfig`.
- `LexiconConfig`.
- `ExperimentConfig`.

Mỗi config có `validate()` và `to_dict()`. `num_workers`, model revision, checkpoint path và target modules đều được validate trước khi chạy.

### `config/loader.py`

`ConfigLoader.load()`, `load_sections()`, `with_overrides()` và `save_resolved()` thay cho dictionary global giữa các cell.

### `core/errors.py`

Định nghĩa `ConfigError`, `AmbiguousSampleError`, `DatasetIntegrityError`, `CheckpointMismatchError`, `ResumeMismatchError` và `IncompleteEvaluationError`.

### `core/artifacts.py`

`ArtifactStore` có `prepare_run()`, `write_json()`, `write_csv()`, `read_json()`, `write_run_manifest()` và `path()`. Các hàm `hash_file()` và `hash_payload()` tạo fingerprint. Ghi file phải atomic.

### `runtime/environment.py`

`RuntimeEnvironment.detect()`, `validate()`, `get_hf_token()`, `subprocess_env()` và `describe()`. Không hard-code token.

### `runtime/distributed.py`

`DistributedContext.initialize()`, `barrier()`, `gather_objects()`, `all_ranks_agree()`, `reduce_sum()` và `close()`. Cleanup nằm trong `finally`.

### `runtime/launcher.py`

`WorkerLauncher.build_command()`, `launch()`, `smoke()`, `train()` và `evaluate()`. Không còn `WORKER_PART_A/B/C` dạng chuỗi nhúng trong notebook.

## 5. Model và AdaLoRA

### `models/base.py` và `models/factory.py`

`ModelBundle` chứa model, processor, revision, device/dtype và metadata. `TranscriptionResult` chứa prediction, thời gian và decode metadata.

`ASRBackend` định nghĩa `load()`, `transcribe()`, `metadata()` và `close()`. `ASRBackendFactory.create()` chọn backend theo `ModelSpec`.

### `models/qwen.py`

`QwenModelLoader` có `load_processor()`, `load_base()`, `load_adapter()`, `resolve_revision()` và `release()`.

`QwenASRBackend` có `load()`, `from_loaded()`, `prepare_request()`, `transcribe()`, `decode_generated()`, `metadata()` và `close()`.

Inference luôn chỉ nhận audio. Reference được lấy từ `SampleRecord` ở tầng evaluation.

### `models/adalora.py`

`AdaLoraManager` có `find_targets()`, `attach()`, `trainable_summary()`, `update_after_step()`, `rank_report()`, `state_dict()`, `load_state_dict()` và `validate_rank_consistency()`.

Target mặc định là `q_proj`/`v_proj` của audio encoder và decoder. Chỉ gọi `update_after_step()` sau optimizer step thành công.

### `models/audit.py`

`AdapterAuditor` có `inspect_files()`, `inspect_tensors()`, `check_rank_pattern()`, `compare_loaded_state()`, `check_forward()` và `audit()`.

Tensor rank 0 không tự động bị xem là lỗi; phải đối chiếu với mask/rank metadata và state sau load.

### Backend khác

Mỗi backend có cùng interface:

- `CTCASRBackend` trong `ctc.py`.
- `WhisperASRBackend` trong `whisper.py`.
- `ArkASRBackend` trong `ark.py`.
- `CohereASRBackend` trong `cohere.py`.
- `GraniteASRBackend` trong `granite.py`.
- `CanaryASRBackend` và `ParakeetASRBackend` trong `nemo.py`.

Dependency backend được import lazy, không làm package core phụ thuộc NeMo/Manim.

## 6. Training và checkpoint

### `training/dataset.py` và `collator.py`

`ASRAudioDataset` có `__len__()`, `__getitem__()`, `__getstate__()` và `close()`.

`ASRDataLoaderFactory` có `create()`, `make_sampler()`, `worker_init_fn()` và `steps_per_epoch()`.

`QwenTrainingCollator` có `build_target()`, `build_messages()`, `__call__()` và `validate_labels()`. Assistant target có dạng `language Vietnamese<asr_text>...`; chỉ phần assistant được học.

### `training/optimization.py`

`StepResult` ghi attempted/successful/skipped, grad norm và AMP scale.

`OptimizerStepController` có `backward()`, `step()`, `zero_grad()`, `state_dict()` và `load_state_dict()`.

`TrainProgress` có `record_step()`, `validate()`, `to_dict()` và `from_dict()`.

### `training/checkpoints.py`

`CheckpointManager` có:

- `save_epoch_adapter()`.
- `save_training_state()`.
- `load_training_state()`.
- `update_best()`.
- `get_best_adapter()`.
- `validate_resume()`.

Tách artifact inference:

```text
epoch_2_adapter/                 # adapter + processor, dùng inference
training_checkpoints/epoch_2/    # optimizer, scheduler, scaler, allocator, RNG
```

Checkpoint resume lưu base revision, config, adapter state, optimizer, scheduler, scaler, AdaLoRA allocator, RNG từng rank, epoch, counters và history. Phiên bản đầu chỉ resume ở ranh giới epoch.

### `training/ada_lora.py` và `worker.py`

`QwenAdaLoraTrainer` có `setup()`, `smoke()`, `fit()`, `train_epoch()`, `train_accumulation_group()`, `validate()`, `save_epoch()` và `close()`.

Thứ tự step:

```text
backward → kiểm tra AMP/gradient → optimizer update
→ successful step → scheduler → AdaLoRA allocation → zero_grad
```

`training/worker.py` chỉ parse args, tạo distributed context, gọi trainer/evaluator và cleanup. Không chứa thuật toán train.

## 7. Evaluation, benchmark và metric

### `evaluation/records.py`

`PredictionRecord` chứa sample/model ID, reference, prediction, tỉnh/vùng, thời gian và error. `ErrorCounts` chứa edit errors và số đơn vị tham chiếu. `EvaluationResult` chứa prediction, counts và coverage.

### `evaluation/normalization.py` và `metrics.py`

`TextNormalizer` có `normalize()`, `tokens()`, `characters()` và `normalize_with_spans()`.

Các hàm thuần gồm `edit_distance()`. `ErrorCounter` có `count()` và `sum_counts()`. `MetricAggregator` có `overall()`, `by_region()`, `by_province()`, `compare_models()` và `sentence_macro()`.

Metric chính là corpus CER/WER: tổng lỗi chia tổng độ dài reference. Prediction rỗng là lỗi thật, không biến thành chuỗi `nan`.

### `evaluation/alignment.py`

`WordAligner.align()`, `prediction_spans()` và `deletions()` phục vụ highlight đúng/sai trong demo.

### `evaluation/store.py`

`PredictionStore.open()`, `completed_keys()`, `pending()`, `upsert()`, `flush()`, `load_all()` và `validate_expected_ids()`.

Resume benchmark kiểm tra dataset fingerprint, model revision, adapter checksum, decode config và metric/schema version. Mẫu lỗi được retry.

### `evaluation/evaluator.py`, `benchmark.py`, `reports.py`

`ASREvaluator.evaluate_model()`, `evaluate_one()`, `merge_distributed()` và `validate_coverage()`.

`BenchmarkRunner.prepare()`, `run_model()`, `run()`, `resume()`, `validate_comparison()` và `close()`.

`BenchmarkReporter.write_predictions()`, `write_overall_metrics()`, `write_region_metrics()`, `write_province_metrics()`, `write_comparison()`, `write_coverage_report()` và `plot_summary()`.

Load/evaluate/close model tuần tự khi chạy trên một GPU. Benchmark phải dùng cùng test IDs cho tất cả model.

## 9. Analysis và từ địa phương

### `analysis/dataset.py`

`DatasetAnalyzer` có `split_summary()`, `province_summary()`, `region_summary()`, `gender_summary()`, `speaker_summary()`, `duration_distribution()`, `compare_sample()`, `plot_distributions()` và `export()`.

### `analysis/lexicon.py`

`LexicalTokenizer` có `tokenize()`, `unique_tokens()` và `describe_policy()`.

`LexicalStatisticsBuilder` có `build_token_counts()`, `build_document_counts()`, `build_speaker_presence()` và `build()`.

`ProvinceSpecificityScorer` có `tf_ipf()`, `log_odds()`, `pmi()` và `score()`.

`LexicalSignificanceTester` có `build_speaker_table()`, `test()` và `adjust_fdr()`.

`LexiconAnalyzer` có `run()`, `rank_candidates()`, `collect_examples()` và `export()`.

Kết quả được gọi là “ứng viên”; không tự kết luận đó là từ địa phương đã xác nhận. Không thêm lại minimum frequency/DF/speaker coverage. Xếp hạng frequency giảm dần và lưu cả bảng đầy đủ lẫn top-K.

### `analysis/experiments.py`

`ExperimentAnalyzer` có `load_results()`, `import_legacy_results()`, `validate_provenance()`, `compare_asr_models()`, `compare_training_histories()`, `plot_comparisons()` và `export_report()`.

## 10. Video demo (`demo/`)

Demo là công cụ độc lập ở thư mục gốc `demo/`, không thuộc package `mdv_asr` và không nằm trong config/CLI tổng quát. Nó dùng lại module đánh giá chung từ `mdv_asr.evaluation`.

### `demo/settings.py`

`DemoConfig` đọc/kiểm tra cấu hình video riêng: sample IDs, model keys, audio override, độ phân giải, FPS, font, palette và output directory.

### `demo/theme.py` và `demo/layout.py`

`DemoTheme` giữ palette/font/layout tokens. `WaveformVisualizer` tính mức waveform và cung cấp dữ liệu cho animation. Hiệu ứng reveal không được thay đổi baseline giữa các từ.

### `demo/scenes.py`

`ASRComparisonScene` có `load_case()`, `build_header()`, `build_audio_panel()`, `build_reference()`, `build_prediction_panels()`, `animate_playback()`, `show_metrics_and_alignment()` và `construct()`.

`TranscriptPreviewScene` render một panel để kiểm tra layout trước khi render đầy đủ.

### `demo/renderer.py`

`DemoRenderer` có `validate_inputs()`, `build_case_manifest()`, `render_case()`, `render_all()`, `render_preview()` và `validate_outputs()`.

`demo/render_demos.py` là entrypoint dòng lệnh mỏng. Cấu hình, prediction inputs và video outputs riêng của demo được đặt dưới `demo/`. Demo chỉ đọc prediction đã benchmark; không gọi ASR trong lúc render. Kết nối dữ liệu bằng sample ID, không dùng filename như ID.

## 11. Workflow dùng từ notebook

### `workflows/dataset.py`

`DatasetWorkflow.from_config()`, `scan()`, `inspect()`, `find_by_filename()`, `play_by_filename()`, `extract()`, `sample()`, `package()` và `close()`.

### `workflows/training.py`

`ASRTrainingWorkflow.prepare()`, `smoke()`, `train()`, `evaluate_best()`, `history()` và `best_adapter()`.

### `workflows/benchmark.py`

`ASRBenchmarkWorkflow.prepare()`, `audit_adapters()`, `predict_by_filename()`, `run_test()`, `run_cases()`, `report()` và `close()`.

Ví dụ notebook:

```python
dataset = DatasetWorkflow.from_config(config.dataset)
dataset.scan()
dataset.play_by_filename("37_0338.wav")

benchmark = ASRBenchmarkWorkflow(config, dataset)
result = benchmark.predict_by_filename("qwen_full", "37_0338.wav")
print(result.reference)
print(result.prediction)
```

Notebook không cần chạy trước một cell lọc miền Trung để prediction hoạt động.

## 12. Chuyển source cũ

| Source cũ | Đích mới |
|---|---|
| `read_dataset.py` | `data/sources`, `data/index`, `data/audio` |
| `extract_vimd_by_region.py` | dataset workflow, reader, preview, benchmark workflow |
| `extract_vimd_named_samples.py` | dataset index và extraction workflow |
| `package_vimd_adalora_kaggle.py` | sampling, gender resolver, packager, inspector |
| `package_super_hard_instance.py` | `ExternalInstanceBuilder` |
| Các notebook Qwen/AdaLoRA | Qwen loader, AdaLoRA manager, trainer, checkpoint manager |
| Verify/benchmark Qwen | adapter auditor, evaluator, benchmark runner, reporter |
| Benchmark Transformers/NeMo | backend classes và benchmark runner |
| Experiment analysis | `analysis/dataset.py`, `analysis/experiments.py` |
| Central province analysis | `analysis/lexicon.py` |
| Demo scripts/configs | `demo/` |
| `norm_volumn.py` | giữ trong `legacy/` vì hiện đang rỗng |

Worker trong `outputs/` là snapshot cũ, không dùng làm source mới. Không xóa artifact hiện có.

## 13. Notebook, CLI và Kaggle

Notebook mới chỉ có các cell:

1. Kiểm tra môi trường.
2. Import package.
3. Load config và override.
4. Khởi tạo object.
5. Smoke/validate.
6. Chạy task.
7. Hiển thị và xuất report.

`cli.py` cung cấp các nhóm lệnh `data inspect`, `data extract`, `data sample`, `data package`, `train`, `benchmark`, `audit`, `analysis dataset` và `analysis lexicon`. Video demo chạy riêng bằng `python demo/render_demos.py`; demo-specific code/config không nằm trong package hay CLI chính.

`scripts/build_kaggle_bundle.py` build wheel, gom config, requirements và checksum; không chứa dataset, checkpoint hay token. Kaggle cài wheel bằng `pip --no-deps` để không thay Torch/NumPy có sẵn.

`scripts/export_notebooks.py` kiểm tra cell `# %%` và sinh `.ipynb`. File `.py` là source chính.

## 14. Kiểm thử và thứ tự triển khai

### Unit tests

- Config validation và serialization.
- Dataset ID/filename/split.
- Arrow/Parquet/ZIP reader.
- Audio mono/stereo/resample/duration/NaN.
- Gender conflict và sampler có seed.
- CER/WER, prediction rỗng/NaN và dấu tiếng Việt.
- Alignment insertion/substitution/deletion.
- Benchmark fingerprint, retry và duplicate.
- Lexicon speaker coverage và FDR.

### Integration tests

- Package → validate → reload → đọc audio.
- Benchmark nhiều model trên cùng test IDs.
- Output benchmark đọc được bởi demo.
- Notebook script chạy trong kernel sạch.
- Wheel build/import không gây side effect.

### GPU tests

- Qwen forward/backward với collator thật.
- Assistant mask đúng.
- AMP skip và nhóm accumulation cuối.
- DDP hai rank thống nhất optimizer/AdaLoRA.
- Allocation đi qua mốc thay đổi rank.
- Adapter save/load round-trip.
- Resume đầy đủ ở cuối epoch.

### Giai đoạn triển khai

1. Tạo package, config, records, runtime, metric và legacy manifest.
2. Hợp nhất reader, index, audio, sampling và packaging.
3. Hợp nhất Qwen, AdaLoRA, auditor, trainer và checkpoint.
4. Hợp nhất benchmark, cache, reports và notebook mỏng.
5. Hoàn thiện các backend ASR và analysis cần thiết.
6. Chuyển demo, build Kaggle bundle, tài liệu và test cuối.

Kết quả hoàn tất: sửa một reader, metric hoặc model loader chỉ cần sửa một module; notebook local/Kaggle, CLI và demo đều tái sử dụng cùng class đó. Dataset, checkpoint và báo cáo cũ vẫn giữ nguyên để đối chiếu.
