# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Benchmark Qwen3-ASR trên toàn bộ test split ViMD
#
# So sánh cùng một test split với ba mô hình: Qwen3-ASR pretrained, AdaLoRA fine-tuned
# trên sample ViMD 31,25 giờ, và AdaLoRA fine-tuned trên toàn bộ train split ViMD.
# File này là notebook ở định dạng Python script (`# %%` cells); giữ nguyên cấu trúc
# thư mục adapter như hướng dẫn trong cell 1. Kết quả và tiến độ được ghi vào
# `/kaggle/working/vimd_three_model_benchmark/`.
#
# Lưu ý trước khi chạy: snapshot adapter hiện tại trong workspace có tensor 0 chiều
# (small: 81; full: 60). Đây có thể làm PEFT cảnh báo hoặc bỏ qua tensor khi nạp.
# Notebook vẫn ghi chẩn đoán vào CSV để tiện điều tra, nhưng metric adapter có cảnh
# báo không nên xem là benchmark cuối cùng cho đến khi xác minh checkpoint.

# %% [markdown]
# ## 1. Upload hai adapter bằng Kaggle Models CLI
#
# Không upload base Qwen 1.7B: hai fine-tuned checkpoints là PEFT AdaLoRA adapters.
# Tạo **một Kaggle Model cha** `qwen-vimd-adalora`, rồi tạo trong đó hai variations
# framework `pyTorch`: `small-vimd` và `full-vimd`. Như vậy chúng có đường dẫn/loader
# tương ứng, còn notebook vẫn nạp cùng `Qwen/Qwen3-ASR-1.7B-hf` làm base.
#
# Checkpoint tốt nhất theo valid CER trong các output hiện tại:
# - small ViMD: `outputs/fine-tuning/vimd_qwen_adalora/epoch_3_adapter/`
# - full ViMD: `nb/outputs/fine-tuning 2/vimd_qwen_adalora_full_2epoch/epoch_2_adapter/`
#
# Chuẩn bị ba thư mục local (các checkpoint được giữ tách biệt):
#
# ```bash
# mkdir -p qwen-vimd-parent qwen-vimd-small qwen-vimd-full
# cp -a outputs/fine-tuning/vimd_qwen_adalora/epoch_3_adapter/. qwen-vimd-small/
# cp -a "nb/outputs/fine-tuning 2/vimd_qwen_adalora_full_2epoch/epoch_2_adapter/." qwen-vimd-full/
# kaggle models init -p qwen-vimd-parent
# # Sửa model-metadata.json: ownerSlug, title, slug, description, isPrivate=true.
# kaggle models create -p qwen-vimd-parent
# kaggle models variations init -p qwen-vimd-small
# # Sửa model-instance-metadata.json: ownerSlug, modelSlug=qwen-vimd-adalora,
# # instanceSlug=small-vimd, framework=pyTorch, usage/overview.
# kaggle models variations create -p qwen-vimd-small -r skip
# kaggle models variations init -p qwen-vimd-full
# # Tương tự, instanceSlug=full-vimd.
# kaggle models variations create -p qwen-vimd-full -r skip
# ```
#
# Lệnh `models init/create` tạo model cha; mỗi variation có metadata riêng rồi được tải
# bằng `models variations create`. Xem [Kaggle Models CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/models.md),
# [variation commands](https://github.com/Kaggle/kaggle-cli/blob/main/docs/model_variations.md)
# và [metadata schema](https://github.com/Kaggle/kaggle-cli/blob/main/docs/models_metadata.md).
# Sau upload, attach **cả hai variations** vào notebook qua Add Input → Models. Cell 2 tự
# tìm `adapter_config.json` dưới `/kaggle/input` theo tên variation; nếu không tìm thấy,
# điền đường dẫn mount thủ công.
#
# ViMD được ghi nhận license CC BY-NC-ND 4.0 trong package hiện tại. Giữ dataset/model
# private và xác nhận quyền lưu trữ/phân phối adapter trước khi chia sẻ công khai.

# %% [code]
# Nếu môi trường Kaggle thiếu package, bỏ comment và chạy riêng cell này rồi restart.
# Không force reinstall torch/numpy của Kaggle.
# %pip install -q "transformers==5.17.0" "peft==0.21.1" "datasets>=3.0" soundfile scipy safetensors pandas

import gc
import json
import math
import os
import re
import struct
import time
import unicodedata
import warnings
from io import BytesIO
from pathlib import Path
from math import gcd

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, Dataset
from IPython.display import display
from peft import PeftModel
from scipy.signal import resample_poly
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("Bật GPU trong Kaggle Settings trước khi chạy benchmark.")

# %% [markdown]
# ## 2. Đường dẫn và cấu hình
#
# Gắn ViMD `thanhlamdev/vimd-dataset` và hai Kaggle Model variations vào notebook.
# Nếu auto-discovery không tìm thấy variation, đặt override paths bên dưới.

# %% [code]
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
MODEL_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
LANGUAGE = "Vietnamese"
MAX_NEW_TOKENS = 768
MODEL_DTYPE = torch.float16
DEVICE = torch.device("cuda:0")

DATASET_ROOT = Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0")
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
ADAPTER_VARIATIONS = {"small_vimd": "small-vimd", "full_vimd": "full-vimd"}
ADAPTER_PATH_OVERRIDES = {"small_vimd": None, "full_vimd": None}
OUTPUT_DIR = Path("/kaggle/working/vimd_three_model_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAVE_EVERY = 25
RESUME = True
# False = vẫn chạy và gắn cảnh báo nếu PEFT báo shape/rank. True = dừng để tránh
# công bố metric của adapter chưa được xác minh.
STRICT_ADAPTER_LOAD = False

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None

def discover_model_variation(variation_slug):
    normalized_slug = variation_slug.lower().replace("_", "-")
    candidates = []
    if KAGGLE_INPUT_ROOT.exists():
        for config_path in KAGGLE_INPUT_ROOT.rglob("adapter_config.json"):
            parts = {part.lower().replace("_", "-") for part in config_path.parts}
            if normalized_slug in parts:
                candidates.append(config_path.parent.resolve())
    candidates = sorted(set(candidates))
    if not candidates:
        found = sorted(str(path.parent) for path in KAGGLE_INPUT_ROOT.rglob("adapter_config.json"))
        raise FileNotFoundError(
            f"Không tìm thấy variation {variation_slug} dưới {KAGGLE_INPUT_ROOT}. "
            f"Các adapter_config.json khả dụng: {found[:20]}. "
            "Attach đúng model hoặc điền ADAPTER_PATH_OVERRIDES."
        )
    if len(candidates) > 1:
        # Nếu đã attach nhiều versions của cùng variation, chọn version mới nhất
        # (version id thường là thư mục số cuối cùng trong mount path).
        def version_key(path):
            return int(path.name) if path.name.isdigit() else -1
        highest = max(version_key(path) for path in candidates)
        latest = [path for path in candidates if version_key(path) == highest]
        if len(latest) != 1:
            raise RuntimeError(f"Nhiều mount paths cho {variation_slug}: {candidates}; đặt override.")
        return latest[0]
    return candidates[0]


ADAPTER_PATHS = {
    key: Path(ADAPTER_PATH_OVERRIDES[key]).resolve() if ADAPTER_PATH_OVERRIDES[key]
    else discover_model_variation(slug)
    for key, slug in ADAPTER_VARIATIONS.items()
}

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(f"Không thấy ViMD tại {DATASET_ROOT}; sửa DATASET_ROOT.")
for model_name, adapter_path in ADAPTER_PATHS.items():
    for required in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter_path / required).is_file():
            raise FileNotFoundError(f"{model_name}: thiếu {adapter_path / required}")
print("ViMD:", DATASET_ROOT)
print("Adapters:", ADAPTER_PATHS)
print("Output:", OUTPUT_DIR)

# %% [markdown]
# ## 3. Nạp toàn bộ test split từ Arrow và đọc audio theo từng mẫu
#
# Chỉ đọc metadata và một audio tại một thời điểm; không giải nén/copy toàn bộ audio.
# Nếu shard lưu audio bằng path thay vì bytes, resolver thử đường dẫn tương đối so với
# thư mục shard và dataset root.

# %% [code]
def find_test_shards(root):
    shards = sorted(root.rglob("vi_md_dataset-test-*.arrow"))
    if not shards:
        # Fallback cho bản export dùng tên file shard khác.
        shards = sorted(p for p in root.rglob("*.arrow") if "test" in p.name.lower())
    if not shards:
        raise FileNotFoundError(f"Không tìm thấy Arrow test shards dưới {root}")
    return shards


def audio_to_array(audio_info, shard_path):
    raw = audio_info.get("bytes")
    if raw is not None:
        source = BytesIO(raw)
    else:
        raw_path = audio_info.get("path")
        if not raw_path:
            raise ValueError(f"Audio row không có bytes/path trong {shard_path}")
        audio_path = Path(raw_path)
        candidates = [audio_path] if audio_path.is_absolute() else [
            shard_path.parent / audio_path,
            DATASET_ROOT / audio_path,
        ]
        resolved = next((p for p in candidates if p.is_file()), None)
        if resolved is None:
            raise FileNotFoundError(f"Không tìm thấy audio path={raw_path}; shard={shard_path}")
        source = str(resolved)
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0]
    if sample_rate != 16000:
        divisor = gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all() or not len(audio):
        raise ValueError(f"Waveform không hợp lệ trong {shard_path}")
    return audio


def load_test_records():
    records = []
    global_index = 0
    shards = find_test_shards(DATASET_ROOT)
    for shard_path in shards:
        raw_ds = Dataset.from_file(str(shard_path))
        needed = [c for c in ("audio", "text", "filename", "region", "province_name", "province_code", "speakerID")
                  if c in raw_ds.column_names]
        if "audio" not in needed or "text" not in needed:
            raise KeyError(f"Shard thiếu audio/text: {shard_path}; columns={raw_ds.column_names}")
        meta_ds = raw_ds.select_columns([c for c in needed if c != "audio"])
        audio_ds = raw_ds.cast_column("audio", Audio(decode=False))
        for local_index, row in enumerate(meta_ds):
            reference = str(row.get("text") or "").strip()
            if not reference:
                raise ValueError(f"Transcript rỗng: {shard_path}, row={local_index}")
            records.append({
                "sample_id": f"test_{global_index:07d}",
                "filename": str(row.get("filename") or f"{global_index:07d}"),
                "reference": reference,
                "region": row.get("region"),
                "province_name": row.get("province_name"),
                "province_code": row.get("province_code"),
                "speakerID": row.get("speakerID"),
                "shard_path": str(shard_path),
                "shard_row_index": local_index,
                "audio_info": audio_ds[local_index]["audio"],
            })
            global_index += 1
    ids = [row["sample_id"] for row in records]
    if not records or len(ids) != len(set(ids)):
        raise ValueError("Test split rỗng hoặc sample_id bị trùng.")
    return records


TEST_RECORDS = load_test_records()
print("Test shards:", len(find_test_shards(DATASET_ROOT)))
print("Test instances:", len(TEST_RECORDS))
print("Audio hours (header scan will occur during inference).")

# %% [markdown]
# ## 4. Kiểm tra adapter package và checkpoint trước khi inference
#
# `PeftModel.from_pretrained(base, adapter_dir)` dùng cùng API cho cả hai thư mục.
# Chỉ đọc header SafeTensors để đếm tensor zero-shape; không nạp toàn bộ trọng số vào RAM.

# %% [code]
def inspect_safetensors_header(path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    tensors = {key: value for key, value in header.items() if key != "__metadata__"}
    zero = [(key, value.get("shape", [])) for key, value in tensors.items()
            if any(dim == 0 for dim in value.get("shape", []))]
    return len(tensors), zero


ADAPTER_DIAGNOSTICS = {}
for model_name, adapter_path in ADAPTER_PATHS.items():
    config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError(f"{model_name}: base model mismatch: {config.get('base_model_name_or_path')}")
    if str(config.get("peft_type", "")).upper() != "ADALORA":
        raise ValueError(f"{model_name}: expected AdaLoRA, got {config.get('peft_type')}")
    tensor_count, zero_shapes = inspect_safetensors_header(adapter_path / "adapter_model.safetensors")
    ADAPTER_DIAGNOSTICS[model_name] = {
        "tensor_count": tensor_count,
        "zero_shape_count": len(zero_shapes),
        "zero_shape_examples": zero_shapes[:5],
        "rank_pattern_entries": len(config.get("rank_pattern", {})),
        "config_base_model": config.get("base_model_name_or_path"),
    }
    print(model_name, ADAPTER_DIAGNOSTICS[model_name])

# %% [markdown]
# ## 5. Metric và hàm load/transcribe
#
# CER/WER chuẩn hóa Unicode NFC, chữ thường, punctuation/whitespace; CER bỏ khoảng trắng.
# Reference transcript giữ nguyên text gốc để có thể audit kết quả.

# %% [code]
def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def error_counts(reference, prediction):
    ref, hyp = normalize_text(reference), normalize_text(prediction)
    ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    ref_words, hyp_words = ref.split(), hyp.split()
    return {
        "cer_errors": edit_distance(ref_chars, hyp_chars),
        "cer_units": max(1, len(ref_chars)),
        "wer_errors": edit_distance(ref_words, hyp_words),
        "wer_units": max(1, len(ref_words)),
    }


PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, revision=MODEL_REVISION, token=HF_TOKEN)


def load_base():
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, token=HF_TOKEN,
        dtype=MODEL_DTYPE, low_cpu_mem_usage=True,
    )
    return model.to(DEVICE).eval()


def transcribe(model, audio):
    inputs = PROCESSOR.apply_transcription_request(audio=audio, language=LANGUAGE)
    model_dtype = next(model.parameters()).dtype
    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = (value.to(device=DEVICE, dtype=model_dtype) if value.is_floating_point()
                          else value.to(DEVICE))
        else:
            moved[key] = value
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**moved, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    prompt_len = int(moved["input_ids"].shape[-1]) if "input_ids" in moved else 0
    generated = output_ids[:, prompt_len:] if prompt_len and output_ids.shape[-1] > prompt_len else output_ids
    decoded = PROCESSOR.decode(generated[0], return_format="transcription_only")
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if isinstance(decoded, dict):
        decoded = decoded.get("text", decoded.get("transcription", str(decoded)))
    return str(decoded).strip(), elapsed

# %% [markdown]
# ## 6. Chạy ba mô hình, lưu tiến độ và cho phép resume
#
# Mỗi adapter được gắn lên một base mới, tránh adapter trước làm thay đổi model sau.
# Kết quả từng model được ghi định kỳ vào CSV; nếu Kaggle session ngắt, chạy lại cell sẽ
# tiếp tục từ các `sample_id` đã hoàn tất. Model status có cảnh báo nếu checkpoint chứa
# tensor zero-shape hoặc PEFT phát cảnh báo lúc load.

# %% [code]
MODEL_SPECS = [
    ("qwen_base", None),
    ("small_vimd", ADAPTER_PATHS["small_vimd"]),
    ("full_vimd", ADAPTER_PATHS["full_vimd"]),
]
LOAD_STATUS = {"qwen_base": "BASE_MODEL"}


def atomic_write_csv(frame, path):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def load_completed_rows(path, model_name):
    if RESUME and path.is_file():
        previous = pd.read_csv(path)
        previous = previous[previous["model"].eq(model_name)]
        return {str(row.sample_id): row._asdict() for row in previous.itertuples(index=False)}
    return {}


def make_row(model_name, record, model, error="", elapsed=np.nan):
    prediction = ""
    if not error:
        audio = audio_to_array(record["audio_info"], Path(record["shard_path"]))
        prediction, elapsed = transcribe(model, audio)
    metrics = error_counts(record["reference"], prediction) if not error else {
        "cer_errors": np.nan, "cer_units": np.nan, "wer_errors": np.nan, "wer_units": np.nan,
    }
    return {
        "sample_id": record["sample_id"], "model": model_name,
        "filename": record["filename"], "region": record["region"],
        "province_name": record["province_name"], "province_code": record["province_code"],
        "speakerID": record["speakerID"], "reference": record["reference"],
        "prediction": prediction, **metrics,
        "CER": metrics["cer_errors"] / metrics["cer_units"] if not error else np.nan,
        "WER": metrics["wer_errors"] / metrics["wer_units"] if not error else np.nan,
        "inference_seconds": elapsed, "error": error,
    }


for model_name, adapter_path in MODEL_SPECS:
    output_path = OUTPUT_DIR / f"predictions_{model_name}.csv"
    results_by_id = load_completed_rows(output_path, model_name)
    pending = [row for row in TEST_RECORDS if row["sample_id"] not in results_by_id]
    print(f"\n{model_name}: cached {len(results_by_id):,}/{len(TEST_RECORDS):,}; pending {len(pending):,}")
    model = None
    load_warning_messages = []
    try:
        model = load_base()
        if adapter_path is not None:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
            model.to(DEVICE).eval()
            load_warning_messages = [str(w.message) for w in caught]
        zero_count = (ADAPTER_DIAGNOSTICS[model_name]["zero_shape_count"]
                      if model_name in ADAPTER_DIAGNOSTICS else 0)
        rank_warning = any("rank_pattern" in message.lower() for message in load_warning_messages)
        LOAD_STATUS[model_name] = (
            "CHECKPOINT_WARNING" if adapter_path is not None and (zero_count or load_warning_messages)
            else "LOADED"
        )
        if adapter_path is not None and (zero_count or rank_warning):
            details = {
                "zero_shape_tensors": zero_count,
                "rank_pattern_warning": rank_warning,
                "load_warnings": load_warning_messages,
            }
            (OUTPUT_DIR / f"load_diagnostics_{model_name}.json").write_text(
                json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print("CHECKPOINT LOAD WARNING:", json.dumps(details, ensure_ascii=False)[:2000])
            if STRICT_ADAPTER_LOAD:
                raise RuntimeError(f"Strict mode: adapter {model_name} có cảnh báo load/shape.")

        started_model = time.perf_counter()
        for index, record in enumerate(pending, 1):
            try:
                result = make_row(model_name, record, model)
            except Exception as exc:
                result = make_row(model_name, record, model, error=repr(exc))
            results_by_id[record["sample_id"]] = result
            if index % SAVE_EVERY == 0 or index == len(pending):
                frame = pd.DataFrame(results_by_id.values()).sort_values("sample_id")
                atomic_write_csv(frame, output_path)
                print(f"{model_name}: {len(results_by_id):,}/{len(TEST_RECORDS):,} | "
                      f"{time.perf_counter() - started_model:.1f}s elapsed")
        if not pending and results_by_id:
            atomic_write_csv(pd.DataFrame(results_by_id.values()).sort_values("sample_id"), output_path)
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Tính metric corpus-level tổng thể, theo vùng và tỉnh
#
# Chỉ mẫu không lỗi inference góp vào CER/WER. File prediction vẫn giữ hàng lỗi để audit.

# %% [code]
def aggregate_metrics(frame, model_name, level, group_name, group_frame):
    successful = group_frame[group_frame["error"].fillna("").eq("")]
    if successful.empty:
        return {"model": model_name, "level": level, "group": group_name,
                "checkpoint_status": LOAD_STATUS.get(model_name, "STATUS_NOT_CAPTURED"),
                "n_samples": 0, "failed_samples": len(group_frame),
                "CER": np.nan, "WER": np.nan, "cer_errors": 0, "cer_units": 0,
                "wer_errors": 0, "wer_units": 0}
    cer_errors = int(successful["cer_errors"].sum())
    cer_units = int(successful["cer_units"].sum())
    wer_errors = int(successful["wer_errors"].sum())
    wer_units = int(successful["wer_units"].sum())
    return {"model": model_name, "level": level, "group": group_name,
            "checkpoint_status": LOAD_STATUS.get(model_name, "STATUS_NOT_CAPTURED"),
            "n_samples": len(successful), "failed_samples": len(group_frame) - len(successful),
            "CER": cer_errors / cer_units, "WER": wer_errors / wer_units,
            "cer_errors": cer_errors, "cer_units": cer_units,
            "wer_errors": wer_errors, "wer_units": wer_units}


all_frames = []
for model_name, _ in MODEL_SPECS:
    path = OUTPUT_DIR / f"predictions_{model_name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Chưa có prediction hoàn chỉnh: {path}")
    frame = pd.read_csv(path)
    if frame["sample_id"].nunique() != len(TEST_RECORDS):
        raise ValueError(f"{model_name}: có {frame['sample_id'].nunique()} sample, cần {len(TEST_RECORDS)}")
    frame["model"] = model_name
    frame["checkpoint_status"] = LOAD_STATUS.get(model_name, "STATUS_NOT_CAPTURED")
    all_frames.append(frame)

predictions = pd.concat(all_frames, ignore_index=True)
atomic_write_csv(predictions, OUTPUT_DIR / "predictions_all_models.csv")
metric_rows = []
for model_name, _ in MODEL_SPECS:
    frame = predictions[predictions["model"].eq(model_name)]
    metric_rows.append(aggregate_metrics(predictions, model_name, "overall", "all", frame))
    for column, level in (("region", "region"), ("province_name", "province")):
        for group, group_frame in frame.groupby(column, dropna=False):
            metric_rows.append(aggregate_metrics(predictions, model_name, level, str(group), group_frame))

metrics = pd.DataFrame(metric_rows)
atomic_write_csv(metrics, OUTPUT_DIR / "metrics_overall_region_province.csv")
atomic_write_csv(metrics[metrics["level"].eq("overall")], OUTPUT_DIR / "metrics_overall.csv")
atomic_write_csv(metrics[metrics["level"].eq("region")], OUTPUT_DIR / "metrics_by_region.csv")
atomic_write_csv(metrics[metrics["level"].eq("province")], OUTPUT_DIR / "metrics_by_province.csv")
display(metrics[metrics["level"].eq("overall")])
print("Kết quả lưu tại:", OUTPUT_DIR)
