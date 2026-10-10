# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Benchmark Qwen3-ASR trên 3 audio ViMD và 1 audio super hard
#
# Notebook tìm `30_0309.wav`, `59_0313.wav`, `37_0338.wav` trong split test ViMD, đồng thời đọc instance `super_hard.wav` từ dataset Kaggle `super_hard_audio`. Sau đó chạy pretrained và adapter epoch 1–3 trên cả bốn audio. Reference của ViMD lấy từ cột `text`; reference super hard lấy từ `sample_metadata.json`. Kết quả gồm transcript, CER/WER, thời gian và CSV trong `/kaggle/working/qwen3_asr_3_audio_benchmark/`.
#
# Trên Kaggle, attach dataset ViMD của `thanhlamdev`, dataset `super_hard_audio` và model output chứa ba adapter. Super hard là nguồn ngoài ViMD, nên chỉ số của nó được báo riêng, không gộp vào chỉ số ba audio ViMD. Đây là benchmark định tính, không thay thế kết quả toàn bộ tập test.

# %% [markdown]
# ## 1. Cài dependencies nếu môi trường chưa có
#
# Notebook không tự cài hay force-reinstall Torch/NumPy. Kaggle cần có `transformers==5.17.0`, `peft==0.21.1`, `accelerate`, `safetensors`, `soundfile`, `scipy`, `pandas`. Nếu thiếu hoặc phiên bản khác, bỏ comment và chạy riêng cell này; sau đó restart session rồi chạy notebook từ đầu.

# %% [code]
# Chỉ chạy khi cần; không cài lại torch/numpy.
# %pip install -q "transformers==5.17.0" "peft==0.21.1" "accelerate>=1.15.0" safetensors soundfile scipy pandas

# %% [markdown]
# ## 2. Import và Kaggle runtime

# %% [code]
import gc
import importlib.metadata as importlib_metadata
import io
import json
import os
import re
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import Audio, Dataset
from scipy.signal import resample_poly
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from peft import PeftModel

print("torch:", torch.__version__)
print("transformers:", importlib_metadata.version("transformers"))
print("peft:", importlib_metadata.version("peft"))
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("CẢNH BÁO: đang chạy CPU, inference sẽ chậm.")

# %% [markdown]
# ## 3. Cấu hình audio, checkpoints và model revision

# %% [code]
# Dataset đã attach trong Kaggle Input.
DATASET_ROOT = Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0")
SUPER_HARD_ROOT = None  # None: tự tìm sample_metadata.json của super_hard_audio trong /kaggle/input
CHECKPOINT_ROOT = None  # thư mục cha chứa epoch_1_adapter, epoch_2_adapter, epoch_3_adapter
CHECKPOINT_PATHS = {
    "epoch_1_adapter": None,
    "epoch_2_adapter": None,
    "epoch_3_adapter": None,
}

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
MODEL_REVISION = "bcd2b5b7f32b480ab5790554cfa8347f246a14f3"
LANGUAGE = "Vietnamese"
MAX_NEW_TOKENS = 768
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
OUTPUT_DIR = Path("/kaggle/working/qwen3_asr_3_audio_benchmark")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EXTRACTED_AUDIO_DIR = OUTPUT_DIR / "extracted_test_audio"
EXTRACTED_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Ba audio ViMD; cell kế tiếp sẽ thêm một audio super hard từ dataset riêng.
TEST_CASES = [
    {"difficulty": "easy", "filename": "30_0309.wav", "split": "test"},
    {"difficulty": "easy", "filename": "59_0313.wav", "split": "test"},
    {"difficulty": "hard", "filename": "37_0338.wav", "split": "test"},
]

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None
print("Dataset:", DATASET_ROOT)
print("Model:", MODEL_ID, "| revision:", MODEL_REVISION)
print("Output:", OUTPUT_DIR)

# %% [markdown]
# ## 4. Trích xuất ba bản ghi từ split test của dataset Kaggle và xác thực adapter
#
# Cell này đọc metadata của các shard test, tìm đúng ba tên file, lấy audio/transcript trực tiếp từ bản ghi gốc, rồi ghi WAV tạm vào `/kaggle/working`. Không cần upload riêng ba audio.

# %% [code]
test_shards = sorted(DATASET_ROOT.rglob("vi_md_dataset-test-*.arrow"))
if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(f"Không thấy dataset Kaggle tại {DATASET_ROOT}")
if not test_shards:
    raise FileNotFoundError(f"Không thấy shard test .arrow dưới {DATASET_ROOT}")

targets = {Path(case["filename"]).stem: case for case in TEST_CASES}
matches = {name: [] for name in targets}
split_offset = 0
for shard_path in test_shards:
    part = Dataset.from_file(str(shard_path))
    if "filename" not in part.column_names or "audio" not in part.column_names:
        raise KeyError(f"Shard thiếu filename/audio: {shard_path}; columns={part.column_names}")
    metadata_columns = [
        col for col in ("filename", "text", "region", "province_code", "province_name", "speakerID", "gender")
        if col in part.column_names
    ]
    metadata = part.select_columns(metadata_columns)
    for local_index, row in enumerate(metadata):
        stem = Path(str(row.get("filename") or "")).stem
        if stem in targets:
            matches[stem].append({
                "part": part,
                "shard_path": shard_path,
                "shard_row_index": local_index,
                "split_row_index": split_offset + local_index,
                **row,
            })
    split_offset += len(part)

missing = [name for name, found in matches.items() if not found]
duplicates = {name: found for name, found in matches.items() if len(found) > 1}
if missing or duplicates:
    duplicate_locations = {
        name: [(str(item["shard_path"]), item["shard_row_index"]) for item in found]
        for name, found in duplicates.items()
    }
    raise RuntimeError(f"Thiếu filename: {missing}; filename trùng: {duplicate_locations}")

def audio_bytes_from_dataset(audio_info, shard_path):
    if audio_info.get("bytes") is not None:
        return audio_info["bytes"]
    source_path = audio_info.get("path")
    if source_path:
        source_path = Path(source_path)
        if not source_path.is_absolute():
            source_path = shard_path.parent / source_path
        if source_path.is_file():
            return source_path.read_bytes()
    raise FileNotFoundError(f"Không đọc được audio trong {shard_path}: {audio_info}")

for stem, case in targets.items():
    source = matches[stem][0]
    audio_part = source["part"].cast_column("audio", Audio(decode=False))
    audio_info = audio_part[int(source["shard_row_index"])]["audio"]
    raw_audio = audio_bytes_from_dataset(audio_info, source["shard_path"])
    output_path = EXTRACTED_AUDIO_DIR / Path(str(source["filename"])).name
    output_path.write_bytes(raw_audio)

    case["audio_path"] = output_path
    case["dataset_source"] = "ViMD"
    case["reference"] = source.get("text")
    case["region"] = source.get("region")
    case["province_name"] = source.get("province_name")
    case["province_code"] = source.get("province_code")
    case["speakerID"] = source.get("speakerID")
    case["source_shard"] = str(source["shard_path"])
    case["shard_row_index"] = int(source["shard_row_index"])
    case["split_row_index"] = int(source["split_row_index"])
    if not case["reference"]:
        raise ValueError(f"Dataset không có transcript cho {case['filename']}")

    audio_info = sf.info(str(output_path))
    print(f"{case['difficulty']} | {case['filename']} | {case['region']} / {case['province_name']} | "
          f"{audio_info.duration:.2f}s | {output_path}")

extracted_manifest = [
    {k: str(v) if isinstance(v, Path) else v for k, v in case.items() if k != "audio"}
    for case in TEST_CASES
]
(OUTPUT_DIR / "extracted_test_manifest.json").write_text(
    json.dumps(extracted_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)

def find_checkpoint(name):
    configured = CHECKPOINT_PATHS.get(name)
    if configured:
        path = Path(configured)
        if path.is_file():
            path = path.parent
    elif CHECKPOINT_ROOT:
        path = Path(CHECKPOINT_ROOT) / name
    else:
        roots = [Path("/kaggle/input"), Path("/kaggle/working")]
        matches = []
        for root in roots:
            if root.exists():
                matches.extend(p.parent for p in root.rglob("adapter_config.json") if p.parent.name == name)
        matches = sorted({p.resolve() for p in matches})
        if len(matches) != 1:
            raise RuntimeError(f"Tìm thấy {len(matches)} checkpoint cho {name}: {matches}. Hãy đặt CHECKPOINT_PATHS.")
        path = matches[0]
    if not (path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Thiếu adapter_config.json: {path}")
    if not (path / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Thiếu adapter_model.safetensors: {path}")
    return path.resolve()

CHECKPOINTS = {f"epoch_{i}_adapter": find_checkpoint(f"epoch_{i}_adapter") for i in (1, 2, 3)}
for case in TEST_CASES:
    info = sf.info(str(case["audio_path"]))
    print(case["difficulty"], case["filename"], "->", case["audio_path"],
          f"| {info.duration:.2f}s | {info.samplerate}Hz | {info.channels} channel(s)")
for name, path in CHECKPOINTS.items():
    config = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError(f"{name} ghi base model không khớp: {config.get('base_model_name_or_path')}")
    print(name, "->", path, "| PEFT type:", config.get("peft_type"), "| rank patterns:", len(config.get("rank_pattern", {})))

# %% [markdown]
# ## 5. Thêm instance super hard từ Kaggle Input
#
# Cell này tìm `sample_metadata.json` của dataset `super_hard_audio` và đọc trực tiếp WAV đã được Kaggle giải nén tại `test/test/audio/`. Không mở ZIP hoặc giải nén lần nữa. Có thể đặt `SUPER_HARD_ROOT` ở cell config nếu tự tìm không được. Audio này là nguồn ngoài ViMD và được đánh giá riêng.

# %% [code]
def is_super_hard_manifest(path):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return any(row.get("sample_id") == "test_super_hard_000000"
               for row in payload.get("records", []))

if SUPER_HARD_ROOT is None:
    metadata_paths = sorted(path for path in Path("/kaggle/input").rglob("sample_metadata.json")
                            if is_super_hard_manifest(path))
    if len(metadata_paths) != 1:
        raise RuntimeError(f"Cần đúng 1 metadata super hard, tìm thấy {len(metadata_paths)}: {metadata_paths}. "
                           "Hãy đặt SUPER_HARD_ROOT trong cell config.")
    super_hard_root = metadata_paths[0].parent
else:
    super_hard_root = Path(SUPER_HARD_ROOT)
    metadata_paths = [super_hard_root / "sample_metadata.json"]
    if not metadata_paths[0].is_file():
        raise FileNotFoundError(f"Thiếu sample_metadata.json: {metadata_paths[0]}")

super_hard_manifest = json.loads(metadata_paths[0].read_text(encoding="utf-8"))
super_hard_records = [row for row in super_hard_manifest.get("records", [])
                      if row.get("sample_id") == "test_super_hard_000000"]
if len(super_hard_records) != 1:
    raise ValueError(f"Metadata phải chứa đúng 1 instance super hard: {metadata_paths[0]}")
super_hard_record = super_hard_records[0]
if not str(super_hard_record.get("text") or "").strip():
    raise ValueError("Transcript super hard trong metadata đang rỗng.")
relative_audio = Path(super_hard_record["audio_path"])
if (relative_audio.is_absolute() or ".." in relative_audio.parts or
        len(relative_audio.parts) < 3 or relative_audio.parts[:2] != ("test", "audio")):
    raise ValueError(f"audio_path trong metadata không đúng cấu trúc test/audio: {relative_audio}")
# Kaggle đã giải nén test.zip vào thư mục test; audio_path lại bắt đầu bằng test/audio.
super_hard_audio_path = super_hard_root / "test" / relative_audio
if not super_hard_audio_path.is_file():
    raise FileNotFoundError(f"Không thấy WAV đã giải nén: {super_hard_audio_path}")
super_hard_info = sf.info(str(super_hard_audio_path))
if super_hard_info.frames <= 0 or super_hard_info.samplerate != 16000 or super_hard_info.channels != 1:
    raise ValueError(f"WAV super hard phải là mono 16 kHz: {super_hard_audio_path}")

TEST_CASES[:] = [case for case in TEST_CASES if case.get("dataset_source") != "external_video"]
TEST_CASES.append({
    "difficulty": "super_hard", "dataset_source": "external_video",
    "filename": super_hard_record["filename"], "split": super_hard_record["split"],
    "audio_path": super_hard_audio_path, "reference": super_hard_record["text"],
    "region": super_hard_record.get("region"),
    "province_name": super_hard_record.get("province_name"),
    "sample_id": super_hard_record["sample_id"],
    "source_manifest": str(metadata_paths[0]),
})
extracted_manifest = [{k: str(v) if isinstance(v, Path) else v for k, v in case.items()
                       if k != "audio"} for case in TEST_CASES]
(OUTPUT_DIR / "extracted_test_manifest.json").write_text(
    json.dumps(extracted_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
print("Super hard:", super_hard_audio_path, f"| {super_hard_info.duration:.2f}s")
print("Tổng audio benchmark:", len(TEST_CASES), "(3 ViMD + 1 external)")

# %% [markdown]
# ## 6. Đọc audio và tính CER/WER

# %% [code]
def load_audio(path):
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0]
    if sample_rate != 16000:
        import math
        divisor = math.gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Audio không hợp lệ: {path}")
    return audio

for case in TEST_CASES:
    case["audio"] = load_audio(case["audio_path"])
    print(case["filename"], case["audio"].shape, "samples @ 16 kHz")

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

def edit_distance(ref, hyp):
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]

def error_counts(reference, prediction):
    ref = normalize_text(reference)
    hyp = normalize_text(prediction)
    ref_chars, hyp_chars = ref.replace(" ", ""), hyp.replace(" ", "")
    ref_words, hyp_words = ref.split(), hyp.split()
    return {
        "cer_errors": edit_distance(ref_chars, hyp_chars), "cer_units": max(1, len(ref_chars)),
        "wer_errors": edit_distance(ref_words, hyp_words), "wer_units": max(1, len(ref_words)),
    }

# %% [markdown]
# ## 7. Load model và benchmark lần lượt
#
# Mỗi lượt nạp một base Qwen mới; sau đó gắn adapter (nếu có), chạy greedy decoding trên bốn audio. CER/WER được báo riêng cho ba mẫu ViMD và mẫu super hard ngoài ViMD. Cảnh báo AdaLoRA được lưu cùng kết quả.

# %% [code]
processor = AutoProcessor.from_pretrained(
    MODEL_ID, token=HF_TOKEN, revision=MODEL_REVISION
)

def load_base_model():
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE).eval()
    return model

def transcribe(model, audio):
    inputs = processor.apply_transcription_request(audio=audio, language=LANGUAGE)
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
    decoded = processor.decode(generated[0], return_format="transcription_only")
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if isinstance(decoded, dict):
        decoded = decoded.get("text", decoded.get("transcription", str(decoded)))
    return str(decoded).strip(), elapsed

def count_zero_shape_tensors(adapter_dir):
    from safetensors import safe_open
    path = adapter_dir / "adapter_model.safetensors"
    zero = []
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        for key in checkpoint.keys():
            shape = tuple(checkpoint.get_slice(key).get_shape())
            if any(dim == 0 for dim in shape):
                zero.append((key, shape))
    return zero

all_prediction_rows = []
summary_rows = []
EVALUATION_SETS = {"vimd_3": "ViMD", "super_hard_1": "external_video"}
model_runs = [("pretrained", None)] + [(name, path) for name, path in CHECKPOINTS.items()]
for model_name, adapter_path in model_runs:
    print(f"\n===== {model_name} =====")
    model = None
    load_warnings = []
    zero_shapes = []
    rows = []
    try:
        base = load_base_model()
        if adapter_path is None:
            model = base
        else:
            zero_shapes = count_zero_shape_tensors(adapter_path)
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False)
            model.to(DEVICE).eval()
            load_warnings = [str(w.message) for w in captured]
            rank_warnings = [w for w in load_warnings if "rank_pattern" in w.lower()]
            if rank_warnings:
                print("rank_pattern warning:", rank_warnings[0][:600])
            print("zero-shape tensors in saved adapter:", len(zero_shapes))

        for case in TEST_CASES:
            try:
                prediction, elapsed = transcribe(model, case["audio"])
                metrics = error_counts(case["reference"], prediction)
                error = ""
            except Exception as exc:
                prediction, elapsed, error = "", np.nan, repr(exc)
                metrics = {"cer_errors": np.nan, "cer_units": np.nan,
                           "wer_errors": np.nan, "wer_units": np.nan}
            row = {
                "model": model_name,
                "dataset_source": case["dataset_source"],
                "sample_id": case.get("sample_id"),
                "difficulty": case["difficulty"],
                "filename": case["filename"],
                "split": case["split"],
                "region": case["region"],
                "province_name": case["province_name"],
                "reference": case["reference"],
                "prediction": prediction,
                **metrics,
                "CER": metrics["cer_errors"] / metrics["cer_units"],
                "WER": metrics["wer_errors"] / metrics["wer_units"],
                "inference_seconds": elapsed,
                "error": error,
            }
            rows.append(row)
            print(f"{case['difficulty']} | {case['filename']} | {elapsed:.2f}s")
            print("REF :", case["reference"])
            print("HYP :", prediction if not error else f"ERROR {error}")

        rank_pattern_warning = any("rank_pattern" in w.lower() for w in load_warnings)
        for evaluation_set, source_name in EVALUATION_SETS.items():
            scoped = [r for r in rows if r["dataset_source"] == source_name]
            successful = [r for r in scoped if not r["error"]]
            if successful:
                cer = sum(r["cer_errors"] for r in successful) / sum(r["cer_units"] for r in successful)
                wer = sum(r["wer_errors"] for r in successful) / sum(r["wer_units"] for r in successful)
            else:
                cer = wer = np.nan
            status = "OK" if len(successful) == len(scoped) and scoped else "INFERENCE_ERRORS"
            if status == "OK" and (zero_shapes or rank_pattern_warning):
                status = "OK_WITH_ADALORA_WARNING"
            summary_rows.append({
                "model": model_name, "evaluation_set": evaluation_set,
                "status": status, "n_samples": len(successful),
                "zero_shape_tensors": len(zero_shapes), "rank_pattern_warning": rank_pattern_warning,
                "CER": cer, "WER": wer,
                "mean_inference_seconds": float(np.mean([r["inference_seconds"] for r in successful])) if successful else np.nan,
                "error": " | ".join(r["error"] for r in scoped if r["error"]),
            })
        all_prediction_rows.extend(rows)
    except Exception as exc:
        message = repr(exc)
        print("LOAD ERROR:", message)
        for evaluation_set in EVALUATION_SETS:
            summary_rows.append({
                "model": model_name, "evaluation_set": evaluation_set,
                "status": "LOAD_ERROR", "n_samples": 0,
                "zero_shape_tensors": len(zero_shapes),
                "rank_pattern_warning": any("rank_pattern" in w.lower() for w in load_warnings),
                "CER": np.nan, "WER": np.nan,
                "mean_inference_seconds": np.nan, "error": message,
            })
    finally:
        if model is not None:
            del model
        if "base" in locals():
            del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

predictions = pd.DataFrame(all_prediction_rows)
summary = pd.DataFrame(summary_rows)
summary["CER_percent"] = 100 * summary["CER"]
summary["WER_percent"] = 100 * summary["WER"]
baseline_rows = summary.loc[summary["model"].eq("pretrained")].set_index("evaluation_set")
summary["CER_delta_vs_pretrained_pp"] = 100 * (summary["CER"] - summary["evaluation_set"].map(baseline_rows["CER"]))
summary["WER_delta_vs_pretrained_pp"] = 100 * (summary["WER"] - summary["evaluation_set"].map(baseline_rows["WER"]))
predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUTPUT_DIR / "summary_cer_wer.csv", index=False, encoding="utf-8-sig")
display(summary[["evaluation_set", "model", "status", "n_samples", "zero_shape_tensors", "rank_pattern_warning",
                 "CER_percent", "WER_percent", "CER_delta_vs_pretrained_pp",
                 "WER_delta_vs_pretrained_pp", "mean_inference_seconds", "error"]].round(4))
print("Predictions:", OUTPUT_DIR / "predictions.csv")
print("Summary:", OUTPUT_DIR / "summary_cer_wer.csv")

# %% [markdown]
# ## 8. Cách đọc kết quả
#
# - `predictions.csv` có transcript và CER/WER của từng audio.
# - `summary_cer_wer.csv` tách `vimd_3` và `super_hard_1`; không trộn nguồn ngoài ViMD vào benchmark ViMD. Corpus CER/WER là tổng lỗi chia tổng số ký tự/từ tham chiếu sau chuẩn hóa Unicode NFC, chữ thường, bỏ dấu câu và gộp khoảng trắng.
# - `OK_WITH_ADALORA_WARNING` nghĩa là suy luận đủ số audio của nhóm nhưng adapter có tensor rank 0 đã prune hoặc có cảnh báo `rank_pattern`; không coi riêng cảnh báo đó là lỗi inference.
# - Mỗi nhóm chỉ có 3 hoặc 1 audio; dùng để demo/debug, không kết luận thống kê về mức cải thiện.
