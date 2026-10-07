# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Kiểm tra checkpoint Qwen3-ASR + AdaLoRA
#
# Notebook này kiểm tra khả năng nạp `epoch_1_adapter`, `epoch_2_adapter`, `epoch_3_adapter`, rồi tính CER/WER so với Qwen3-ASR pretrained trên **cùng sample của split test** trong `sample_metadata.json`.
#
# Mặc định chỉ chạy 20 audio để kiểm tra nhanh. Đặt `MAX_SAMPLES = None` để chạy toàn bộ split test (484 audio). Có thể đọc adapter từ Kaggle Model đã attach làm input: đặt từng đường dẫn variation/version trong `CHECKPOINT_PATHS`. Nếu Kaggle hiển thị thư mục adapter là `empty`, notebook sẽ báo thiếu file và bỏ qua checkpoint đó.
#

# %% [markdown]
# ## 1. Import và kiểm tra môi trường

# %% [code]
# Nếu thiếu thư viện, cài ở cell riêng rồi restart session; tránh force-reinstall torch/numpy trên Kaggle.
import gc
import json
import os
import random
import re
import time
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
from peft import PeftModel

print("torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("transformers:", __import__("transformers").__version__)
print("peft:", __import__("peft").__version__)


# %% [markdown]
# ## 2. Cấu hình dataset, checkpoint và sample

# %% [code]
# Có thể điền đường dẫn cụ thể nếu tự động tìm không đúng.
DATASET_ROOT = None       # thư mục chứa sample_metadata.json + train/valid/test audio hoặc ZIP
CHECKPOINT_ROOT = None    # dùng nếu cả ba epoch_*_adapter nằm trong cùng một thư mục
CHECKPOINT_PATHS = {
    # Đường dẫn trực tiếp tới variation/version của Kaggle Model (nếu upload riêng từng epoch).
    # Ví dụ: "/kaggle/input/<model-slug>/pytorch/epoch-1/1"
    "epoch_1_adapter": None,
    "epoch_2_adapter": None,
    "epoch_3_adapter": None,
}
MODEL_REVISION = None    # nên điền model_revision từ run_config.json của lần fine-tune để so sánh đúng base
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
EVAL_SPLIT = "test"       # chỉ dùng test để đánh giá khả năng tổng quát hóa
MAX_SAMPLES = 20          # smoke test nhanh; None = toàn bộ split test
SEED = 42
MAX_NEW_TOKENS = 768
OUTPUT_DIR = Path("/kaggle/working/qwen_checkpoint_check") if Path("/kaggle/working").exists() else Path.cwd() / "qwen_checkpoint_check"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Đọc secret nếu có; không in token ra output.
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = None

def locate_manifest_root(configured):
    if configured:
        root = Path(configured)
        if (root / "sample_metadata.json").is_file():
            return root
        raise FileNotFoundError(f"Không thấy sample_metadata.json trong {root}")
    roots = [Path("/kaggle/input"), Path.cwd()]
    found = []
    for root in roots:
        if root.exists():
            found.extend(root.rglob("sample_metadata.json"))
    found = sorted(set(found))
    if len(found) != 1:
        raise RuntimeError("Cần đặt DATASET_ROOT; tìm thấy các manifest: " + str(found[:10]))
    return found[0].parent

def locate_checkpoint_root(configured):
    if configured:
        root = Path(configured)
        if not root.is_dir():
            raise FileNotFoundError(f"CHECKPOINT_ROOT không tồn tại: {root}")
        return root
    candidates = []
    for root in [Path("/kaggle/working"), Path("/kaggle/input")]:
        if root.exists():
            candidates.extend(p for p in root.rglob("vimd_qwen_adalora") if p.is_dir())
    candidates = sorted(set(candidates))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        return None
    raise RuntimeError("Có nhiều thư mục vimd_qwen_adalora; hãy đặt CHECKPOINT_ROOT: " + str(candidates))

DATASET_ROOT = locate_manifest_root(DATASET_ROOT)
if CHECKPOINT_ROOT is not None:
    CHECKPOINT_ROOT = locate_checkpoint_root(CHECKPOINT_ROOT)
elif not any(CHECKPOINT_PATHS.values()):
    CHECKPOINT_ROOT = locate_checkpoint_root(None)
manifest = json.loads((DATASET_ROOT / "sample_metadata.json").read_text(encoding="utf-8"))
all_rows = [r for r in manifest["records"] if r["split"] == EVAL_SPLIT]
if not all_rows:
    raise ValueError(f"Manifest không có records split={EVAL_SPLIT!r}")
if MAX_SAMPLES is not None and len(all_rows) > MAX_SAMPLES:
    rng = np.random.default_rng(SEED)
    chosen = sorted(rng.choice(len(all_rows), size=MAX_SAMPLES, replace=False).tolist())
    eval_rows = [all_rows[i] for i in chosen]
else:
    eval_rows = all_rows

print("Dataset:", DATASET_ROOT)
print("Checkpoint root:", CHECKPOINT_ROOT)
print("Split:", EVAL_SPLIT, "| selected:", len(eval_rows), "/", len(all_rows))
print("Output:", OUTPUT_DIR)


# %% [markdown]
# ## 3. Xác minh adapter có file và kiểm tra tensor rank 0

# %% [code]
def adapter_files(adapter_dir):
    if adapter_dir is None or not adapter_dir.is_dir():
        return []
    expected = [adapter_dir / "adapter_config.json", adapter_dir / "adapter_model.safetensors", adapter_dir / "adapter_model.bin"]
    return [p for p in expected if p.is_file()]

def inspect_zero_shape_tensors(adapter_dir):
    safe_path = adapter_dir / "adapter_model.safetensors"
    bin_path = adapter_dir / "adapter_model.bin"
    try:
        if safe_path.is_file():
            from safetensors import safe_open
            zero_keys = []
            with safe_open(str(safe_path), framework="pt", device="cpu") as f:
                for key in f.keys():
                    shape = tuple(f.get_slice(key).get_shape())
                    if any(dim == 0 for dim in shape):
                        zero_keys.append((key, shape))
            return zero_keys
        if bin_path.is_file():
            state = torch.load(bin_path, map_location="cpu", weights_only=True)
            return [(key, tuple(value.shape)) for key, value in state.items()
                    if torch.is_tensor(value) and any(dim == 0 for dim in value.shape)]
    except Exception as exc:
        return [("INSPECTION_ERROR", repr(exc))]
    return []

checkpoint_info = {}
for epoch in (1, 2, 3):
    name = f"epoch_{epoch}_adapter"
    configured_path = CHECKPOINT_PATHS.get(name)
    adapter_dir = (Path(configured_path) if configured_path else
                   CHECKPOINT_ROOT / name if CHECKPOINT_ROOT else None)
    files = adapter_files(adapter_dir)
    zero_tensors = inspect_zero_shape_tensors(adapter_dir) if files else []
    ready = bool(adapter_dir and (adapter_dir / "adapter_config.json").is_file()
                 and any((adapter_dir / filename).is_file() for filename in ("adapter_model.safetensors", "adapter_model.bin")))
    checkpoint_info[name] = {"path": adapter_dir, "ready": ready, "files": files, "zero_tensors": zero_tensors}
    print(f"{name}:", "READY" if ready else "MISSING/EMPTY", "| files:", [p.name for p in files],
          "| zero-shape tensors:", len(zero_tensors))
    if zero_tensors:
        print("  examples:", zero_tensors[:4])

training_config = {}
if CHECKPOINT_ROOT:
    run_config_path = CHECKPOINT_ROOT / "run_config.json"
    if run_config_path.is_file():
        training_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        print("Training config revision:", training_config.get("model_revision"))
        print("Training library versions:", training_config.get("transformers_version"), training_config.get("peft_version"))
        MODEL_REVISION = MODEL_REVISION or training_config.get("model_revision")
        MODEL_REVISION = MODEL_REVISION or training_config.get("model_revision")
else:
    print("Chưa tìm thấy checkpoint root; pretrained vẫn có thể được đánh giá.")


# %% [markdown]
# ## 4. Đọc audio giống pipeline fine-tuning và định nghĩa CER/WER

# %% [code]
import io
import zipfile
from math import gcd

def audio_candidates(row):
    rel = Path(row["audio_path"])
    if rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"audio_path không an toàn: {row.get('sample_id')}")
    return [DATASET_ROOT / row["split"] / rel, DATASET_ROOT / rel]

ZIP_MODE = all((DATASET_ROOT / f"{s}.zip").is_file() for s in ("train", "valid", "test"))
if not ZIP_MODE:
    missing = [r["sample_id"] for r in eval_rows if not any(p.is_file() for p in audio_candidates(r))]
    if missing:
        raise FileNotFoundError(f"Không tìm được audio trong ZIP hoặc thư mục; ví dụ: {missing[:5]}")

def load_waveform(row):
    if ZIP_MODE:
        with zipfile.ZipFile(DATASET_ROOT / row["archive"], "r") as zf:
            raw = zf.read(row["audio_path"])
    else:
        path = next(p for p in audio_candidates(row) if p.is_file())
        raw = path.read_bytes()
    audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0]
    if sr != 16000:
        g = gcd(int(sr), 16000)
        audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Audio lỗi: {row['sample_id']}")
    return audio

def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def corpus_cer_wer(frame):
    cer_err = cer_den = wer_err = wer_den = 0
    for row in frame.itertuples(index=False):
        ref, hyp = normalize_text(row.reference), normalize_text(row.prediction)
        cer_err += edit_distance(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
        cer_den += max(1, len(ref.replace(" ", "")))
        rt, ht = ref.split(), hyp.split()
        wer_err += edit_distance(rt, ht)
        wer_den += max(1, len(rt))
    return cer_err / cer_den, wer_err / wer_den, cer_err, cer_den, wer_err, wer_den

print("Audio layout:", "ZIP" if ZIP_MODE else "directories")
sample_audio = load_waveform(eval_rows[0])
print("Audio check:", eval_rows[0]["sample_id"], sample_audio.shape, sample_audio.dtype, "16 kHz mono")


# %% [markdown]
# ## 5. Chạy pretrained và từng adapter, ghi nhận lỗi nạp/inference

# %% [code]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL_DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32
MODEL_REVISION = training_config.get("model_revision") if CHECKPOINT_ROOT else None
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN, revision=MODEL_REVISION)

def load_base_model():
    model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_ID, token=HF_TOKEN, revision=MODEL_REVISION, dtype=MODEL_DTYPE,
        low_cpu_mem_usage=True,
    )
    model.to(DEVICE).eval()
    return model

def transcribe(model, audio):
    inputs = processor.apply_transcription_request(audio=audio, language="Vietnamese")
    moved = {}
    dtype = next(model.parameters()).dtype
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device=DEVICE, dtype=dtype) if value.is_floating_point() else value.to(DEVICE)
        else:
            moved[key] = value
    with torch.inference_mode():
        ids = model.generate(**moved, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    prompt_len = int(moved["input_ids"].shape[-1]) if "input_ids" in moved else 0
    generated = ids[:, prompt_len:] if prompt_len and ids.shape[-1] > prompt_len else ids
    decoded = processor.decode(generated[0], return_format="transcription_only")
    if isinstance(decoded, (list, tuple)):
        decoded = decoded[0]
    if isinstance(decoded, dict):
        decoded = decoded.get("text", decoded.get("transcription", str(decoded)))
    return str(decoded).strip()

prediction_rows = []
summary_rows = []

def evaluate_one(name, adapter_dir=None, zero_tensor_count=0):
    model = None
    local_rows = []
    try:
        base = load_base_model()
        if adapter_dir is not None:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
            model.to(DEVICE).eval()
            save_warnings = [str(w.message) for w in caught]
            if save_warnings:
                print(name, "load warnings:", save_warnings[:2])
        else:
            model = base
        for i, row in enumerate(eval_rows, 1):
            start = time.perf_counter()
            try:
                audio = load_waveform(row)
                pred = transcribe(model, audio)
                error = ""
            except Exception as exc:
                pred, error = "", repr(exc)
            local_rows.append({
                "sample_id": row["sample_id"], "model": name,
                "split": row["split"], "filename": row.get("filename"),
                "region": row.get("region"), "province_name": row.get("province_name"),
                "reference": row["text"], "prediction": pred, "error": error,
                "inference_seconds": time.perf_counter() - start,
            })
            if i % 10 == 0 or i == len(eval_rows):
                print(f"{name}: {i}/{len(eval_rows)}")
        prediction_rows.extend(local_rows)
        failed = [r for r in local_rows if r["error"]]
        if failed:
            status = "INFERENCE_ERRORS"
            cer = wer = np.nan
            print(name, "inference errors:", failed[:2])
        elif len(local_rows) != len(eval_rows):
            status = "INCOMPLETE"
            cer = wer = np.nan
        else:
            cer, wer, *_ = corpus_cer_wer(pd.DataFrame(local_rows))
            status = "OK_ZERO_SHAPE_WARNING" if zero_tensor_count else "OK"
        summary_rows.append({"model": name, "status": status, "n_samples": len(local_rows),
                             "zero_shape_tensors": zero_tensor_count, "CER": cer, "WER": wer,
                             "error": ""})
    except Exception as exc:
        message = repr(exc)
        print(name, "LOAD ERROR:", message)
        summary_rows.append({"model": name, "status": "LOAD_ERROR", "n_samples": 0,
                             "zero_shape_tensors": zero_tensor_count, "CER": np.nan, "WER": np.nan,
                             "error": message})
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

evaluate_one("pretrained")
for epoch in (1, 2, 3):
    name = f"epoch_{epoch}_adapter"
    info = checkpoint_info[name]
    if not info["ready"]:
        summary_rows.append({"model": name, "status": "MISSING_OR_EMPTY_CHECKPOINT", "n_samples": 0,
                             "zero_shape_tensors": 0, "CER": np.nan, "WER": np.nan,
                             "error": f"Không thấy adapter_config.json + adapter_model file tại {info['path']}"})
        continue
    evaluate_one(name, info["path"], len(info["zero_tensors"]))

predictions = pd.DataFrame(prediction_rows)
summary = pd.DataFrame(summary_rows)
predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False, encoding="utf-8-sig")
summary.to_csv(OUTPUT_DIR / "metrics_cer_wer.csv", index=False, encoding="utf-8-sig")
display(summary)
print("Predictions:", OUTPUT_DIR / "predictions.csv")
print("Metrics:", OUTPUT_DIR / "metrics_cer_wer.csv")


# %% [markdown]
# ## Đọc kết quả
#
# - `OK`: adapter nạp được và có dự đoán cho toàn bộ sample đã chọn.
# - `OK_ZERO_SHAPE_WARNING`: nạp/chạy được nhưng file adapter có tensor rank 0; cần xem đây là kết quả chẩn đoán, chưa nên khẳng định checkpoint đầy đủ.
# - `MISSING_OR_EMPTY_CHECKPOINT`: thiếu `adapter_config.json` hoặc `adapter_model.safetensors` / `adapter_model.bin` (phù hợp với trường hợp Kaggle báo thư mục `empty`).
# - `LOAD_ERROR` hoặc `INFERENCE_ERRORS`: xem cột `error`; CER/WER không được tính như kết quả hợp lệ khi thiếu dự đoán.
#
# Đặt `MAX_SAMPLES = None` để tính metric trên toàn bộ 484 audio test. CER/WER dùng cùng chuẩn hóa như notebook fine-tuning: Unicode NFC, chữ thường, bỏ dấu câu và chuẩn hóa khoảng trắng; từ địa phương không bị đổi sang từ phổ thông.
#
