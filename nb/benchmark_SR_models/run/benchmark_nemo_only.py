# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # ViMD benchmark — NeMo-only
#
# Chạy Canary-Qwen và Parakeet; tách khỏi môi trường Transformers. NeMo dùng đúng sample test và thứ tự trong `sample_metadata.json` của benchmark Transformers.
#

# %% [code]
# Cài môi trường NeMo-only trước khi import nemo.
# NumPy 1.26.4 là bản tương thích ổn định với NeMo trên image Kaggle hiện tại.
# Kaggle đã cung cấp torch/torchaudio CUDA; không cài đè để tránh mất CUDA.
%pip install -q -U --force-reinstall "numpy==1.26.4"
%pip install -q -U datasets huggingface_hub soundfile
%pip install -q -U "nemo_toolkit[asr,tts]"
# Đảm bảo NeMo không kéo NumPy trở lại 2.x.
%pip install -q -U --no-deps "numpy==1.26.4"

import importlib.metadata as importlib_metadata
print("numpy:", importlib_metadata.version("numpy"))
print("nemo_toolkit:", importlib_metadata.version("nemo_toolkit"))
print("Đã cài môi trường NeMo-only. Bắt buộc Restart Session trước khi chạy tiếp.")


# %% [code]
from pathlib import Path
import csv
import json
import os
import tempfile
import time
import traceback
import unicodedata
import gc

import pandas as pd
import kagglehub
import numpy as np
import soundfile as sf
import torch
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi


# %% [code]
# Benchmark configuration.
DATASET_HANDLE = "thanhlamdev/vimd-dataset"
SEED = 20260916
MAX_TEST_ROWS = None          # None = toàn bộ sample trong metadata Transformers.
# Manifest sample cố định do benchmark Transformers sinh ra và đã upload lên Kaggle.
TRANSFORMERS_SAMPLE_METADATA = "/kaggle/input/datasets/lqdung/dataset-sample/sample_metadata.json"
RESUME = False                # Đặt True chỉ khi CSV được tạo từ cùng sample_metadata.json.
RETRY_ERRORS = False          # True = chạy lại các dòng lỗi trong CSV cũ.
SLEEP_BETWEEN_REQUESTS = 0.0
OUTPUT_DIR = Path("/kaggle/working/vimd_nemo_test_benchmark") if Path("/kaggle/working").exists() else Path.cwd() / "vimd_nemo_test_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_CSV = OUTPUT_DIR / "vimd_nemo_test_predictions.csv"
AGGREGATE_CSV = OUTPUT_DIR / "vimd_nemo_test_aggregate.csv"
METADATA_JSON = OUTPUT_DIR / "vimd_nemo_test_metadata.json"

TOP7_MODELS = [
    "nvidia/canary-qwen-2.5b",
    "nvidia/parakeet-tdt-0.6b-v2",
]
MODEL_RANKS = dict(zip(TOP7_MODELS, [15, 21]))
MODEL_BACKEND = {
    TOP7_MODELS[0]: "nemo_salm",
    TOP7_MODELS[1]: "nemo_asr",
}

RESULT_COLUMNS = [
    "record_type", "model", "backend", "split", "row_index", "filename",
    "region", "province_name", "reference", "prediction", "wer", "cer",
    "inference_seconds", "error",
]

def get_hf_token():
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN").strip()
    except Exception:
        return os.environ.get("HUGGINGFACE_TOKEN", "").strip()

HF_TOKEN = get_hf_token()
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
HF_API = HfApi(token=HF_TOKEN) if HF_TOKEN else None
print("HF token present:", bool(HF_TOKEN))
print("Models:", TOP7_MODELS)
print("Transformers sample metadata:", TRANSFORMERS_SAMPLE_METADATA)
print("Output:", OUTPUT_DIR)


# %% [code]
def split_of(path):
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    for split in ("train", "valid", "validation", "test"):
        if name.startswith(split + "-") or f"-{split}-" in name or split in parts:
            return "valid" if split == "validation" else split
    return None


def grouped_files(extension):
    groups = {"train": [], "valid": [], "test": []}
    for path in sorted(DATASET_ROOT.rglob(extension)):
        split = split_of(path)
        if split in groups:
            groups[split].append(str(path))
    return {split: files for split, files in groups.items() if files}

DATASET_ROOT = Path(kagglehub.dataset_download(DATASET_HANDLE))
parquet_files = grouped_files("*.parquet")
arrow_files = grouped_files("*.arrow")
if arrow_files.get("test"):
    test_dataset = concatenate_datasets([Dataset.from_file(path) for path in arrow_files.get("test", [])])
elif parquet_files.get("test"):
    test_dataset = load_dataset("parquet", data_files={"test": parquet_files["test"]}, streaming=False)["test"]
else:
    raise FileNotFoundError(f"Không tìm thấy file test trong {DATASET_ROOT}")

test_dataset = test_dataset.cast_column("audio", Audio(decode=False))
print("Dataset:", DATASET_ROOT)
print("Test rows:", len(test_dataset))
print("Columns:", test_dataset.column_names)

SAMPLE_METADATA_PATH = Path(TRANSFORMERS_SAMPLE_METADATA)
if not SAMPLE_METADATA_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy sample metadata tại: {SAMPLE_METADATA_PATH}")
sample_payload = json.loads(SAMPLE_METADATA_PATH.read_text(encoding="utf-8"))
sample_metadata_records = sample_payload.get("records", sample_payload if isinstance(sample_payload, list) else [])
if not sample_metadata_records:
    raise ValueError(f"Sample metadata không có records: {SAMPLE_METADATA_PATH}")
if MAX_TEST_ROWS is not None:
    sample_metadata_records = sample_metadata_records[:min(MAX_TEST_ROWS, len(sample_metadata_records))]

# Map original_dataset_index -> row trong test split. Không random/sample lại.
sample_records = []
for sample_position, metadata_row in enumerate(sample_metadata_records):
    original_index = metadata_row.get("original_dataset_index", metadata_row.get("row_index"))
    if original_index is None:
        raise KeyError(f"Thiếu original_dataset_index/row_index ở sample {sample_position}")
    original_index = int(original_index)
    if original_index < 0 or original_index >= len(test_dataset):
        raise IndexError(f"original_dataset_index={original_index} vượt test split ({len(test_dataset)} rows)")
    dataset_row = test_dataset[original_index]
    sample_records.append({
        "sample_index": int(metadata_row.get("sample_index", sample_position)),
        "original_dataset_index": original_index,
        "split": metadata_row.get("split", "test"),
        "filename": metadata_row.get("filename") or dataset_row.get("filename") or f"row_{original_index}.wav",
        "region": metadata_row.get("region", dataset_row.get("region")),
        "province_name": metadata_row.get("province_name", dataset_row.get("province_name")),
        "text": metadata_row.get("text", dataset_row.get("text", "")),
        "audio": dataset_row["audio"],
        "duration_seconds": metadata_row.get("duration_seconds"),
    })
print("Transformers sample metadata:", SAMPLE_METADATA_PATH)
print("Sample rows used by NeMo:", len(sample_records))
print("Sample duration (minutes):", round(sum(float(r.get("duration_seconds") or 0) for r in sample_records) / 60, 2))


# %% [code]
def audio_bytes(audio_info):
    if audio_info.get("bytes") is not None:
        return audio_info["bytes"]
    path = audio_info.get("path")
    if path and Path(path).exists():
        return Path(path).read_bytes()
    raise FileNotFoundError(audio_info)


AUDIO_CACHE_DIR = Path("/kaggle/temp/vimd_audio_cache")
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
NEMO_AUDIO_CACHE_DIR = Path("/kaggle/temp/vimd_nemo_audio")
NEMO_AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def materialize_audio(audio_info, row_index):
    path = audio_info.get("path")
    if path and Path(path).exists():
        return str(path)
    target = AUDIO_CACHE_DIR / f"{row_index:08d}.wav"
    if not target.exists():
        target.write_bytes(audio_bytes(audio_info))
    return str(target)


def normalize_audio_for_nemo(audio_path):
    """Write a mono/16-kHz WAV for NeMo models.

    NeMo's SALM/ASR APIs receive file paths and expect mono 16-kHz
    audio. Do not write normalized files under /kaggle/input.
    """
    source = Path(audio_path)
    target = NEMO_AUDIO_CACHE_DIR / f"{source.stem}_mono16k.wav"
    if target.exists():
        return str(target)

    audio, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    sf.write(target, audio.astype(np.float32), 16000, subtype="PCM_16")
    return str(target)


def load_audio_16k(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return np.asarray(audio, dtype=np.float32), 16000


def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text or "").lower())
    return " ".join("".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text).split())


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def wer(reference, prediction):
    ref = normalize_text(reference).split()
    hyp = normalize_text(prediction).split()
    return edit_distance(ref, hyp) / max(1, len(ref))


def cer(reference, prediction):
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(prediction).replace(" ", "")
    return edit_distance(list(ref), list(hyp)) / max(1, len(ref))


def extract_prediction(result):
    if isinstance(result, (list, tuple)):
        return extract_prediction(result[0]) if result else ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text", result.get("generated_text", ""))).strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    return str(result).strip()


MODEL_KWARGS = {"token": HF_TOKEN} if HF_TOKEN else {}
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


def model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_inputs(inputs, device, dtype=None):
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    if dtype is not None and hasattr(inputs, "items"):
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and torch.is_floating_point(value):
                inputs[key] = value.to(dtype)
    return inputs


def build_bad_words_ids(tokenizer):
    eos_ids = tokenizer.eos_token_id
    keep_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids or [])
    bad_ids = set(tokenizer.all_special_ids) - keep_ids
    bad_ids.update(
        token_id for token, token_id in tokenizer.get_added_vocab().items()
        if token.startswith("<") and token.endswith(">") and token_id not in keep_ids
    )
    return [[token_id] for token_id in sorted(bad_ids)]


def load_local_model(model_id):
    backend = MODEL_BACKEND[model_id]
    if False and backend == "transformers_qwen3":
        # Dùng class cụ thể để tránh AutoModelForMultimodalLM lazy-import kéo theo
        # các backend multimodal không cần thiết và che mất lỗi dependency gốc.
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, **MODEL_KWARGS)
        model = Qwen3ASRForConditionalGeneration.from_pretrained(
            model_id, dtype=TORCH_DTYPE, device_map="auto", **MODEL_KWARGS
        ).eval()
        return {"backend": backend, "processor": processor, "model": model}
    if False and backend == "transformers_ark_remote_code":
        # ARK-ASR-3B currently resolves to Edge0/ARK-ASR-3B on the Hub.
        source_id = "Edge0/ARK-ASR-3B" if model_id == "AutoArk-AI/ARK-ASR-3B" else model_id
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
        processor = AutoProcessor.from_pretrained(source_id, trust_remote_code=True, **MODEL_KWARGS)
        tokenizer = AutoTokenizer.from_pretrained(source_id, trust_remote_code=True, **MODEL_KWARGS)
        model = AutoModelForCausalLM.from_pretrained(
            source_id, trust_remote_code=True, dtype=TORCH_DTYPE,
            device_map="auto", attn_implementation="sdpa", **MODEL_KWARGS
        ).eval()
        return {"backend": backend, "processor": processor, "tokenizer": tokenizer, "model": model}
    if False and backend == "transformers_cohere":
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, **MODEL_KWARGS)
        model = CohereAsrForConditionalGeneration.from_pretrained(
            model_id, trust_remote_code=True, dtype=TORCH_DTYPE,
            device_map="auto", **MODEL_KWARGS
        ).eval()
        return {"backend": backend, "processor": processor, "model": model}
    if False and backend == "transformers_granite":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        processor = AutoProcessor.from_pretrained(model_id, **MODEL_KWARGS)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, dtype=TORCH_DTYPE, device_map="auto", **MODEL_KWARGS
        ).eval()
        return {"backend": backend, "processor": processor, "model": model}
    if backend == "nemo_salm":
        from nemo.collections.speechlm2.models import SALM
        model = SALM.from_pretrained(model_id).eval()
        if torch.cuda.is_available():
            model = model.cuda()
            if torch.cuda.get_device_capability()[0] < 8:
                model = model.half()
        return {"backend": backend, "model": model}
    if backend == "nemo_asr":
        import nemo.collections.asr as nemo_asr
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_id).eval()
        if torch.cuda.is_available():
            model = model.cuda()
            if torch.cuda.get_device_capability()[0] < 8:
                model = model.half()
        return {"backend": backend, "model": model}
    raise KeyError(f"Unsupported backend: {backend}")


def transcribe_qwen(bundle, audio_path):
    processor, model = bundle["processor"], bundle["model"]
    inputs = processor.apply_transcription_request(audio=audio_path)
    inputs = move_inputs(inputs, model_device(model), getattr(model, "dtype", TORCH_DTYPE))
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    try:
        return processor.decode(generated_ids, return_format="transcription_only")[0].strip()
    except Exception:
        return extract_prediction(processor.batch_decode(generated_ids, skip_special_tokens=True))


def transcribe_ark(bundle, audio_path):
    processor, tokenizer, model = bundle["processor"], bundle["tokenizer"], bundle["model"]
    conversation = [{"role": "user", "content": [
        {"type": "audio", "path": audio_path},
        {"type": "text", "text": "Please transcribe this audio."},
    ]}]
    inputs = processor.apply_chat_template(
        conversation, add_generation_prompt=True, return_tensors="pt",
        sampling_rate=16000, audio_padding="longest",
        text_kwargs={"padding": "longest"}, audio_max_length=40 * 16000,
    )
    inputs = move_inputs(inputs, model_device(model), getattr(model, "dtype", TORCH_DTYPE))
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, do_sample=False, max_new_tokens=256,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bad_words_ids=build_bad_words_ids(tokenizer),
        )
    return extract_prediction(tokenizer.batch_decode(output_ids, skip_special_tokens=True))


def transcribe_cohere(bundle, audio_path):
    processor, model = bundle["processor"], bundle["model"]
    audio, sample_rate = load_audio_16k(audio_path)
    inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt", language="vi")
    inputs = move_inputs(inputs, model_device(model), getattr(model, "dtype", TORCH_DTYPE))
    chunk_index = inputs.get("audio_chunk_index") if hasattr(inputs, "get") else None
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256)
    try:
        return extract_prediction(processor.decode(output_ids, skip_special_tokens=True, audio_chunk_index=chunk_index, language="vi"))
    except TypeError:
        return extract_prediction(processor.decode(output_ids, skip_special_tokens=True))


def transcribe_granite(bundle, audio_path):
    processor, model = bundle["processor"], bundle["model"]
    audio, sample_rate = load_audio_16k(audio_path)
    tokenizer = processor.tokenizer
    chat = [{"role": "user", "content": "<|audio|>transcribe the speech with proper punctuation and capitalization."}]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, audio, device=model_device(model), return_tensors="pt")
    inputs = move_inputs(inputs, model_device(model), getattr(model, "dtype", TORCH_DTYPE))
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False, num_beams=1)
    generated_ids = output_ids[:, inputs["input_ids"].shape[-1]:]
    return extract_prediction(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))


def transcribe_canary(bundle, audio_path):
    model = bundle["model"]
    audio_path = normalize_audio_for_nemo(audio_path)
    prompt = [[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [audio_path]}]]
    with torch.inference_mode():
        answer_ids = model.generate(prompts=prompt, max_new_tokens=128)
    return extract_prediction(model.tokenizer.ids_to_text(answer_ids[0].detach().cpu()))


def transcribe_parakeet(bundle, audio_path):
    audio_path = normalize_audio_for_nemo(audio_path)
    result = bundle["model"].transcribe([audio_path])
    return extract_prediction(result)


def transcribe_local(bundle, audio_path):
    fn = {
        "transformers_qwen3": transcribe_qwen,
        "transformers_ark_remote_code": transcribe_ark,
        "transformers_cohere": transcribe_cohere,
        "transformers_granite": transcribe_granite,
        "nemo_salm": transcribe_canary,
        "nemo_asr": transcribe_parakeet,
    }[bundle["backend"]]
    return fn(bundle, audio_path)


def unload_local_model(bundle):
    bundle.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# %% [code]
# Run full test benchmark with one local model loaded at a time.
completed = set()
if RESUME and RESULTS_CSV.exists():
    with RESULTS_CSV.open("r", newline="", encoding="utf-8") as f:
        for old_row in csv.DictReader(f):
            if not RETRY_ERRORS or not old_row.get("error"):
                completed.add((old_row.get("model"), old_row.get("row_index")))
    print("Resume rows:", len(completed))

file_mode = "a" if RESUME else "w"
write_header = file_mode == "w" or not RESULTS_CSV.exists()
with RESULTS_CSV.open(file_mode, newline="", encoding="utf-8") as result_file:
    writer = csv.DictWriter(result_file, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
    if write_header:
        writer.writeheader()
    for model_id in TOP7_MODELS:
        model_started = time.perf_counter()
        print(f"\nLoading {model_id} ({MODEL_BACKEND[model_id]})...")
        try:
            bundle = load_local_model(model_id)
        except Exception as exc:
            error = repr(exc)
            print(f"LOAD ERROR {model_id}: {error}")
            traceback.print_exc()
            for row in sample_records:
                row_index = int(row["sample_index"])
                key = (model_id, str(row_index))
                if key in completed:
                    continue
                writer.writerow({"record_type": "error", "model": model_id, "backend": MODEL_BACKEND[model_id], "split": "test", "row_index": row_index, "filename": row.get("filename"), "error": error})
            result_file.flush()
            continue
        try:
            for row in sample_records:
                row_index = int(row["sample_index"])
                key = (model_id, str(row_index))
                if key in completed:
                    continue
                filename = row.get("filename") or f"row_{row_index}.wav"
                reference = row.get("text", "")
                started = time.perf_counter()
                prediction, error = "", ""
                try:
                    # Use the original dataset row recorded in sample_metadata.json.
                    audio_path = materialize_audio(row["audio"], int(row["original_dataset_index"]))
                    prediction = transcribe_local(bundle, audio_path)
                except Exception as exc:
                    error = repr(exc)
                elapsed = time.perf_counter() - started
                writer.writerow({
                    "record_type": "sample", "model": model_id,
                    "backend": MODEL_BACKEND[model_id], "split": "test",
                    "row_index": row_index, "filename": filename,
                    "region": row.get("region"), "province_name": row.get("province_name"),
                    "reference": reference, "prediction": prediction,
                    "wer": wer(reference, prediction) if not error else "",
                    "cer": cer(reference, prediction) if not error else "",
                    "inference_seconds": elapsed, "error": error,
                })
                result_file.flush()
                if error:
                    print(f"ERROR {model_id} row={row_index}: {error}")
                elif row_index % 10 == 0:
                    print(f"{model_id} row={row_index} elapsed={elapsed:.2f}s")
        finally:
            unload_local_model(bundle)
            print(f"Finished {model_id} in {(time.perf_counter() - model_started) / 60:.1f} min")

print("Predictions:", RESULTS_CSV)


# %% [code]
# Aggregate WER/CER and write benchmark metadata.
import pandas as pd

predictions = pd.read_csv(RESULTS_CSV)
valid = predictions[(predictions["record_type"] == "sample") & (predictions["error"].fillna("") == "")].copy()
if not valid.empty:
    aggregate = valid.groupby("model", as_index=False).agg(
        n_success=("prediction", "size"),
        wer=("wer", "mean"),
        cer=("cer", "mean"),
        inference_seconds_mean=("inference_seconds", "mean"),
        inference_seconds_total=("inference_seconds", "sum"),
    )
else:
    aggregate = pd.DataFrame(columns=["model", "n_success", "wer", "cer", "inference_seconds_mean", "inference_seconds_total"])
errors = predictions[predictions["error"].fillna("") != ""].groupby("model").size().rename("n_errors")
aggregate = aggregate.merge(errors, on="model", how="left").fillna({"n_errors": 0})
aggregate["n_errors"] = aggregate["n_errors"].astype(int)
aggregate.to_csv(AGGREGATE_CSV, index=False)
metadata = {
    "dataset_handle": DATASET_HANDLE,
    "dataset_root": str(DATASET_ROOT),
    "split": "test",
    "n_test_rows": len(sample_records),
    "sample_metadata_source": str(SAMPLE_METADATA_PATH),
    "sample_metadata_selection": {
        "selection_method": sample_payload.get("selection_method"),
        "seed": sample_payload.get("seed"),
        "source_n_samples": sample_payload.get("n_samples"),
        "source_total_duration_seconds": sample_payload.get("total_duration_seconds"),
    },
    "models": TOP7_MODELS,
    "leaderboard_ranks": MODEL_RANKS,
    "backend": MODEL_BACKEND,
    "hf_token_present": bool(HF_TOKEN),
    "local_inference": True,
    "results_csv": str(RESULTS_CSV),
    "aggregate_csv": str(AGGREGATE_CSV),
    "seed": SEED,
}
METADATA_JSON.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(aggregate.sort_values("wer"))
print("Aggregate:", AGGREGATE_CSV)
print("Metadata:", METADATA_JSON)


# %% [markdown]
# ## NeMo-only
#
# Canary yêu cầu NeMo mới và GPU CUDA; Parakeet là model NeMo ASR. NumPy được pin 1.26.4 để tránh lỗi ABI trên Kaggle.
#
