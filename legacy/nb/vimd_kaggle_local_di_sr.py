# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # ViMD Kaggle local: DI và SR không streaming
#
# Notebook này tải dataset đã upload bằng `kagglehub`, đọc toàn bộ Parquet/Arrow thành `Dataset` thông thường và chạy các task DI/SR. Không sử dụng `streaming=True`.
#
# Kết quả lưu tại `/kaggle/working/vimd_results.csv`.

# %% [markdown]
# ## Thiết lập Kaggle
#
# Nếu Kaggle báo thiếu package, chạy một cell cài đặt tạm trước cell import rồi restart kernel. Model SR `wav2vec2-base-vi-vlsp2020` cần `pyctcdecode`/`kenlm` để dùng LM; notebook có fallback greedy CTC nếu decoder LM không tương thích.
#
# ```python
# %pip install -q datasets transformers accelerate soundfile pyctcdecode
# %pip install -q https://github.com/kpu/kenlm/archive/master.zip
# ```

# %% [code]
from pathlib import Path
import math
import csv
import io
import json
import os
import random
import struct
import time
import unicodedata
import wave
from collections import Counter

import kagglehub
import numpy as np
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

try:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoFeatureExtractor, AutoModel, AutoModelForAudioClassification, get_linear_schedule_with_warmup
except ImportError as exc:
    raise ImportError("Cài torch và transformers trước khi chạy DI/SR.") from exc


# %% [code]
DATASET_HANDLE = "thanhlamdev/vimd-dataset"
HF_TOKEN = 'hf_REDACTED_TOKEN'

SEED = 20260916

if HF_TOKEN:
    try:
        from huggingface_hub import login
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Hugging Face login: OK (token không hiển thị)")
    except Exception as exc:
        print("Không login Hugging Face; model public vẫn có thể chạy:", repr(exc))
MODEL_HF_KWARGS = {"token": HF_TOKEN} if HF_TOKEN else {}

DATASET_ROOT = Path(kagglehub.dataset_download(DATASET_HANDLE))
OUTPUT_DIR = Path("/kaggle/working/vimd_experiment") if Path("/kaggle/working").exists() else Path.cwd() / "vimd_kaggle_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_DIR = OUTPUT_DIR / "sample_test_30min_audio"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT = OUTPUT_DIR / "di_checkpoints"
RESULTS_CSV = OUTPUT_DIR / "vimd_results.csv"
print("Dataset path:", DATASET_ROOT)
print("Files:", len(list(DATASET_ROOT.rglob("*"))))

# %% [markdown]
# ## 1. Tìm file và load non-streaming
#
# Hàm này hỗ trợ cả dataset Parquet và Arrow. Các file được nhóm theo tên `train`, `valid`/`validation`, `test`.

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
        if split in groups: groups[split].append(str(path))
    return {split: files for split, files in groups.items() if files}

parquet_files = grouped_files("*.parquet")
arrow_files = grouped_files("*.arrow")
if arrow_files:
    print("Loading Arrow non-streaming:", {k: len(v) for k, v in arrow_files.items()})
    dataset = DatasetDict({split: concatenate_datasets([Dataset.from_file(path) for path in files]) for split, files in arrow_files.items()})
elif parquet_files:
    print("Loading Parquet non-streaming:", {k: len(v) for k, v in parquet_files.items()})
    dataset = load_dataset("parquet", data_files=parquet_files, streaming=False)
else:
    raise FileNotFoundError(f"Không tìm thấy Parquet/Arrow trong {DATASET_ROOT}")

dataset_no_decode = {split: part.cast_column("audio", Audio(decode=False)) for split, part in dataset.items()}
print("Rows:", {split: len(part) for split, part in dataset_no_decode.items()})
print("Columns:", dataset_no_decode[next(iter(dataset_no_decode))].column_names)

# %% [markdown]
# ## 2. Audio preprocessing và sample train 4 giờ, valid/test 30 phút

# %% [code]
# Cấu hình sample dùng chung cho DI và SR.
SAMPLE_TRAIN_SECONDS = 4 * 60 * 60      #4h
SAMPLE_VALID_SECONDS = 30 * 60          #30min
SAMPLE_TEST_SECONDS = 30 * 60           #30min
SAMPLE_REGIONS = ("North", "Central", "South")
SAMPLE_TARGETS = {"train": SAMPLE_TRAIN_SECONDS, "valid": SAMPLE_VALID_SECONDS, "test": SAMPLE_TEST_SECONDS}

# %% [code]
def audio_bytes(info):
    if info.get("bytes") is not None:
        return info["bytes"]
    path = info.get("path")
    if path and Path(path).exists():
        return Path(path).read_bytes()
    raise FileNotFoundError(info)

def wav_encoding(data):
    marker = data.find(b"fmt ")
    if marker < 0: return "UNKNOWN"
    size = struct.unpack_from("<I", data, marker + 4)[0]; tag = struct.unpack_from("<H", data, marker + 8)[0]
    if tag == 1: return "PCM"
    if tag == 3: return "IEEE_FLOAT"
    if tag == 0xFFFE and size >= 40:
        subtype = struct.unpack_from("<H", data, marker + 32)[0]; return "IEEE_FLOAT" if subtype == 3 else ("PCM" if subtype == 1 else f"EXTENSIBLE_{subtype}")
    return f"FORMAT_{tag}"

def wav_metadata(data):
    try:
        with wave.open(io.BytesIO(data), "rb") as f:
            return {"sample_rate": f.getframerate(), "channels": f.getnchannels(), "sample_width": f.getsampwidth(), "compression": f.getcomptype(), "encoding": wav_encoding(data), "frames": f.getnframes(), "duration_seconds": f.getnframes() / f.getframerate()}
    except wave.Error:
        import soundfile as sf
        info = sf.info(io.BytesIO(data)); return {"sample_rate": info.samplerate, "channels": info.channels, "sample_width": 4, "compression": info.format, "encoding": wav_encoding(data), "frames": info.frames, "duration_seconds": info.duration}

def wav_float32(data, target_rate=16000):
    if wav_encoding(data) == "IEEE_FLOAT":
        import soundfile as sf
        signal, source_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=False); signal = np.asarray(signal, dtype="float32")
        if signal.ndim > 1: signal = signal.mean(axis=1)
    else:
        with wave.open(io.BytesIO(data), "rb") as f:
            source_rate, channels, width, compression = f.getframerate(), f.getnchannels(), f.getsampwidth(), f.getcomptype(); frames = f.readframes(f.getnframes())
        if width == 2: signal = np.frombuffer(frames, dtype="<i2").astype("float32") / 32768.0
        elif width == 4 and wav_encoding(data) == "IEEE_FLOAT": signal = np.frombuffer(frames, dtype="<f4").astype("float32")
        elif width == 4: signal = np.frombuffer(frames, dtype="<i4").astype("float32") / 2147483648.0
        elif width == 1: signal = (np.frombuffer(frames, dtype="u1").astype("float32") - 128) / 128.0
        else: raise ValueError(f"Unsupported WAV width={width}, compression={compression}")
        if channels > 1: signal = signal.reshape(-1, channels).mean(axis=1)
    if source_rate != target_rate:
        target_length = max(1, round(len(signal) * target_rate / source_rate)); signal = np.interp(np.linspace(0, 1, target_length, endpoint=False), np.linspace(0, 1, len(signal), endpoint=False), signal).astype("float32")
    return signal, target_rate


def _balanced_random_order(split):
    region_indices = {region: [] for region in SAMPLE_REGIONS}
    for idx, row in enumerate(dataset_no_decode[split].select_columns(["region"])):
        region = row.get("region")
        if region in region_indices: region_indices[region].append(idx)
    rng = random.Random(SEED + {"train": 0, "valid": 1, "test": 2}[split])
    for indices in region_indices.values(): rng.shuffle(indices)
    order, pointers = [], {region: 0 for region in SAMPLE_REGIONS}
    while True:
        added = False
        for region in SAMPLE_REGIONS:
            pointer = pointers[region]
            if pointer < len(region_indices[region]): order.append(region_indices[region][pointer]); pointers[region] += 1; added = True
        if not added: break
    return order

def _sample_split(split, target_seconds):
    records, selected_indices, total_seconds, used_regions = [], [], 0.0, set()
    for idx in _balanced_random_order(split):
        row = dataset_no_decode[split][idx]
        try: data = audio_bytes(row["audio"]); meta = wav_metadata(data)
        except Exception: continue
        name = row.get("filename") or f"row_{idx}.wav"; audio_path = None
        if split == "test":
            audio_path = SAMPLE_DIR / f"{len(records):05d}_{split}_{Path(name).name}"; audio_path.write_bytes(data)
        records.append({"sample_id": f"{split}_{len(records):05d}", "split": split, "row_index": idx, "audio_path": str(audio_path) if audio_path else None, "filename": name, "region": row.get("region"), "province_code": row.get("province_code"), "province_name": row.get("province_name"), "speakerID": row.get("speakerID"), "gender": row.get("gender"), "text": row.get("text"), **meta})
        selected_indices.append(idx); total_seconds += meta["duration_seconds"]; used_regions.add(row.get("region"))
        if total_seconds >= target_seconds and set(SAMPLE_REGIONS).issubset(used_regions): break
    return records, selected_indices, total_seconds

sample_records_by_split, sample_indices, sample_seconds = {}, {}, {}
for split, target in SAMPLE_TARGETS.items():
    records, indices, seconds = _sample_split(split, target); sample_records_by_split[split] = records; sample_indices[split] = indices; sample_seconds[split] = seconds
    print(f"{split}: {len(records)} samples, {seconds / 3600:.2f} hours, regions={sorted({r['region'] for r in records})}")

# SR chỉ dùng test 30 phút; DI dùng cả ba sample split.
sample_records = sample_records_by_split["test"]
di_sampled_dataset_no_decode = {split: dataset_no_decode[split].select(sample_indices[split]) for split in SAMPLE_TARGETS}
all_sample_records = [record for split in ("train", "valid", "test") for record in sample_records_by_split[split]]
metadata = {"seed": SEED, "targets_seconds": SAMPLE_TARGETS, "actual_seconds": sample_seconds, "counts": {split: len(records) for split, records in sample_records_by_split.items()}, "region_seconds": {split: {region: sum(r["duration_seconds"] for r in records if r.get("region") == region) for region in SAMPLE_REGIONS} for split, records in sample_records_by_split.items()}, "records": all_sample_records}
metadata_path = OUTPUT_DIR / "vimd_sample_metadata.json"; metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
metadata_fields = ["sample_id", "split", "row_index", "audio_path", "filename", "region", "province_code", "province_name", "speakerID", "gender", "sample_rate", "channels", "sample_width", "compression", "encoding", "frames", "duration_seconds", "text"]
with (OUTPUT_DIR / "vimd_sample_metadata.csv").open("w", newline="", encoding="utf-8") as metadata_file:
    writer = csv.DictWriter(metadata_file, fieldnames=metadata_fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(all_sample_records)
print("Sample metadata:", metadata_path)

# %% [markdown]
# ## 3. Metrics và CSV

# %% [code]
RESULT_COLUMNS = [
    "record_type", "task", "model", "checkpoint", "split", "filename",
    "region", "province_name", "reference", "prediction", "error", "correct",
    "wer", "cer", "f1_macro", "n_samples", "inference_seconds",
]

# %% [code]
def normalize_text(text):
    text = unicodedata.normalize("NFC", str(text or "").lower()); return " ".join("".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in text).split())
def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]
def wer(ref, hyp):
    a, b = normalize_text(ref).split(), normalize_text(hyp).split(); return edit_distance(a, b) / max(1, len(a))
def cer(ref, hyp):
    a, b = normalize_text(ref).replace(" ", ""), normalize_text(hyp).replace(" ", ""); return edit_distance(list(a), list(b)) / max(1, len(a))
def macro_f1(true, pred, labels):
    scores = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(true, pred)); fp = sum(a != label and b == label for a, b in zip(true, pred)); fn = sum(a == label and b != label for a, b in zip(true, pred)); p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0; scores.append(2 * p * r / (p + r) if p + r else 0)
    return sum(scores) / max(1, len(scores))
result_rows = []
def write_results():
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f: writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, extrasaction="ignore"); writer.writeheader(); writer.writerows(result_rows)
    print("CSV:", RESULTS_CSV, "rows:", len(result_rows))

# %% [markdown]
# ## 4. SR/ASR inference
#
# Model được nạp trực tiếp, audio luôn được chuyển mono/16 kHz. Nếu LM decoder lỗi AddedToken/alphabet, notebook dùng greedy CTC.

# %% [code]
RUN_SR = True
MAX_SR_SAMPLES = None
VLSP_USE_KENLM = False
SR_MODEL_SPECS = {
    "wav2vec2-base-vietnamese": "dragonSwing/wav2vec2-base-vietnamese",
    "wav2vec2-base-vietnamese-160h": "khanhld/wav2vec2-base-vietnamese-160h",
    "wav2vec2-base-vietnamese-250h": "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    "wav2vec2-base-vi-vlsp2020": "nguyenvulebinh/wav2vec2-base-vi-vlsp2020",
    "whisper-base": "openai/whisper-base",
}
SR_ACTIVE_MODELS = [
    "wav2vec2-base-vietnamese",
    "wav2vec2-base-vietnamese-160h",
    "wav2vec2-base-vietnamese-250h",
    "wav2vec2-base-vi-vlsp2020",
    "whisper-base",
]

# %% [code]
def load_vlsp_legacy_model(model_id):
    # VLSP2020 có feature_transform tùy chỉnh; dùng class trong model_handling.py
    # để không bỏ qua các trọng số feature_transform.*.
    import importlib.util
    import sys
    from huggingface_hub import hf_hub_download
    handling_path = hf_hub_download(
        repo_id=model_id,
        filename="model_handling.py",
    )
    spec = importlib.util.spec_from_file_location("vlsp_model_handling", handling_path)
    module = importlib.util.module_from_spec(spec)
    # model_handling.py có thể truy cập sys.modules[__name__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.Wav2Vec2ForCTC.from_pretrained(model_id)

def load_sr(model_id):
    # Load one SR checkpoint with an architecture-specific processor.
    import torch
    from transformers import AutoModelForCTC, AutoProcessor
    model_key = model_id.lower()
    is_whisper = "whisper" in model_key
    is_vlsp = "vlsp2020" in model_key

    if is_whisper:
        # Whisper là encoder-decoder, không dùng AutoModelForCTC.
        processor = AutoProcessor.from_pretrained(model_id)
        from transformers import AutoModelForSpeechSeq2Seq
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
        kind, mode = "seq2seq", "default"
    elif is_vlsp:
        # KenLM cũ có thể lỗi alphabet với <s>/</s>. Greedy là mặc định.
        processor = None
        mode = "greedy"
        if globals().get("VLSP_USE_KENLM", False):
            try:
                from transformers import Wav2Vec2ProcessorWithLM
                processor = Wav2Vec2ProcessorWithLM.from_pretrained(model_id)
                mode = "kenlm"
            except Exception as lm_error:
                print(f"{model_id}: bỏ qua KenLM, dùng greedy CTC: {lm_error!r}")
        if processor is None:
            from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_id, extra_special_tokens=[])
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        try:
            model = load_vlsp_legacy_model(model_id)
        except Exception as legacy_error:
            print(f"Không load được class VLSP legacy, dùng AutoModelForCTC: {legacy_error!r}")
            model = AutoModelForCTC.from_pretrained(model_id)
        kind = "ctc"
    else:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCTC.from_pretrained(model_id)
        kind, mode = "ctc", "default"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"Loaded {model_id} | kind={kind} | processor={mode} | device={device}")
    return processor, model, kind, device, mode

def transcribe(path, processor, model, kind, device, mode):
    import torch
    signal, rate = wav_float32(Path(path).read_bytes(), 16000)
    with torch.no_grad():
        if kind == "seq2seq":
            feature_extractor = processor.feature_extractor
            max_samples = getattr(feature_extractor, "n_samples", 30 * 16000)
            predictions = []
            for start in range(0, len(signal), max_samples):
                chunk = signal[start:start + max_samples]
                if len(chunk) == 0:
                    continue
                inputs = processor(
                    chunk, sampling_rate=rate, return_tensors="pt",
                    padding="max_length", truncation=True,
                    max_length=max_samples, return_attention_mask=True,
                )
                if inputs["input_features"].shape[-1] != 3000:
                    raise ValueError(f"Whisper input shape không đúng: {inputs['input_features'].shape}")
                inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
                model.generation_config.max_length = None
                generated_ids = model.generate(**inputs, max_new_tokens=256)
                text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                if text:
                    predictions.append(text)
            return " ".join(predictions).strip()

        inputs = processor(signal, sampling_rate=rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
        logits = model(**inputs).logits
        if mode == "kenlm":
            try:
                return processor.decode(logits[0].cpu().numpy(), beam_width=100).text.strip()
            except Exception as lm_error:
                print("KenLM decode lỗi, fallback greedy:", repr(lm_error))
        return processor.batch_decode(logits.argmax(-1), skip_special_tokens=True)[0].strip()

if RUN_SR:
    for model_name in SR_ACTIVE_MODELS:
        model_id = SR_MODEL_SPECS[model_name]
        processor = model = None
        try:
            processor, model, kind, device, mode = load_sr(model_id); selected = sample_records if MAX_SR_SAMPLES is None else sample_records[:MAX_SR_SAMPLES]; scored = []
            for record in selected:
                start = time.perf_counter()
                try:
                    prediction = transcribe(record["audio_path"], processor, model, kind, device, mode); scored.append({"record_type": "sample", "task": "SR", "model": model_name, "checkpoint": model_id, "split": record["split"], "filename": record["filename"], "region": record["region"], "province_name": record["province_name"], "reference": record["text"], "prediction": prediction, "wer": wer(record["text"], prediction), "cer": cer(record["text"], prediction), "inference_seconds": time.perf_counter() - start})
                except Exception as exc: scored.append({"record_type": "error", "task": "SR", "model": model_name, "checkpoint": model_id, "filename": record["filename"], "prediction": repr(exc), "error": repr(exc)})
            result_rows.extend(scored); valid = [r for r in scored if r["record_type"] == "sample"]
            if valid: result_rows.append({"record_type": "aggregate", "task": "SR", "model": model_name, "checkpoint": model_id, "wer": sum(r["wer"] for r in valid) / len(valid), "cer": sum(r["cer"] for r in valid) / len(valid), "n_samples": len(valid)})
            write_results()
        except Exception as exc:
            print(f"Không load được SR model {model_name}:", repr(exc))
            result_rows.append({"record_type": "error", "task": "SR", "model": model_name, "checkpoint": model_id, "error": repr(exc)})
            write_results()
        finally:
            # Giải phóng checkpoint trước khi load model kế tiếp, tránh OOM GPU.
            import gc, torch
            processor = None
            model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
else: print("RUN_SR=False")

# %% [markdown]
# ## 5. DI checkpoint và macro-F1

# %% [code]
RUN_DI_TRAINING = True
RUN_DI_EVALUATION = True
DI_TASKS_TO_RUN = ["DI_VN_3", "DI_North", "DI_Central", "DI_South", "DI_VN_63"]
DI_TASKS = {
    "DI_VN_3": {"region": None, "label_column": "region", "labels": ["North", "Central", "South"]},
    "DI_North": {"region": "North", "label_column": "province_name", "labels": None},
    "DI_Central": {"region": "Central", "label_column": "province_name", "labels": None},
    "DI_South": {"region": "South", "label_column": "province_name", "labels": None},
    "DI_VN_63": {"region": None, "label_column": "province_name", "labels": None},
}
DI_BACKBONE = "nguyenvulebinh/wav2vec2-base-vi"
DI_LEARNING_RATE = None  # None = chọn theo backbone trong Appendix C.
DI_EPOCHS = 15
DI_BATCH_SIZE = 64  # effective batch size trên 2 GPU.
DI_PER_DEVICE_BATCH_SIZE = 2
DI_GRAD_ACCUMULATION_STEPS = 16
DI_USE_MULTI_GPU = True
DI_USE_FP16 = True
DI_WEIGHT_DECAY = 0.0
DI_WARMUP_RATIO = 0.1
# None = dùng toàn bộ sample train/valid/test.
DI_MAX_TRAIN_SAMPLES = None
DI_VALID_MAX_SAMPLES = None
DI_EVAL_MAX_SAMPLES = None

# %% [code]
def accepts(row, task):
    region = DI_TASKS[task]["region"]
    return region is None or row.get("region") == region


def labels_for(task):
    if DI_TASKS[task]["labels"] is not None:
        return list(DI_TASKS[task]["labels"])
    values = set()
    for stream in dataset_no_decode.values():
        for row in stream.select_columns(["region", "province_name"]):
            if accepts(row, task) and row.get("province_name"):
                values.add(row["province_name"])
    return sorted(values)


def di_learning_rate(model_id):
    name = model_id.lower()
    if "whisper" in name or "large" in name or "xlsr" in name or "xls-r" in name:
        return 6e-5 if "whisper" not in name else 3e-5
    return 3e-5


class DIDialectClassifier(torch.nn.Module):
    """Wav2Vec2 backbone with the paper's two-layer DI classification head."""
    def __init__(self, backbone_id, num_labels, **hf_kwargs):
        super().__init__()
        from transformers import AutoModel
        self.backbone_id = backbone_id
        self.backbone = AutoModel.from_pretrained(backbone_id, **hf_kwargs)
        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("DI backbone không có config.hidden_size; hãy dùng backbone wav2vec2.")
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, num_labels),
        )

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state
        if attention_mask is not None and hasattr(self.backbone, "_get_feature_vector_attention_mask"):
            mask = self.backbone._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            mask = mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}


def collator_factory(extractor):
    def collate(features):
        batch = extractor.pad(
            [{"input_values": f["input_values"]} for f in features],
            padding=True, return_tensors="pt"
        )
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.long)
        return batch
    return collate


def _di_prepare_split(task, split, label2id, cache_dir, shuffle=False):
    part = di_sampled_dataset_no_decode[split].filter(
        lambda row: accepts(row, task),
        cache_file_name=str(cache_dir / f"{split}_filtered_paper.arrow"),
        load_from_cache_file=False,
    )
    if shuffle:
        part = part.shuffle(
            seed=SEED,
            indices_cache_file_name=str(cache_dir / f"{split}_shuffle_paper.arrow"),
            load_from_cache_file=False,
        )
    max_samples = DI_MAX_TRAIN_SAMPLES if split == "train" else DI_VALID_MAX_SAMPLES
    if max_samples is not None:
        part = part.select(range(min(max_samples, len(part))))

    def preprocess(row):
        signal, rate = wav_float32(audio_bytes(row["audio"]), 16000)
        label_name = row[DI_TASKS[task]["label_column"]]
        return {"input_values": signal, "label": label2id[label_name]}

    return part.map(
        preprocess,
        remove_columns=part.column_names,
        cache_file_name=str(cache_dir / f"{split}_preprocessed_paper.arrow"),
        load_from_cache_file=False,
    )


def _di_score_loader(model, loader, labels, device, max_samples=None, amp_enabled=False):
    true, pred = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(**{key: value for key, value in batch.items() if key != "labels"})
            predicted = outputs["logits"].argmax(-1).cpu().tolist()
            actual = batch["labels"].cpu().tolist()
            for actual_id, predicted_id in zip(actual, predicted):
                true.append(labels[actual_id]); pred.append(labels[predicted_id])
                if max_samples is not None and len(true) >= max_samples:
                    return macro_f1(true, pred, labels)
    return macro_f1(true, pred, labels)


def _unwrap_di_model(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def _save_di_checkpoint(model, extractor, checkpoint, labels):
    checkpoint.mkdir(parents=True, exist_ok=True)
    base_model = _unwrap_di_model(model)
    torch.save(base_model.backbone.state_dict(), checkpoint / "backbone_model.bin")
    torch.save(base_model.classifier.state_dict(), checkpoint / "classifier.bin")
    extractor.save_pretrained(checkpoint)
    (checkpoint / "di_config.json").write_text(json.dumps({
        "backbone": base_model.backbone_id,
        "labels": labels,
        "head": "linear-relu-dropout-linear",
        "epochs": DI_EPOCHS,
        "effective_batch_size": DI_BATCH_SIZE,
        "per_device_batch_size": DI_PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": DI_GRAD_ACCUMULATION_STEPS,
        "warmup_ratio": DI_WARMUP_RATIO,
        "multi_gpu": bool(DI_USE_MULTI_GPU),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_di_checkpoint(checkpoint, labels):
    from transformers import AutoFeatureExtractor
    config = json.loads((checkpoint / "di_config.json").read_text(encoding="utf-8"))
    extractor = AutoFeatureExtractor.from_pretrained(checkpoint, **MODEL_HF_KWARGS)
    model = DIDialectClassifier(config["backbone"], len(labels), **MODEL_HF_KWARGS)
    model.backbone.load_state_dict(torch.load(checkpoint / "backbone_model.bin", map_location="cpu"))
    model.classifier.load_state_dict(torch.load(checkpoint / "classifier.bin", map_location="cpu"))
    return extractor, model


def train_di(task):

    labels = labels_for(task)
    label2id = {label: i for i, label in enumerate(labels)}
    extractor = AutoFeatureExtractor.from_pretrained(DI_BACKBONE, **MODEL_HF_KWARGS)
    base_model = DIDialectClassifier(DI_BACKBONE, len(labels), **MODEL_HF_KWARGS)

    gpu_count = torch.cuda.device_count()
    use_multi_gpu = bool(DI_USE_MULTI_GPU and gpu_count >= 2)
    if use_multi_gpu:
        device = torch.device("cuda:0")
        base_model.to(device)
        model = torch.nn.DataParallel(base_model, device_ids=list(range(gpu_count)), output_device=0)
        loader_batch_size = DI_PER_DEVICE_BATCH_SIZE * gpu_count
        print(f"{task}: DataParallel trên {gpu_count} GPU, {DI_PER_DEVICE_BATCH_SIZE} mẫu/GPU, global batch={loader_batch_size}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = base_model.to(device)
        loader_batch_size = DI_PER_DEVICE_BATCH_SIZE
        if DI_USE_MULTI_GPU:
            print(f"{task}: chỉ phát hiện {gpu_count} GPU; chạy single-GPU với batch={loader_batch_size}")

    model.train()
    cache_dir = OUTPUT_DIR / "di_cache" / task
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_set = _di_prepare_split(task, "train", label2id, cache_dir, shuffle=True)
    valid_set = _di_prepare_split(task, "valid", label2id, cache_dir, shuffle=False)
    collator = collator_factory(extractor)
    train_loader = DataLoader(train_set, batch_size=loader_batch_size, collate_fn=collator, num_workers=0, pin_memory=use_multi_gpu)
    valid_loader = DataLoader(valid_set, batch_size=loader_batch_size, collate_fn=collator, num_workers=0, pin_memory=use_multi_gpu)

    learning_rate = DI_LEARNING_RATE or di_learning_rate(DI_BACKBONE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=DI_WEIGHT_DECAY)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / DI_GRAD_ACCUMULATION_STEPS))
    total_steps = max(1, DI_EPOCHS * updates_per_epoch)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=round(total_steps * DI_WARMUP_RATIO),
        num_training_steps=total_steps,
    )
    amp_enabled = bool(DI_USE_FP16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    checkpoint = CHECKPOINT_ROOT / task
    best_f1 = -1.0

    for epoch in range(1, DI_EPOCHS + 1):
        model.train(); optimizer.zero_grad(set_to_none=True); running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                loss = model(**batch)["loss"].mean()
                scaled_loss = loss / DI_GRAD_ACCUMULATION_STEPS
            scaler.scale(scaled_loss).backward()
            should_update = step % DI_GRAD_ACCUMULATION_STEPS == 0 or step == len(train_loader)
            if should_update:
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.detach().cpu())

        valid_f1 = _di_score_loader(model, valid_loader, labels, device, DI_VALID_MAX_SAMPLES, amp_enabled)
        mean_loss = running_loss / max(1, len(train_loader))
        print(f"{task} epoch {epoch}/{DI_EPOCHS} loss={mean_loss:.4f} valid_macro_f1={valid_f1:.4f}")
        if valid_f1 >= best_f1:
            best_f1 = valid_f1
            _save_di_checkpoint(model, extractor, checkpoint, labels)

    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return checkpoint, labels


trained = {}
if RUN_DI_TRAINING:
    for task in DI_TASKS_TO_RUN:
        trained[task] = train_di(task)
else:
    print("RUN_DI_TRAINING=False")

# %% [code]
def evaluate_di(task, checkpoint, labels):
    import torch
    from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if (checkpoint / "di_config.json").exists():
        extractor, model = _load_di_checkpoint(checkpoint, labels)
    else:
        extractor = AutoFeatureExtractor.from_pretrained(checkpoint, **MODEL_HF_KWARGS)
        model = AutoModelForAudioClassification.from_pretrained(checkpoint, **MODEL_HF_KWARGS)
    model.to(device).eval(); true, pred, rows = [], [], []
    for row in di_sampled_dataset_no_decode["test"]:
        if not accepts(row, task):
            continue
        if DI_EVAL_MAX_SAMPLES is not None and len(true) >= DI_EVAL_MAX_SAMPLES:
            break
        signal, rate = wav_float32(audio_bytes(row["audio"]), 16000)
        inputs = extractor(signal, sampling_rate=rate, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items() if hasattr(value, "to")}
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
            predicted = labels[int(logits.argmax(-1).item())]
        actual = row[DI_TASKS[task]["label_column"]]
        true.append(actual); pred.append(predicted)
        rows.append({"record_type": "sample", "task": task, "model": "DI", "checkpoint": str(checkpoint), "split": "test", "filename": row.get("filename"), "region": row.get("region"), "province_name": row.get("province_name"), "reference": actual, "prediction": predicted, "correct": actual == predicted})
    score = macro_f1(true, pred, labels)
    rows.append({"record_type": "aggregate", "task": task, "model": "DI", "checkpoint": str(checkpoint), "split": "test", "f1_macro": score, "n_samples": len(true)})
    result_rows.extend(rows)
    if "write_results" in globals(): write_results()
    else: write_csv()
    print(task, "macro-F1=", score, "n=", len(true))

if RUN_DI_EVALUATION:
    for task in DI_TASKS_TO_RUN:
        checkpoint = CHECKPOINT_ROOT / task
        if task in trained:
            checkpoint, labels = trained[task]
        elif (checkpoint / "di_config.json").exists():
            labels = json.loads((checkpoint / "di_config.json").read_text(encoding="utf-8"))["labels"]
        elif (checkpoint / "labels.json").exists():
            labels = json.loads((checkpoint / "labels.json").read_text(encoding="utf-8"))
        else:
            print("Bỏ qua", task, "- chưa có checkpoint"); continue
        evaluate_di(task, checkpoint, labels)
else:
    print("RUN_DI_EVALUATION=False")
if "write_results" in globals(): write_results()
else: write_csv()
print("Hoàn tất:", RESULTS_CSV)

# %% [markdown]
# ## Cách chạy
#
# 1. Attach dataset `thanhlamdev/vimd-dataset` vào Kaggle hoặc dùng đúng `DATASET_HANDLE`.
# 2. Chạy cell load dataset; kiểm tra `Rows` và `Columns`.
# 3. Chạy SR với `RUN_SR=True`; xem WER/CER trong CSV.
# 4. Muốn tạo DI checkpoint, đặt `RUN_DI_TRAINING=True` và bật GPU Kaggle.
# 5. Checkpoint lưu tại `/kaggle/working/vimd_experiment/di_checkpoints/<task>`.
# 6. CSV duy nhất: `/kaggle/working/vimd_experiment/vimd_results.csv`.
