# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Qwen3-ASR 1.7B + AdaLoRA trên toàn bộ ViMD — Kaggle T4 x2
#
# Notebook fine-tune Qwen3-ASR bằng AdaLoRA trên dataset gốc Kaggle `thanhlamdev/vimd-dataset` (ViMD), không dùng mini-dataset 31,25 giờ. Toàn bộ split train được dùng để train trong **2 epoch**; toàn bộ valid dùng chọn adapter, toàn bộ test dùng so sánh pretrained/adapter.
#
# Notebook đọc audio trực tiếp từ Arrow/Parquet của dataset. Không yêu cầu `sample_metadata.json`, ZIP mini-dataset, và không giải nén/copy audio sang `/kaggle/working`. Nếu dataset được attach sẵn vào Kaggle Input, hãy bật Internet để KaggleHub resolve dataset; có thể đặt `DATASET_ROOT` thủ công nếu cần.
#
# Chạy tuần tự từ trên xuống. Cần bật **2 GPU Tesla T4** và đặt Kaggle Secret `HF_TOKEN` nếu model yêu cầu xác thực.
#
# Luồng: nạp split gốc và tạo manifest chỉ chứa metadata + chỉ mục shard → kiểm tra/decode audio → smoke test DDP hai GPU → train 2 epoch → chọn adapter theo valid CER → so sánh pretrained/adapter trên toàn bộ test. Kết quả ghi trong `/kaggle/working/vimd_qwen_adalora_full_2epoch`.
#
#
# Có thêm bản Python dạng percent-format để xem/chỉnh cell như script: `nb/qwen3_asr_adalora_kaggle_t4x2.py`.
#

# %% [code]
# Chỉ cài các thư viện model/training và đọc dataset cần thiết.
# Giữ nguyên torch, torchvision, torchaudio và numpy có sẵn trên Kaggle.
%pip install -q "transformers==5.17.0" "peft==0.21.1" "accelerate==1.15.0" "soundfile>=0.12" "scipy>=1.10" "datasets>=3.0" "kagglehub>=0.3" "huggingface_hub>=0.30"
print("Cài đặt xong. Nếu Kaggle yêu cầu restart kernel, hãy restart rồi chạy lại từ Cell 2.")


# %% [code]
import os
import sys
import json
import math
import time
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
import peft
import accelerate
import soundfile as sf
import scipy

print("Python:", sys.version.split()[0])
print("torch:", torch.__version__, "| transformers:", transformers.__version__)
print("peft:", peft.__version__, "| accelerate:", accelerate.__version__)
print("numpy:", np.__version__, "| scipy:", scipy.__version__, "| soundfile:", sf.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("Cần bật GPU trong Kaggle Settings.")
gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print("GPU count:", torch.cuda.device_count(), "|", gpu_names)
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"Notebook này cấu hình DDP cho đúng 2 GPU; hiện thấy {torch.cuda.device_count()}.")
if not all("T4" in name.upper() for name in gpu_names):
    print("Cảnh báo: cấu hình này được thiết kế cho 2 Tesla T4; đang thấy GPU khác.")
disk = shutil.disk_usage("/kaggle/working")
print("Working free GB:", round(disk.free / 1024**3, 2))
HF_TOKEN = None
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    print("Đã đọc HF_TOKEN từ Kaggle Secrets (không hiển thị giá trị).")
except Exception:
    print("Không thấy HF_TOKEN Secret; tiếp tục không token (phù hợp với model public).")

# %% [code]
# Cấu hình fine-tuning toàn bộ ViMD.
DATASET_HANDLE = "thanhlamdev/vimd-dataset"
DATASET_ROOT = None  # Có thể đặt thành /kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0
MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
SEED = 42
EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.03
MAX_NEW_TOKENS = 768
NUM_WORKERS = 2
INIT_R = 16
TARGET_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj"]
OUTPUT_DIR = Path("/kaggle/working/vimd_qwen_adalora_full_2epoch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG = {
    "dataset_handle": DATASET_HANDLE, "model_id": MODEL_ID, "seed": SEED, "epochs": EPOCHS,
    "per_device_batch_size": PER_DEVICE_BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
    "warmup_ratio": WARMUP_RATIO, "max_new_tokens": MAX_NEW_TOKENS,
    "init_r": INIT_R, "target_r": TARGET_R, "lora_alpha": LORA_ALPHA,
    "lora_dropout": LORA_DROPOUT, "target_modules": TARGET_MODULES,
    "precision": "fp16", "gradient_checkpointing": True,
    "world_size": 2, "output_dir": str(OUTPUT_DIR),
}
(OUTPUT_DIR / "run_config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(CONFIG, ensure_ascii=False, indent=2))


# %% [code]
# Nạp các shard Arrow/Parquet của ViMD gốc và tạo manifest metadata nhẹ.
import kagglehub
from datasets import Audio, Dataset, concatenate_datasets, load_dataset

def resolve_dataset_root(configured_root, handle):
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"DATASET_ROOT không tồn tại: {root}")
        return root
    attached_candidates = [
        Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0"),
        Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset"),
    ]
    for candidate in attached_candidates:
        if candidate.is_dir() and (any(candidate.rglob("*.arrow")) or any(candidate.rglob("*.parquet"))):
            return candidate.resolve()
    # Với dataset đã attach trong Kaggle, KaggleHub trả về thư mục được mount/cache.
    return Path(kagglehub.dataset_download(handle)).resolve()

DATASET_ROOT = resolve_dataset_root(DATASET_ROOT, DATASET_HANDLE)

def split_of(path):
    path = Path(path)
    parts = [part.lower() for part in path.parts]
    name = path.name.lower()
    for split in ("train", "valid", "validation", "test"):
        if name.startswith(split + "-") or f"-{split}-" in name or split in parts:
            return "valid" if split == "validation" else split
    return None

def grouped_files(suffix):
    groups = {"train": [], "valid": [], "test": []}
    for file_path in sorted(DATASET_ROOT.rglob(f"*{suffix}")):
        split = split_of(file_path)
        if split in groups:
            groups[split].append(file_path.resolve())
    return {split: files for split, files in groups.items() if files}

arrow_files = grouped_files(".arrow")
parquet_files = grouped_files(".parquet")
if arrow_files:
    SOURCE_FORMAT = "arrow"
    split_sources = arrow_files
elif parquet_files:
    SOURCE_FORMAT = "parquet"
    split_sources = parquet_files
else:
    raise FileNotFoundError(f"Không tìm thấy shard Arrow/Parquet trong {DATASET_ROOT}")
if not all(split in split_sources for split in ("train", "valid", "test")):
    raise ValueError(f"Dataset cần đủ train/valid/test; tìm thấy: {list(split_sources)}")
print("Dataset root:", DATASET_ROOT)
print("Format:", SOURCE_FORMAT, "| shard counts:", {s: len(v) for s, v in split_sources.items()})

records_by_split = {s: [] for s in ("train", "valid", "test")}
source_datasets = {}
manifest_records = []

def read_audio_metadata(audio_info):
    if not isinstance(audio_info, dict):
        raise TypeError(f"Audio field cần dict khi dùng Audio(decode=False), nhận {type(audio_info)}")
    raw = audio_info.get("bytes")
    audio_path = audio_info.get("path")
    if raw is not None:
        info = sf.info(io.BytesIO(raw))
    elif audio_path:
        candidate = Path(audio_path)
        if not candidate.is_absolute():
            candidate = DATASET_ROOT / candidate
        info = sf.info(str(candidate))
    else:
        raise ValueError("Audio row không có bytes hoặc path")
    return float(info.duration), str(audio_path or "")

for split, files in split_sources.items():
    for file_path in files:
        if SOURCE_FORMAT == "arrow":
            source_ds = Dataset.from_file(str(file_path)).cast_column("audio", Audio(decode=False))
            source_file = file_path
        else:
            # Convert parquet to the Hugging Face Arrow cache once. Worker ranks then
            # memory-map that cache rather than repeatedly scanning parquet.
            source_ds = load_dataset("parquet", data_files={"train": [str(file_path)]}, split="train")
            source_ds = source_ds.cast_column("audio", Audio(decode=False))
            cache_files = source_ds.cache_files
            if len(cache_files) != 1:
                raise RuntimeError(f"Không xác định được Arrow cache duy nhất cho {file_path}: {cache_files}")
            source_file = Path(cache_files[0]["filename"]).resolve()
        source_datasets[str(source_file)] = source_ds
        for source_row_index in range(len(source_ds)):
            row = source_ds[source_row_index]
            text = str(row.get("text") or "").strip()
            if not text:
                raise ValueError(f"Transcript rỗng: {split} {file_path.name} row={source_row_index}")
            audio_info = row["audio"]
            duration, audio_path = read_audio_metadata(audio_info)
            row_index = len(records_by_split[split])
            name = row.get("filename") or (Path(audio_path).name if audio_path else f"{split}_{row_index:07d}.wav")
            record = {
                "sample_id": f"{split}_{row_index:07d}", "split": split,
                "row_index": row_index, "source_file": str(source_file),
                "source_audio_root": str(file_path.parent.resolve()),
                "source_row_index": source_row_index, "audio_path": audio_path,
                "filename": str(name), "text": text, "duration_seconds": duration,
                "region": row.get("region"), "province_name": row.get("province_name"),
                "province_code": row.get("province_code"),
                "speakerID": row.get("speakerID", row.get("speaker_id")),
            }
            records_by_split[split].append(record)
            manifest_records.append(record)

if any(not records_by_split[s] for s in records_by_split):
    raise ValueError(f"Có split rỗng: { {s: len(v) for s, v in records_by_split.items()} }")
if len({r["sample_id"] for r in manifest_records}) != len(manifest_records):
    raise ValueError("sample_id bị trùng")
MANIFEST = {
    "source_dataset": DATASET_HANDLE, "source_license": "See source Kaggle dataset card",
    "source_format": SOURCE_FORMAT, "records": manifest_records,
}
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"
MANIFEST_PATH.write_text(json.dumps(MANIFEST, ensure_ascii=False), encoding="utf-8")
print("Manifest:", MANIFEST_PATH, "| records:", len(manifest_records))
for split, rows in records_by_split.items():
    hours = sum(r["duration_seconds"] for r in rows) / 3600
    provinces = len({r["province_name"] for r in rows if r["province_name"]})
    speakers = len({str(r["speakerID"]) for r in rows if r["speakerID"]})
    print(f"{split}: {len(rows):,} clips | {hours:.2f} audio hours | {provinces} provinces | {speakers} speakers")
speaker_sets = {s: {str(r["speakerID"]) for r in rows if r.get("speakerID")} for s, rows in records_by_split.items()}
print("Speaker overlap train/valid:", len(speaker_sets["train"] & speaker_sets["valid"]))
print("Speaker overlap train/test:", len(speaker_sets["train"] & speaker_sets["test"]))
print("Speaker overlap valid/test:", len(speaker_sets["valid"] & speaker_sets["test"]))
CONFIG.update({"dataset_root": str(DATASET_ROOT), "dataset_source_format": SOURCE_FORMAT,
               "split_counts": {s: len(v) for s, v in records_by_split.items()},
               "split_hours": {s: sum(r["duration_seconds"] for r in v) / 3600 for s, v in records_by_split.items()}})
(OUTPUT_DIR / "run_config.json").write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")


# %% [code]
# Kiểm tra audio từ các shard gốc; chỉ nạp vài clip để xác nhận decode/resample.
import io
from math import gcd
from scipy.signal import resample_poly

def read_source_audio(row):
    ds = source_datasets[row["source_file"]]
    audio_info = ds[int(row["source_row_index"])]["audio"]
    if audio_info.get("bytes") is not None:
        source = io.BytesIO(audio_info["bytes"])
    else:
        audio_path = Path(audio_info["path"])
        if not audio_path.is_absolute():
            audio_path = Path(row["source_audio_root"]) / audio_path
            if not audio_path.is_file():
                audio_path = DATASET_ROOT / audio_info["path"]
        source = str(audio_path)
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0]
    if sample_rate != 16000:
        divisor = gcd(int(sample_rate), 16000)
        audio = resample_poly(audio, 16000 // divisor, int(sample_rate) // divisor).astype(np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.ndim != 1 or not np.isfinite(audio).all():
        raise ValueError(f"Audio sai shape/NaN: {row['sample_id']}")
    actual_duration = len(audio) / 16000
    if abs(actual_duration - float(row["duration_seconds"])) > 0.15:
        raise ValueError(f"Duration lệch: {row['sample_id']} {actual_duration:.3f}s vs {row['duration_seconds']:.3f}s")
    return audio, actual_duration

for split in ("train", "valid", "test"):
    rows = records_by_split[split]
    longest = max(rows, key=lambda r: r["duration_seconds"])
    for row in (rows[0], longest):
        audio, duration = read_source_audio(row)
        print(split, row["filename"], row["sample_id"], audio.shape, audio.dtype,
              "16kHz mono", f"{duration:.2f}s", "source:", Path(row["source_file"]).name)
print("Direct Arrow/Parquet audio access checks passed.")


# %% [markdown]
# ## Target, checkpoint và đánh giá
#
# Mỗi mẫu đưa **audio** vào user message; assistant target là `language Vietnamese<asr_text>{transcript}`. Không đưa tỉnh/vùng vào model input; chúng chỉ dùng để phân tích kết quả. Transcript giữ nguyên chính tả trong manifest, không chuẩn hóa từ địa phương thành từ phổ thông.
#
# Sau mỗi epoch đo CER trên valid và giữ adapter có CER thấp nhất. CER/WER dùng Unicode NFC, chữ thường, bỏ dấu câu và chuẩn hóa khoảng trắng; CER bỏ khoảng trắng. Báo cáo cuối dùng corpus error counts chia tổng số ký tự/từ tham chiếu, tổng thể, theo vùng và theo tỉnh.

# %% [code]
# Worker script chạy ở tiến trình Python mới. Mỗi torchrun rank giữ một bản model trên một GPU.
WORKER_PART_A = r'''import argparse
import csv
import json
import math
import os
import random
import re
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path
from io import BytesIO
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration, get_linear_schedule_with_warmup
from datasets import Audio, Dataset as HFDataset
from peft import AdaLoraConfig, PeftModel, TaskType, get_peft_model

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["smoke", "train", "test"], required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", required=True)
    return p.parse_args()

args = parse_args()
CFG = json.loads(Path(args.config).read_text(encoding="utf-8"))
OUT = Path(args.output)
DATA_ROOT = Path(args.data_root)
MANIFEST = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
ROWS = {s: [r for r in MANIFEST["records"] if r["split"] == s] for s in ("train", "valid", "test")}
TOKEN = os.environ.get("HF_TOKEN") or None
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
RANK = int(os.environ.get("RANK", "0"))
WORLD = int(os.environ.get("WORLD_SIZE", "1"))
IS_DISTRIBUTED = WORLD > 1
torch.cuda.set_device(LOCAL_RANK)
if IS_DISTRIBUTED:
    dist.init_process_group(backend="nccl")
DEVICE = torch.device("cuda", LOCAL_RANK)
torch.manual_seed(int(CFG["seed"]) + RANK)
np.random.seed(int(CFG["seed"]) + RANK)
random.seed(int(CFG["seed"]) + RANK)

def log(*items):
    if RANK == 0:
        print(*items, flush=True)

class ManifestAudioDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.source_cache = {}
        self.cache_pid = os.getpid()
    def __len__(self):
        return len(self.rows)
    def __getstate__(self):
        state = self.__dict__.copy()
        state["source_cache"] = {}
        return state
    def _get_source(self, path):
        if self.cache_pid != os.getpid():
            self.source_cache = {}
            self.cache_pid = os.getpid()
        path = str(path)
        if path not in self.source_cache:
            self.source_cache[path] = HFDataset.from_file(path).cast_column("audio", Audio(decode=False))
        return self.source_cache[path]
    def _read_audio(self, row):
        source_row = self._get_source(row["source_file"])[int(row["source_row_index"])]
        audio_info = source_row["audio"]
        raw = audio_info.get("bytes")
        if raw is not None:
            source = BytesIO(raw)
        else:
            audio_path = Path(audio_info["path"])
            if not audio_path.is_absolute():
                rel_path = audio_path
                candidates = []
                if row.get("source_audio_root"):
                    candidates.append(Path(row["source_audio_root"]) / rel_path)
                candidates.append(DATA_ROOT / rel_path)
                audio_path = next((p for p in candidates if p.is_file()), None)
                if audio_path is None:
                    raise FileNotFoundError(
                        f"Audio path not found for {row['sample_id']}: {candidates}"
                    )
            source = str(audio_path)
        audio, sr = sf.read(source, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1, dtype=np.float32) if audio.shape[1] > 1 else audio[:, 0]
        if sr != 16000:
            div = gcd(int(sr), 16000)
            audio = resample_poly(audio, 16000 // div, int(sr) // div).astype(np.float32)
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Bad waveform " + row["sample_id"])
        return audio
    def __getitem__(self, idx):
        row = self.rows[idx]
        return {"row": row, "audio": self._read_audio(row)}

def identity_collate(items):
    return items[0]

def make_loader(rows, shuffle=False, batch_size=1):
    ds = ManifestAudioDataset(rows)
    sampler = DistributedSampler(ds, num_replicas=WORLD, rank=RANK, shuffle=shuffle,
                                 seed=int(CFG["seed"]), drop_last=False) if IS_DISTRIBUTED else None
    loader = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        shuffle=(shuffle and sampler is None), num_workers=int(CFG["num_workers"]),
                        pin_memory=False, collate_fn=identity_collate,
                        persistent_workers=(int(CFG["num_workers"]) > 0))
    return loader, sampler

def load_processor():
    return AutoProcessor.from_pretrained(CFG["model_id"], token=TOKEN)

def make_model(revision=None):
    kw = {"token": TOKEN, "dtype": torch.float16, "low_cpu_mem_usage": True}
    if revision:
        kw["revision"] = revision
    model = Qwen3ASRForConditionalGeneration.from_pretrained(CFG["model_id"], **kw)
    model.to(DEVICE)
    return model

def revision_of(model):
    return getattr(model.config, "_commit_hash", None)

def save_actual_revision(model):
    if RANK == 0:
        path = OUT / "run_config.json"
        cfg = json.loads(path.read_text(encoding="utf-8")) if path.exists() else CFG
        cfg["model_revision"] = revision_of(model)
        cfg["transformers_version"] = __import__("transformers").__version__
        cfg["peft_version"] = __import__("peft").__version__
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    barrier()

def barrier():
    if IS_DISTRIBUTED:
        dist.barrier()

def make_inputs(processor, item, output_labels=False):
    row, audio = item["row"], item["audio"]
    user_msg = {"role": "user", "content": [{"type": "audio", "audio": audio}]}
    messages = [user_msg]
    if output_labels:
        target = "language Vietnamese<asr_text>" + str(row["text"]).strip()
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    out = processor.apply_chat_template(
        [messages], tokenize=True, return_dict=True,
        processor_kwargs={"output_labels": output_labels, "sampling_rate": 16000,
                          "padding": "longest", "truncation": False},
    )
    if output_labels:
        if "labels" not in out:
            raise KeyError("Processor không trả labels; kiểm tra template/output_labels.")
        if not torch.is_tensor(out["labels"]) or int((out["labels"] != -100).sum()) == 0:
            raise ValueError("Labels rỗng hoặc không được mask đúng.")
    moved = {}
    for key, val in out.items():
        if torch.is_tensor(val):
            if val.is_floating_point():
                moved[key] = val.to(device=DEVICE, dtype=torch.float16)
            else:
                moved[key] = val.to(device=DEVICE)
        else:
            moved[key] = val
    return moved

def output_loss(output):
    if isinstance(output, dict):
        loss = output.get("loss")
    else:
        loss = getattr(output, "loss", None)
    if isinstance(loss, dict):
        loss = loss.get("loss")
    if loss is None:
        raise TypeError("Model forward không trả scalar loss.")
    if not torch.is_tensor(loss):
        loss = torch.as_tensor(loss, device=DEVICE)
    return loss.mean()

def find_adalora_tuner(peft_model):
    for obj in (peft_model, getattr(peft_model, "base_model", None),
                getattr(getattr(peft_model, "base_model", None), "model", None)):
        if obj is not None and hasattr(obj, "update_and_allocate"):
            return obj
    raise AttributeError("PEFT AdaLoRA wrapper không có update_and_allocate().")

def make_adapter_config(total_steps, delta_t=10):
    return AdaLoraConfig(
        init_r=int(CFG["init_r"]), target_r=int(CFG["target_r"]),
        target_modules=list(CFG["target_modules"]),
        lora_alpha=int(CFG["lora_alpha"]), lora_dropout=float(CFG["lora_dropout"]),
        bias="none", task_type=TaskType.CAUSAL_LM, total_step=int(total_steps),
        tinit=max(1, round(0.05 * total_steps)), tfinal=max(1, round(0.05 * total_steps)),
        deltaT=int(delta_t),
    )

def report_targets(base):
    names = [n for n, m in base.named_modules() if isinstance(m, torch.nn.Linear)
             and any(n.endswith(s) for s in CFG["target_modules"])]
    audio = [n for n in names if "audio" in n.lower() or "encoder" in n.lower()]
    decoder = [n for n in names if n not in audio]
    log("Matched q/v Linear modules:", len(names), "| audio/encoder:", len(audio), "| remaining:", len(decoder))
    log("Target examples:", names[:12])
    if not names or not audio or not decoder:
        raise RuntimeError("Không tìm thấy q_proj/v_proj đồng thời ở audio encoder và language decoder.")
    return names

def wrap_adalora(base, total_steps, delta_t=10):
    names = report_targets(base)
    model = get_peft_model(base, make_adapter_config(total_steps, delta_t=delta_t))
    model.print_trainable_parameters()
    tuner = find_adalora_tuner(model)
    return model, tuner, names

def normalize(text):
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

def err_counts(ref, hyp):
    r, h = normalize(ref), normalize(hyp)
    rt, ht = r.split(), h.split()
    return edit_distance(rt, ht), max(1, len(rt)), edit_distance(r.replace(" ", ""), h.replace(" ", "")), max(1, len(r.replace(" ", "")))

def decode_prediction(model, processor, item):
    inputs = processor.apply_transcription_request(audio=item["audio"], language="Vietnamese")
    moved = {}
    for key, val in inputs.items():
        if torch.is_tensor(val):
            moved[key] = val.to(device=DEVICE, dtype=torch.float16) if val.is_floating_point() else val.to(device=DEVICE)
        else:
            moved[key] = val
    inputs = moved
    with torch.inference_mode():
        ids = model.generate(**inputs, max_new_tokens=int(CFG["max_new_tokens"]), do_sample=False)
    prompt_len = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
    generated = ids[:, prompt_len:] if prompt_len and ids.shape[-1] > prompt_len else ids
    try:
        decoded = processor.decode(generated[0], return_format="transcription_only")
        if isinstance(decoded, (list, tuple)):
            decoded = decoded[0]
        if isinstance(decoded, dict):
            decoded = decoded.get("text", decoded.get("transcription", str(decoded)))
        return str(decoded).strip()
    except Exception:
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

def gather_objects(obj):
    if not IS_DISTRIBUTED:
        return [obj]
    output = [None for _ in range(WORLD)]
    dist.all_gather_object(output, obj)
    return output

def evaluate(model, processor, split, name):
    rows = ROWS[split]
    loader, sampler = make_loader(rows, shuffle=False, batch_size=1)
    if sampler is not None:
        sampler.set_epoch(0)
    model.eval()
    local = []
    with torch.inference_mode():
        for item in loader:
            row = item["row"]
            start = time.perf_counter()
            try:
                pred = decode_prediction(model, processor, item)
                error = ""
            except Exception as exc:
                pred, error = "", repr(exc)
            local.append({
                "sample_id": row["sample_id"], "model": name, "split": split,
                "filename": row.get("filename"), "region": row.get("region"),
                "province_name": row.get("province_name"), "speakerID": row.get("speakerID"),
                "reference": row["text"], "prediction": pred, "error": error,
                "inference_seconds": time.perf_counter() - start,
            })
    chunks = gather_objects(local)
    if RANK == 0:
        unique = {}
        for chunk in chunks:
            for item in chunk:
                unique[item["sample_id"]] = item
        result = list(unique.values())
        if len(result) != len(rows):
            raise RuntimeError(f"{name}/{split}: got {len(result)} unique predictions, expected {len(rows)}")
        result.sort(key=lambda x: x["sample_id"])
    else:
        result = None
    barrier()
    return result

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def make_base_processor_and_model():
    proc = load_processor()
    base = make_model()
    save_actual_revision(base)
    return proc, base

'''

# %% [code]
# Worker phần B: custom AdaLoRA train loop và DDP smoke test.
WORKER_PART_B = r'''def train():
    train_loader, sampler = make_loader(ROWS["train"], shuffle=True, batch_size=1)
    steps_per_epoch = math.ceil(len(train_loader) / int(CFG["gradient_accumulation_steps"]))
    total_steps = steps_per_epoch * int(CFG["epochs"])
    log("train rows/rank:", len(train_loader), "steps/epoch:", steps_per_epoch, "total optimizer steps:", total_steps)
    processor, base = make_base_processor_and_model()
    peft_model, tuner, target_names = wrap_adalora(base, total_steps)
    peft_model.config.use_cache = False
    try:
        peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        peft_model.gradient_checkpointing_enable()
    ddp = DDP(peft_model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False)
    optimizer = torch.optim.AdamW((p for p in ddp.parameters() if p.requires_grad),
                                  lr=float(CFG["learning_rate"]), weight_decay=float(CFG["weight_decay"]))
    warmup = int(float(CFG["warmup_ratio"]) * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, total_steps)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    global_step = 0
    attempted_steps = 0
    skipped_steps = 0
    best_cer = float("inf")
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(int(CFG["epochs"])):
        if sampler is not None:
            sampler.set_epoch(epoch)
        ddp.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        n_batches = len(train_loader)
        for batch_index, item in enumerate(train_loader):
            inputs = make_inputs(processor, item, output_labels=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = ddp(**inputs)
                loss = output_loss(output)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite train loss at epoch={epoch+1}, batch={batch_index}")
            losses.append(float(loss.detach()))
            group_start = (batch_index // int(CFG["gradient_accumulation_steps"])) * int(CFG["gradient_accumulation_steps"])
            group_size = min(int(CFG["gradient_accumulation_steps"]), n_batches - group_start)
            scaler.scale(loss / group_size).backward()
            should_step = ((batch_index + 1) % int(CFG["gradient_accumulation_steps"]) == 0) or (batch_index + 1 == n_batches)
            if should_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(ddp.parameters(), 1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                new_scale = scaler.get_scale()
                attempted_steps += 1
                did_step = new_scale >= old_scale
                if did_step:
                    global_step += 1
                    scheduler.step()
                    tuner.update_and_allocate(global_step)
                else:
                    skipped_steps += 1
                    print(
                        f"[rank{RANK}] GradScaler skipped update: "
                        f"epoch={epoch + 1}/{CFG['epochs']} batch={batch_index + 1}/{n_batches} "
                        f"attempt={attempted_steps}/{total_steps} successful={global_step} "
                        f"scale={old_scale}->{new_scale} grad_norm={float(grad_norm.detach())}",
                        flush=True,
                    )
                optimizer.zero_grad(set_to_none=True)
                if RANK == 0 and global_step and global_step % 50 == 0:
                    log(f"epoch={epoch+1}/{CFG['epochs']} step={global_step}/{total_steps} loss={losses[-1]:.4f}")

        barrier()
        valid_rows = evaluate(ddp.module, processor, "valid", f"adalora_epoch_{epoch+1}")
        if RANK == 0:
            errors = [r for r in valid_rows if r["error"]]
            if errors:
                raise RuntimeError(f"Validation có {len(errors)} inference error(s), ví dụ: {errors[:2]}")
            cer_errors = sum(err_counts(r["reference"], r["prediction"])[2] for r in valid_rows)
            cer_units = sum(err_counts(r["reference"], r["prediction"])[3] for r in valid_rows)
            wer_errors = sum(err_counts(r["reference"], r["prediction"])[0] for r in valid_rows)
            wer_units = sum(err_counts(r["reference"], r["prediction"])[1] for r in valid_rows)
            val_cer, val_wer = cer_errors / cer_units, wer_errors / wer_units
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            history.append({"epoch": epoch + 1, "optimizer_steps": global_step,
                            "optimizer_steps_attempted": attempted_steps,
                            "optimizer_steps_skipped": skipped_steps,
                            "train_loss": mean_loss, "valid_cer": val_cer, "valid_wer": val_wer,
                            "cer_errors": cer_errors, "cer_ref_chars": cer_units,
                            "wer_errors": wer_errors, "wer_ref_words": wer_units})
            log(f"epoch {epoch+1}: loss={mean_loss:.5f} valid CER={val_cer:.5f} WER={val_wer:.5f}")
            if val_cer < best_cer:
                best_cer = val_cer
                adapter_dir = OUT / f"epoch_{epoch+1}_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                ddp.module.save_pretrained(adapter_dir)
                processor.save_pretrained(adapter_dir)
                (OUT / "best_epoch.txt").write_text(str(epoch + 1), encoding="utf-8")
                log("Saved new best adapter:", adapter_dir)
            write_csv(OUT / "train_history.csv", history)
            cfgpath = OUT / "run_config.json"
            cfgdata = json.loads(cfgpath.read_text(encoding="utf-8"))
            cfgdata.update({"optimizer_steps_completed": global_step,
                            "optimizer_steps_planned": total_steps,
                            "optimizer_steps_attempted": attempted_steps,
                            "optimizer_steps_skipped": skipped_steps,
                            "steps_per_epoch_per_rank": steps_per_epoch,
                            "adalora_tinit": max(1, round(0.05 * total_steps)),
                            "adalora_tfinal": max(1, round(0.05 * total_steps)),
                            "matched_target_modules": len(target_names)})
            cfgpath.write_text(json.dumps(cfgdata, ensure_ascii=False, indent=2), encoding="utf-8")
            if len(history) != epoch + 1:
                raise RuntimeError("History row count mismatch.")
        barrier()

    if attempted_steps != total_steps:
        raise RuntimeError(f"Đã thử {attempted_steps} bước, dự kiến {total_steps}.")
    if global_step + skipped_steps != attempted_steps:
        raise RuntimeError(
            f"Đếm bước không khớp: successful={global_step}, "
            f"skipped={skipped_steps}, attempted={attempted_steps}."
        )
    if RANK == 0 and skipped_steps:
        log(f"Train đã đi hết dữ liệu; GradScaler bỏ qua {skipped_steps}/{attempted_steps} lần cập nhật.")

def smoke():
    # Tạo một batch trên mỗi rank, lấy các clip dài nhất để kiểm tra memory/forward-backward.
    candidates = sorted(ROWS["train"], key=lambda r: float(r["duration_seconds"]), reverse=True)[:max(2, WORLD)]
    row = candidates[RANK % len(candidates)]
    item = ManifestAudioDataset([row])[0]
    processor, base = make_base_processor_and_model()
    model, tuner, target_names = wrap_adalora(base, total_steps=20, delta_t=1)
    model.config.use_cache = False
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        model.gradient_checkpointing_enable()
    ddp = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK, find_unused_parameters=False)
    ddp.train()
    inputs = make_inputs(processor, item, output_labels=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = output_loss(ddp(**inputs))
    if not torch.isfinite(loss):
        raise FloatingPointError("Smoke test loss không hữu hạn.")
    loss.backward()
    optimizer = torch.optim.AdamW((p for p in ddp.parameters() if p.requires_grad), lr=1e-4)
    optimizer.step()
    tuner.update_and_allocate(1)
    optimizer.zero_grad(set_to_none=True)
    barrier()
    ddp.eval()
    # Smoke generation kiểm tra luôn đường suy luận/template.
    pred = decode_prediction(ddp.module, processor, {"row": row, "audio": item["audio"]})
    peak = torch.cuda.max_memory_allocated(DEVICE) / 1024**3
    rank_report = {"rank": RANK, "device": torch.cuda.get_device_name(DEVICE),
                   "loss": float(loss), "peak_vram_GB": peak,
                   "audio_seconds": float(row["duration_seconds"]), "prediction_chars": len(pred),
                   "matched_target_modules": len(target_names)}
    reports = gather_objects(rank_report)
    if RANK == 0:
        if len(reports) != 2 or {entry["rank"] for entry in reports} != {0, 1}:
            raise RuntimeError("Smoke test không xác nhận được cả hai DDP rank.")
        log("Two-GPU smoke reports:", reports)
    barrier()

'''

# %% [code]
# Worker phần C: test inference, metric gathering và entry point torchrun.
WORKER_PART_C = r'''def test():
    processor = load_processor()
    all_predictions = []
    # Load/evaluate the base model first, then release it before loading adapter.
    base = make_model()
    save_actual_revision(base)
    base.eval()
    base_rows = evaluate(base, processor, "test", "base")
    if RANK == 0:
        all_predictions.extend(base_rows)
        del base_rows
    del base
    torch.cuda.empty_cache()
    barrier()
    best_file = OUT / "best_epoch.txt"
    if not best_file.exists():
        raise FileNotFoundError("Thiếu best_epoch.txt; hãy chạy train trước.")
    best_epoch = int(best_file.read_text(encoding="utf-8").strip())
    base = make_model()
    adapter_path = OUT / f"epoch_{best_epoch}_adapter"
    adapted = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    adapted.to(DEVICE).eval()
    adapter_rows = evaluate(adapted, processor, "test", "adalora")
    if RANK == 0:
        all_predictions.extend(adapter_rows)
        write_csv(OUT / "test_predictions.csv", all_predictions)
        errors = [r for r in all_predictions if r["error"]]
        log("Wrote", OUT / "test_predictions.csv", "| rows:", len(all_predictions), "| errors:", len(errors))
        if errors:
            log("First errors:", errors[:3])
    barrier()

def main():
    try:
        if args.mode == "smoke":
            smoke()
        elif args.mode == "train":
            train()
        else:
            test()
        barrier()
    finally:
        if IS_DISTRIBUTED and dist.is_initialized():
            dist.destroy_process_group()

if __name__ == "__main__":
    main()
'''
WORKER_SOURCE = WORKER_PART_A + WORKER_PART_B + WORKER_PART_C
compile(WORKER_SOURCE, 'train_worker.py', 'exec')
print('Worker assembled:', len(WORKER_SOURCE.splitlines()), 'lines')

# %% [code]
# Ghi worker vào thư mục writable và kiểm tra cú pháp/import bằng Python mới.
WORKER_DIR = OUTPUT_DIR / "worker"
WORKER_DIR.mkdir(parents=True, exist_ok=True)
WORKER_PATH = WORKER_DIR / "train_worker.py"
WORKER_PATH.write_text(WORKER_SOURCE, encoding="utf-8")
check = subprocess.run(
    [sys.executable, "-m", "py_compile", str(WORKER_PATH)],
    capture_output=True, text=True,
)
if check.returncode:
    raise RuntimeError(check.stderr)
probe = subprocess.run(
    [sys.executable, "-c",
     "from transformers.models.qwen3_asr import Qwen3ASRForConditionalGeneration; "
     "from peft import AdaLoraConfig, PeftModel; import torch.distributed"],
    capture_output=True, text=True,
)
if probe.returncode:
    raise RuntimeError("Fresh-process import failed:\n" + probe.stderr)
print("Worker syntax/import OK:", WORKER_PATH)

# %% [code]
# Smoke DDP: hai process, hai GPU; forward + backward + AdaLoRA allocation + generate.
# Clip dài nhất được lấy từ toàn bộ train split để thử peak memory trước khi train.
# Nếu cell này lỗi hoặc hết VRAM thì không chạy full training.
env = os.environ.copy()
if HF_TOKEN:
    env["HF_TOKEN"] = HF_TOKEN
smoke_cmd = [
    "torchrun", "--standalone", "--nproc_per_node=2", str(WORKER_PATH),
    "--mode", "smoke", "--manifest", str(MANIFEST_PATH),
    "--data-root", str(DATASET_ROOT), "--output", str(OUTPUT_DIR),
    "--config", str(OUTPUT_DIR / "run_config.json"),
]
print("Running:", " ".join(smoke_cmd))
run = subprocess.run(smoke_cmd, env=env, text=True)
if run.returncode:
    raise RuntimeError(f"2-GPU smoke test thất bại (exit={run.returncode}); xem traceback phía trên.")
print("Smoke test thành công trên 2 rank.")

# %% [code]
# Fine-tune trên toàn bộ split train trong 2 epoch; valid chỉ dùng chọn checkpoint.
env = os.environ.copy()
if HF_TOKEN:
    env["HF_TOKEN"] = HF_TOKEN
train_cmd = [
    "torchrun", "--standalone", "--nproc_per_node=2", str(WORKER_PATH),
    "--mode", "train", "--manifest", str(MANIFEST_PATH),
    "--data-root", str(DATASET_ROOT), "--output", str(OUTPUT_DIR),
    "--config", str(OUTPUT_DIR / "run_config.json"),
]
per_rank_rows = math.ceil(len(records_by_split["train"]) / 2)
steps_per_epoch = math.ceil(per_rank_rows / GRADIENT_ACCUMULATION_STEPS)
print(f"Kế hoạch: {len(records_by_split['train']):,} train clips × {EPOCHS} epoch | "
      f"{steps_per_epoch * EPOCHS:,} optimizer steps; effective batch mỗi bước {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * 2}")
run = subprocess.run(train_cmd, env=env, text=True)
if run.returncode:
    raise RuntimeError(f"Training dừng với exit={run.returncode}; xem traceback/log phía trên.")
history_path = OUTPUT_DIR / "train_history.csv"
if not history_path.is_file():
    raise FileNotFoundError("Training kết thúc nhưng chưa có train_history.csv")


# %% [code]
# Tóm tắt history và vẽ train loss/valid CER/WER.
import matplotlib.pyplot as plt
history = pd.read_csv(OUTPUT_DIR / "train_history.csv")
display(history)
best_row = history.loc[history["valid_cer"].idxmin()]
print("Best epoch:", int(best_row["epoch"]), "| valid CER:", best_row["valid_cer"], "| valid WER:", best_row["valid_wer"])
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(history["epoch"], history["train_loss"], marker="o", label="train loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train loss")
ax2 = ax1.twinx()
ax2.plot(history["epoch"], history["valid_cer"], marker="s", color="tab:red", label="valid CER")
ax2.plot(history["epoch"], history["valid_wer"], marker="^", color="tab:orange", label="valid WER")
ax2.set_ylabel("Valid error rate")
fig.tight_layout()
plt.show()

# %% [code]
# Benchmark toàn bộ test hai lần: base và adapter tốt nhất, cùng toàn bộ split test.
env = os.environ.copy()
if HF_TOKEN:
    env["HF_TOKEN"] = HF_TOKEN
test_cmd = [
    "torchrun", "--standalone", "--nproc_per_node=2", str(WORKER_PATH),
    "--mode", "test", "--manifest", str(MANIFEST_PATH),
    "--data-root", str(DATASET_ROOT), "--output", str(OUTPUT_DIR),
    "--config", str(OUTPUT_DIR / "run_config.json"),
]
run = subprocess.run(test_cmd, env=env, text=True)
if run.returncode:
    raise RuntimeError(f"Test inference lỗi (exit={run.returncode}); xem log phía trên.")
pred_path = OUTPUT_DIR / "test_predictions.csv"
predictions = pd.read_csv(pred_path)
display(predictions.head())
expected_test_n = len(records_by_split["test"])
model_counts = predictions.groupby("model")["sample_id"].nunique().to_dict()
print("Rows:", len(predictions), "| expected per model:", expected_test_n, "| by model:", model_counts)
if model_counts != {"adalora": expected_test_n, "base": expected_test_n}:
    raise ValueError("Mỗi model phải có đúng số sample_id duy nhất của split test.")
if predictions["error"].fillna("").astype(str).str.len().gt(0).any():
    print("Có inference error; xem cột error trước khi diễn giải metric.")


# %% [code]
# Corpus-level CER/WER: tổng edit errors / tổng số đơn vị tham chiếu.
import re
import unicodedata

def norm_eval(text):
    text = unicodedata.normalize("NFC", str(text or "")).lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return " ".join(text.split())

def edit_count(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def corpus_metric(frame):
    cer_err = cer_den = wer_err = wer_den = 0
    for row in frame.itertuples(index=False):
        ref, hyp = norm_eval(row.reference), norm_eval(row.prediction)
        rt, ht = ref.split(), hyp.split()
        wer_err += edit_count(rt, ht)
        wer_den += max(1, len(rt))
        rc, hc = ref.replace(" ", ""), hyp.replace(" ", "")
        cer_err += edit_count(list(rc), list(hc))
        cer_den += max(1, len(rc))
    return {"n_samples": len(frame), "cer_errors": cer_err, "reference_chars": cer_den,
            "CER": cer_err / cer_den, "wer_errors": wer_err,
            "reference_words": wer_den, "WER": wer_err / wer_den}

valid_predictions = predictions[predictions["error"].fillna("").eq("")]
if len(valid_predictions) != len(predictions):
    raise ValueError("Có sample lỗi inference; không tổng hợp metric như thể đủ dữ liệu.")
scope = []
for model_name, model_frame in predictions.groupby("model"):
    for level, col in [("overall", None), ("region", "region"), ("province", "province_name")]:
        groups = [(("all",), model_frame)] if col is None else model_frame.groupby(col, dropna=False)
        for key, group in groups:
            if col is None:
                value = "all"
            else:
                value = key
            scope.append({"model": model_name, "level": level, "group": value, **corpus_metric(group)})
metrics = pd.DataFrame(scope)
overall = metrics[metrics["level"] == "overall"].set_index("model")
base_cer, ada_cer = overall.loc["base", "CER"], overall.loc["adalora", "CER"]
base_wer, ada_wer = overall.loc["base", "WER"], overall.loc["adalora", "WER"]
metrics["delta_CER_vs_base"] = np.nan
metrics["delta_WER_vs_base"] = np.nan
for level in metrics["level"].unique():
    base_map = metrics[(metrics["model"] == "base") & (metrics["level"] == level)].set_index("group")
    idx = (metrics["model"] == "adalora") & (metrics["level"] == level)
    metrics.loc[idx, "delta_CER_vs_base"] = metrics.loc[idx].apply(lambda r: r["CER"] - base_map.loc[r["group"], "CER"], axis=1)
    metrics.loc[idx, "delta_WER_vs_base"] = metrics.loc[idx].apply(lambda r: r["WER"] - base_map.loc[r["group"], "WER"], axis=1)
metrics.to_csv(OUTPUT_DIR / "test_metrics_overall_region_province.csv", index=False, encoding="utf-8-sig")
metrics[metrics["level"] == "overall"].to_csv(OUTPUT_DIR / "test_metrics_overall.csv", index=False, encoding="utf-8-sig")
metrics[metrics["level"] == "region"].to_csv(OUTPUT_DIR / "test_metrics_by_region.csv", index=False, encoding="utf-8-sig")
metrics[metrics["level"] == "province"].to_csv(OUTPUT_DIR / "test_metrics_by_province.csv", index=False, encoding="utf-8-sig")
print("Overall base:", {"CER": base_cer, "WER": base_wer})
print("Overall AdaLoRA:", {"CER": ada_cer, "WER": ada_wer})
print("Relative comparison (negative delta is an improvement):", {"delta_CER": ada_cer-base_cer, "delta_WER": ada_wer-base_wer})
print("Wrote metrics under", OUTPUT_DIR)
display(metrics[metrics["level"] == "overall"])
