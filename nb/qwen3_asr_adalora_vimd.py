# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Qwen3-ASR 1.7B + AdaLoRA trên ViMD
#
# Notebook này là một thí nghiệm ASR low-resource độc lập. Nó giữ backbone Qwen đóng băng, gắn AdaLoRA vào các projection attention, huấn luyện bằng loss chính thức của Qwen và so sánh WER/CER với model gốc.
#
# **Lưu ý:** mặc định chỉ chạy smoke test và chưa fine-tune. Đặt `RUN_TRAINING=True` sau khi kiểm tra được một batch forward. AdaLoRA phải gọi `update_and_allocate()` ở mỗi optimizer step; vì vậy notebook dùng custom loop thay vì `Trainer` mặc định.

# %% [code]
# Nếu môi trường chưa có dependency, bỏ dấu # ở cell dưới rồi restart kernel.
# %pip install -U "transformers>=5.13.0" "peft>=0.17.0" "accelerate>=1.2.0" jiwer soundfile datasets

import os
import io
import csv
import json
import math
import random
import struct
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForMultimodalLM, get_linear_schedule_with_warmup
from peft import AdaLoraConfig, get_peft_model
        
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
LANGUAGE = "Vietnamese"
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = (torch.bfloat16 if DEVICE.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16 if DEVICE.type == "cuda" else torch.float32)
        
# Đổi các path này khi chạy Kaggle/local khác.
PROJECT_DIR = Path.cwd()
OUTPUT_DIR = PROJECT_DIR / "outputs" / "qwen3_asr_adalora"
AUDIO_CACHE_DIR = OUTPUT_DIR / "audio_cache"
ADAPTER_DIR = OUTPUT_DIR / "adapter"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True); AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE_METADATA_FILE = Path(os.environ.get("VIMD_SAMPLE_METADATA", PROJECT_DIR / "outputs/vimd_local_experiment/vimd_sample_metadata.json"))
VIMD_CACHE_DIR = os.environ.get("VIMD_CACHE_DIR", "")

# Smoke mặc định; đặt None để dùng toàn bộ records trong sample_metadata.
MAX_TRAIN_SAMPLES = 32
MAX_VALID_SAMPLES = 8
MAX_TEST_SAMPLES = 8
MAX_AUDIO_SECONDS = 30.0
EPOCHS = 2
PER_DEVICE_BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
INIT_R = 12
TARGET_R = 4
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULE_SUFFIXES = ["q_proj", "v_proj"]
RUN_TRAINING = False
RUN_BASELINE_EVAL = False
RUN_ADALORA_EVAL = False

print("torch:", torch.__version__, "device:", DEVICE, "dtype:", DTYPE)
        
assert int(torch.__version__.split(".")[0]) >= 2, "Cần PyTorch 2.x trở lên."


# %% [markdown]
# ## 1. Đọc sample ViMD và materialize audio vào thư mục ghi được
#
# Không ghi vào `/kaggle/input`. Notebook dùng `row_index` trong `sample_metadata.json` để lấy bytes từ Arrow cache, sau đó ghi WAV vào `outputs/qwen3_asr_adalora/audio_cache`.

# %% [code]
from datasets import Dataset, DatasetDict, Audio, concatenate_datasets
        
def locate_arrow_cache():
    if VIMD_CACHE_DIR:
        root = Path(VIMD_CACHE_DIR)
        if any(root.glob("vi_md_dataset-*.arrow")):
            return root
        candidates = list(root.rglob("vi_md_dataset-*.arrow")) if root.exists() else []
        if candidates:
            return candidates[0].parent
    candidates = [
        Path.home() / ".cache/huggingface/datasets",
        PROJECT_DIR / "data",
    ]
    for root in candidates:
        if root.exists():
            hits = list(root.rglob("vi_md_dataset-train-*.arrow"))
            if hits:
                return hits[0].parent
    return None

def load_vimd_from_arrow(cache_root):
    if cache_root is None:
        raise FileNotFoundError("Không tìm thấy Arrow cache. Đặt biến môi trường VIMD_CACHE_DIR hoặc dùng sample_metadata có audio_path hợp lệ.")
    output = {}
    for split in ("train", "valid", "test"):
        paths = sorted(cache_root.rglob(f"vi_md_dataset-{split}-*.arrow"))
        if not paths:
            raise FileNotFoundError(f"Thiếu shard {split} trong {cache_root}")
        output[split] = concatenate_datasets([Dataset.from_file(str(p)) for p in paths]).cast_column("audio", Audio(decode=False))
    return output

sample_records = []
if SAMPLE_METADATA_FILE.is_file():
    payload = json.loads(SAMPLE_METADATA_FILE.read_text(encoding="utf-8"))
    sample_records = payload.get("records", payload if isinstance(payload, list) else [])
else:
    print("Chưa có sample_metadata:", SAMPLE_METADATA_FILE)

dataset_no_decode = None
if sample_records and not all(r.get("audio_path") and Path(str(r["audio_path"])).is_file() for r in sample_records):
    dataset_no_decode = load_vimd_from_arrow(locate_arrow_cache())

def get_audio_bytes(audio_info):
    if audio_info.get("bytes") is not None:
        return audio_info["bytes"]
    path = audio_info.get("path")
    if path and Path(path).is_file():
        return Path(path).read_bytes()
    raise FileNotFoundError("Audio record không có bytes/path hợp lệ")

def materialize_record(record):
    existing = record.get("audio_path")
    if existing and Path(str(existing)).is_file():
        return Path(str(existing))
    if dataset_no_decode is None:
        raise FileNotFoundError("Cần VIMD_CACHE_DIR để materialize record này")
    split = str(record["split"]); row_index = int(record["row_index"])
    data = get_audio_bytes(dataset_no_decode[split][row_index]["audio"])
    filename = Path(str(record.get("filename") or f"row_{row_index}.wav")).name
    path = AUDIO_CACHE_DIR / f"{split}_{row_index}_{filename}"
    if not path.exists(): path.write_bytes(data)
    return path

def choose_records(split, limit):
    rows = [dict(r) for r in sample_records if str(r.get("split")) == split and r.get("text")]
    rows = [r for r in rows if float(r.get("duration_seconds", 0) or 0) <= MAX_AUDIO_SECONDS]
    rng = random.Random(SEED + {"train": 0, "valid": 1, "test": 2}[split]); rng.shuffle(rows)
    if limit is not None: rows = rows[:int(limit)]
    for r in rows: r["audio_path"] = str(materialize_record(r))
    return rows

train_records = choose_records("train", MAX_TRAIN_SAMPLES)
valid_records = choose_records("valid", MAX_VALID_SAMPLES)
test_records = choose_records("test", MAX_TEST_SAMPLES)
print({"train": len(train_records), "valid": len(valid_records), "test": len(test_records)})
if not train_records:
    raise RuntimeError("Không có train records. Kiểm tra SAMPLE_METADATA_FILE và VIMD_CACHE_DIR.")

# %% [markdown]
# ## 2. Load Qwen và kiểm tra baseline
#
# Qwen3-ASR dùng `AutoProcessor` + `AutoModelForMultimodalLM`, không phải CTC. Khi fine-tune, transcript phải nằm trong assistant message theo dạng `language Vietnamese<asr_text>...`; processor tạo `labels` và tự mask phần audio/padding.

# %% [code]
processor = AutoProcessor.from_pretrained(MODEL_ID, token=HF_TOKEN)
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, token=HF_TOKEN, dtype=DTYPE)
model = model.to(DEVICE)
print("Loaded", MODEL_ID, "on", next(model.parameters()).device, "dtype", next(model.parameters()).dtype)

def move_batch(batch):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(DEVICE, dtype=DTYPE) if value.is_floating_point() else value.to(DEVICE)
        else:
            moved[key] = value
    return moved

def training_inputs(record):
    transcript = str(record["text"]).strip()
    conversation = [[
        {"role": "user", "content": [{"type": "audio", "path": str(record["audio_path"])}]},
        {"role": "assistant", "content": [{"type": "text", "text": f"language {LANGUAGE}<asr_text>{transcript}"}]},
    ]]
    batch = processor.apply_chat_template(
        conversation, tokenize=True, return_dict=True,
        processor_kwargs={"output_labels": True},
    )
    return move_batch(batch)

smoke_inputs = training_inputs(train_records[0])
print("Training input keys:", list(smoke_inputs.keys()))
model.eval()
with torch.no_grad():
    smoke_loss = model(**smoke_inputs).loss
print("Smoke loss:", float(smoke_loss))


# %% [markdown]
# ## 3. Gắn AdaLoRA
#
# Không hard-code đường dẫn layer của Qwen. Notebook dò các `nn.Linear` có suffix `q_proj`/`v_proj`, in số module khớp và gắn adapter vào chúng. Đây là cấu hình đầu tiên để kiểm tra tính khả thi; nếu muốn ưu tiên thích nghi acoustic, cần mở rộng suffix sau khi xem danh sách module audio encoder/projector.

# %% [code]
import torch.nn as nn

linear_names = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)]
target_modules = [suffix for suffix in TARGET_MODULE_SUFFIXES if any(name.endswith(suffix) for name in linear_names)]
print("Matched target suffixes:", target_modules)
print("Examples:", [name for name in linear_names if any(name.endswith(s) for s in target_modules)][:20])
if not target_modules:
    raise RuntimeError("Không tìm thấy q_proj/v_proj. Hãy in linear_names và chỉnh TARGET_MODULE_SUFFIXES theo version Transformers.")

steps_per_epoch = math.ceil(len(train_records) / max(1, GRAD_ACCUM_STEPS))
TOTAL_STEPS = max(1, steps_per_epoch * EPOCHS)
TINIT = max(1, int(0.10 * TOTAL_STEPS))
TFINAL = max(1, int(0.20 * TOTAL_STEPS))
adalora_config = AdaLoraConfig(
    r=TARGET_R,
    target_r=TARGET_R,
    init_r=INIT_R,
    target_modules=target_modules,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    tinit=TINIT,
    tfinal=TFINAL,
    deltaT=10,
    total_step=TOTAL_STEPS,
)
model = get_peft_model(model, adalora_config)
model.print_trainable_parameters()
print({"total_steps": TOTAL_STEPS, "tinit": TINIT, "tfinal": TFINAL})

# %% [markdown]
# ## 4. Custom training loop
#
# `update_and_allocate` được gọi sau `loss.backward()` và `optimizer.step()` ở mỗi update. Nếu bỏ dòng này, adapter chỉ hoạt động như LoRA cố định và không còn là AdaLoRA đúng nghĩa.

# %% [code]
def adalora_update(global_step):
    target = getattr(model, "base_model", model)
    if hasattr(target, "update_and_allocate"):
        target.update_and_allocate(global_step)
    elif hasattr(model, "update_and_allocate"):
        model.update_and_allocate(global_step)
    else:
        raise AttributeError("PEFT model không có update_and_allocate; kiểm tra phiên bản peft.")

def validate_loss(records):
    if not records: return float("nan")
    model.eval(); values = []
    with torch.no_grad():
        for record in records:
            values.append(float(model(**training_inputs(record)).loss))
    model.train(); return float(np.mean(values))

def train_adalora():
    model.train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    warmup_steps = int(WARMUP_RATIO * TOTAL_STEPS)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, TOTAL_STEPS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda" and DTYPE == torch.float16))
    global_step = 0; optimizer.zero_grad(set_to_none=True)
    history = []
    for epoch in range(EPOCHS):
        order = list(range(len(train_records))); random.Random(SEED + epoch).shuffle(order)
        running = []
        for local_step, index in enumerate(order, start=1):
            batch = training_inputs(train_records[index])
            autocast_ctx = torch.autocast(device_type="cuda", dtype=DTYPE) if DEVICE.type == "cuda" else nullcontext()
            with autocast_ctx:
                loss = model(**batch).loss / max(1, GRAD_ACCUM_STEPS)
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()
            running.append(float(loss.detach()) * max(1, GRAD_ACCUM_STEPS))
            should_step = local_step % max(1, GRAD_ACCUM_STEPS) == 0 or local_step == len(order)
            if should_step:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(optimizer); scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
                scheduler.step(); global_step += 1
                adalora_update(global_step)
                optimizer.zero_grad(set_to_none=True)
        val = validate_loss(valid_records)
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(running)), "valid_loss": val, "global_step": global_step}
        history.append(row); print(row)
    return pd.DataFrame(history)

if RUN_TRAINING:
    history = train_adalora()
    history.to_csv(OUTPUT_DIR / "training_history.csv", index=False)
    model.save_pretrained(ADAPTER_DIR)
    processor.save_pretrained(ADAPTER_DIR)
    print("Saved adapter:", ADAPTER_DIR)
else:
    print("RUN_TRAINING=False: chưa bắt đầu fine-tune")

# %% [markdown]
# ## 5. Đánh giá baseline và AdaLoRA bằng WER/CER

# %% [code]
def normalize_text(text):
    return " ".join(str(text or "").strip().lower().split())

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1): cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]

def record_wer(ref, hyp):
    r, h = normalize_text(ref).split(), normalize_text(hyp).split()
    return edit_distance(r, h) / max(1, len(r))

def record_cer(ref, hyp):
    r, h = normalize_text(ref).replace(" ", ""), normalize_text(hyp).replace(" ", "")
    return edit_distance(list(r), list(h)) / max(1, len(r))

def generate_one(record):
    inputs = processor.apply_transcription_request(audio=str(record["audio_path"]), language=LANGUAGE)
    inputs = move_batch(inputs)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.decode(generated_ids, return_format="transcription_only")[0]

def evaluate_records(records, split_name, mode_name):
    rows = []
    for i, record in enumerate(records):
        start = time.perf_counter()
        try:
            prediction = generate_one(record)
            error = ""
        except Exception as exc:
            prediction = ""; error = repr(exc)
        rows.append({
            "model": MODEL_ID, "adapter": mode_name, "split": split_name,
            "filename": record.get("filename"), "region": record.get("region"),
            "province_name": record.get("province_name"), "reference": record.get("text"),
            "prediction": prediction, "wer": record_wer(record.get("text"), prediction),
            "cer": record_cer(record.get("text"), prediction), "error": error,
            "inference_seconds": time.perf_counter() - start,
        })
        if i % 5 == 0: print(mode_name, i, "/", len(records))
    return rows

all_results = []
if RUN_BASELINE_EVAL:
    model.eval()
    with model.disable_adapter() if hasattr(model, "disable_adapter") else nullcontext():
        all_results.extend(evaluate_records(test_records, "test", "base"))
if RUN_ADALORA_EVAL:
    model.eval()
    all_results.extend(evaluate_records(test_records, "test", "adalora"))
if all_results:
    result_df = pd.DataFrame(all_results)
    result_df.to_csv(OUTPUT_DIR / "qwen_adalora_predictions.csv", index=False, encoding="utf-8-sig")
    metrics = result_df.groupby("adapter", dropna=False)[["wer", "cer"]].mean().reset_index()
    metrics.to_json(OUTPUT_DIR / "qwen_adalora_metrics.json", orient="records", force_ascii=False, indent=2)
    display(metrics)
else:
    print("Chưa chạy evaluation; bật RUN_BASELINE_EVAL/RUN_ADALORA_EVAL ở cell config.")

# %% [markdown]
# ## Cách đọc kết quả
#
# - So sánh `base` và `adalora` trên **cùng test records**, không đổi sample giữa hai lần chạy.
# - Nếu WER/CER giảm trên test nhưng valid loss tăng, có thể đang overfit; giảm epoch hoặc learning rate.
# - Cấu hình `q_proj/v_proj` chủ yếu kiểm tra thích nghi decoder. Để kiểm tra thích nghi âm học/phương ngữ nghiêm túc, in `linear_names`, chọn thêm các projection thuộc audio encoder/projector rồi chạy một ablation riêng.
# - Kết quả chỉ chứng minh AdaLoRA có chạy được trên Qwen ASR; để kết luận hiệu quả low-resource cần so sánh ít nhất: base, LoRA cùng target/rank, AdaLoRA cùng số update, và full fine-tune nếu đủ VRAM.
