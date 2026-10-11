
import argparse
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
                audio_path = DATA_ROOT / audio_path
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


def train():
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
    best_cer = float("inf")
    optimizer.zero_grad(set_to_none=True)

    attempted_steps = 0
    skipped_steps = 0
    
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
                        f"epoch={epoch + 1}/{CFG['epochs']} "
                        f"batch={batch_index + 1}/{n_batches} "
                        f"attempt={attempted_steps}/{total_steps} "
                        f"successful={global_step} "
                        f"scale={old_scale}->{new_scale} "
                        f"grad_norm={float(grad_norm.detach())}",
                        flush=True,
                    )
                
                optimizer.zero_grad(set_to_none=True)
        barrier()
        valid_rows = evaluate(ddp.module, processor, "valid", f"adalora_epoch_{epoch+1}")
        if RANK == 0:
            cer_errors = sum(err_counts(r["reference"], r["prediction"])[2] for r in valid_rows)
            cer_units = sum(err_counts(r["reference"], r["prediction"])[3] for r in valid_rows)
            wer_errors = sum(err_counts(r["reference"], r["prediction"])[0] for r in valid_rows)
            wer_units = sum(err_counts(r["reference"], r["prediction"])[1] for r in valid_rows)
            val_cer, val_wer = cer_errors / cer_units, wer_errors / wer_units
            mean_loss = float(np.mean(losses)) if losses else float("nan")
            history.append({"epoch": epoch + 1, "optimizer_steps": global_step,
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
            cfgdata.update({"optimizer_steps_completed": global_step, "optimizer_steps_planned": total_steps,
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
        raise RuntimeError(
            f"Đã thử {attempted_steps} bước, dự kiến {total_steps}."
        )
    
    if global_step + skipped_steps != attempted_steps:
        raise RuntimeError(
            f"Đếm bước không khớp: successful={global_step}, "
            f"skipped={skipped_steps}, attempted={attempted_steps}."
        )
    
    if RANK == 0 and skipped_steps:
        log(
            f"Train đã đi hết dữ liệu; GradScaler bỏ qua {skipped_steps}/"
            f"{attempted_steps} lần cập nhật."
        )

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


def test():
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
    if args.mode == "smoke":
        smoke()
    elif args.mode == "train":
        train()
    else:
        test()
    barrier()
    if IS_DISTRIBUTED:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
