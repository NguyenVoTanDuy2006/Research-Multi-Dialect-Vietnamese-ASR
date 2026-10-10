# %% [markdown]
# # Qwen3-ASR 1.7B + AdaLoRA — ViMD small, Kaggle T4 x2
#
# Notebook được lưu dạng Python script với cell markers. Chạy từ trên xuống
# trong Kaggle sau khi attach dataset package `vimd_small` vào Input và bật 2 GPU.
# Dataset đang dùng là package 31.25 giờ: train/valid/test cùng metadata và ZIP.
# Audio là input, transcript là target. Tỉnh/vùng chỉ phục vụ kiểm tra metadata.
# Train 3 epoch; chọn adapter theo valid CER; cuối notebook so sánh base và
# adapter trên test. Kết quả được ghi dưới `/kaggle/working/vimd_qwen_adalora_small_3epoch`.

# %% [code]
from pathlib import Path
import json
import subprocess
import sys
import zipfile

# Repo GitHub hiện tại; Kaggle clone branch source_code.
REPO_URL = "https://github.com/NguyenVoTanDuy2006/Research-Multi-Dialect-Vietnamese-ASR.git"
REPO_BRANCH = "source_code"
REPO_DIR = Path("/kaggle/working/mdv_asr_source")

if not REPO_DIR.is_dir():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
                    REPO_URL, str(REPO_DIR)], check=True)
else:
    print("Dùng source clone hiện có:", REPO_DIR)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(REPO_DIR / "requirements-kaggle.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
                str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR / "src"))

# %% [code]
import torch
from mdv_asr_module.runtime.environment import RuntimeEnvironment

if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(f"Cần bật 2 GPU Kaggle; phát hiện {torch.cuda.device_count()} GPU.")
gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
print("GPUs:", gpu_names)
if not all("T4" in name.upper() for name in gpu_names):
    print("Lưu ý: notebook dùng cấu hình batch/precision tối ưu cho T4 x2.")

runtime = RuntimeEnvironment(platform="kaggle", project_root=REPO_DIR)
print("HF_TOKEN available:", bool(runtime.get_hf_token()))

# %% [code]
# Để None cho notebook tự tìm sample_metadata.json trong /kaggle/input.
# Nếu attach nhiều package có cùng tên manifest, đặt Path root cụ thể.
DATASET_ROOT = None
EXPECTED_SPLIT_COUNTS = {"train": 4676, "valid": 485, "test": 484}

if DATASET_ROOT is None:
    manifests = sorted(Path("/kaggle/input").rglob("sample_metadata.json"))
    package_roots = [p.parent.resolve() for p in manifests
                     if all((p.parent / name).is_file() for name in ("train.zip", "valid.zip", "test.zip"))]
    package_roots = sorted(set(package_roots))
    if len(package_roots) != 1:
        raise RuntimeError(
            f"Cần đúng một package ViMD small có sample_metadata.json + 3 ZIP; "
            f"tìm thấy {package_roots}. Nếu có nhiều package, điền DATASET_ROOT."
        )
    DATASET_ROOT = package_roots[0]
else:
    DATASET_ROOT = Path(DATASET_ROOT).expanduser().resolve()

for required in ("sample_metadata.json", "train.zip", "valid.zip", "test.zip"):
    if not (DATASET_ROOT / required).is_file():
        raise FileNotFoundError(f"Thiếu {required} trong {DATASET_ROOT}")
print("ViMD small package:", DATASET_ROOT)

# %% [markdown]
# ## Đọc metadata và dựng manifest nhẹ
#
# Không giải nén audio và không đọc toàn bộ nội dung 3 ZIP. Dùng metadata đã có
# duration/transcript/audio member để tạo manifest cho worker; kiểm tra số mẫu,
# speaker và sự tồn tại của từng đường dẫn bên trong ZIP.

# %% [code]
from collections import Counter
from mdv_asr_module.config import ConfigLoader
from mdv_asr_module.data.index import DatasetIndex
from mdv_asr_module.data.manifest import ManifestRepository
from mdv_asr_module.data.records import AudioLocator, SampleRecord

metadata_path = DATASET_ROOT / "sample_metadata.json"
package = json.loads(metadata_path.read_text(encoding="utf-8"))
rows = package.get("records", package.get("samples", []))
if not isinstance(rows, list) or not rows:
    raise ValueError("sample_metadata.json không có records hợp lệ.")

counts = Counter(str(row.get("split", "")) for row in rows)
if dict(counts) != EXPECTED_SPLIT_COUNTS:
    raise ValueError(f"Số mẫu không đúng package ViMD small dự kiến: {dict(counts)}")

zip_members = {}
records = []
for row in rows:
    split = str(row["split"])
    archive_name = str(row.get("archive") or f"{split}.zip")
    archive_path = (DATASET_ROOT / archive_name).resolve()
    if archive_name not in zip_members:
        with zipfile.ZipFile(archive_path) as archive:
            zip_members[archive_name] = set(archive.namelist())
    member = str(row.get("audio_path") or row.get("audio_member") or "")
    if not member or member not in zip_members[archive_name]:
        raise FileNotFoundError(f"Không tìm thấy audio member {member!r} trong {archive_name}")
    records.append(SampleRecord(
        dataset_id="vimd_small", sample_id=str(row["sample_id"]), split=split,
        filename=str(row.get("filename") or Path(member).name), text=str(row.get("text") or ""),
        duration_seconds=float(row["duration_seconds"]),
        audio_locator=AudioLocator(kind="zip", path=str(archive_path), member=member),
        region=row.get("region"), province_name=row.get("province_name"),
        province_code=row.get("province_code"), speaker_id=row.get("speakerID"),
        gender=str(row.get("gender") or "unknown"), source_gender=str(row.get("gender") or "unknown"),
        metadata={"archive": archive_name, "sample_rate": row.get("sample_rate"),
                  "channels": row.get("channels")},
    ))

index = DatasetIndex(records)
OUTPUT_DIR = Path("/kaggle/working/vimd_qwen_adalora_small_3epoch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"
ManifestRepository().save(index, MANIFEST_PATH, metadata={
    "source_dataset": package.get("source_dataset"),
    "source_license": package.get("source_license"),
    "package_root": str(DATASET_ROOT),
})

for split in ("train", "valid", "test"):
    split_rows = index.for_split(split)
    hours = sum(row.duration_seconds for row in split_rows) / 3600
    speakers = len({row.speaker_id for row in split_rows if row.speaker_id})
    print(f"{split}: {len(split_rows):,} clips | {hours:.2f} h | {speakers:,} speakers")
print("Manifest:", MANIFEST_PATH)
print("ZIP paths checked:", sum(len(value) for value in zip_members.values()), "members")

# %% [code]
# Dùng config nhỏ đã nằm trong source repo: AdaLoRA, 3 epoch, batch 1/GPU,
# accumulation 8, q_proj/v_proj ở audio encoder và language decoder.
config_loader = ConfigLoader()
config = config_loader.load(REPO_DIR / "configs/training/qwen_small_vimd_3epoch.json")
config_data = config.to_dict()
config_data["dataset"]["root"] = str(DATASET_ROOT)
config_data["output_dir"] = str(OUTPUT_DIR)
config_data["train"]["world_size"] = 2
config_data["train"]["method"] = "adalora"
config_path = config_loader.save_resolved(
    config_loader.from_dict(config_data), OUTPUT_DIR / "resolved_config.json")
print("Config:", config_path)
print("Method:", config_data["train"].get("method", "adalora"),
      "| epochs:", config_data["train"]["epochs"],
      "| lr:", config_data["train"]["learning_rate"])

# %% [markdown]
# ## Smoke test
#
# Chạy forward/backward và decode trên 2 GPU trước. Cell train chỉ bắt đầu nếu
# smoke chạy hết, để phát hiện lỗi model/dataset hoặc thiếu VRAM sớm.

# %% [code]
from mdv_asr_module.runtime.launcher import WorkerLauncher

launcher = WorkerLauncher(environment=runtime)
LAUNCH = {"task": "asr", "config_path": config_path,
          "manifest_path": MANIFEST_PATH, "nproc_per_node": 2}
launcher.smoke(**LAUNCH)

# %% [markdown]
# ## Fine-tuning
#
# Huấn luyện đủ 3 epoch trên train; mỗi epoch đo valid CER/WER và lưu adapter.
# Có thể đặt `RESUME_FROM` thành file `training_checkpoints/epoch_XXXX.pt` để
# tiếp tục từ ranh giới epoch sau khi Kaggle session bị ngắt.

# %% [code]
RESUME_FROM = None
extra_args = ["--resume-from", str(RESUME_FROM)] if RESUME_FROM else []
launcher.train(**LAUNCH, extra_args=extra_args)

# %% [code]
import pandas as pd

history_path = OUTPUT_DIR / "train_history.csv"
if not history_path.is_file():
    raise FileNotFoundError(f"Không thấy history sau train: {history_path}")
display(pd.read_csv(history_path))
best = json.loads((OUTPUT_DIR / "best_checkpoint.json").read_text(encoding="utf-8"))
print("Best epoch:", best["epoch"], "| validation metrics:", best["metrics"])
print("Adapter folders:", sorted(path.name for path in OUTPUT_DIR.glob("epoch_*_adapter")))
print("Resume checkpoints:", sorted(path.name for path in (OUTPUT_DIR / "training_checkpoints").glob("*.pt")))

# %% [markdown]
# ## So sánh trên test: Qwen gốc và AdaLoRA
#
# Chạy hai model trên cùng 484 mẫu test; lưu prediction, CER/WER tổng thể,
# so sánh paired và báo cáo theo tỉnh/vùng trong OUTPUT_DIR.

# %% [code]
benchmark_data = config.to_dict()
benchmark_data["output_dir"] = str(OUTPUT_DIR)
benchmark_data["benchmark"] = {
    "models": [
        {"key": "qwen_base", "backend": "qwen3_asr",
         "model_id": config.model.model_id, "dtype": config.model.dtype,
         "display_name": "Qwen3-ASR 1.7B (pretrained)"},
        {"key": "qwen_small_vimd_adalora", "backend": "qwen3_asr",
         "model_id": config.model.model_id, "dtype": config.model.dtype,
         "adapter_path": str(Path(best["adapter_path"]).resolve()),
         "display_name": "Qwen3-ASR 1.7B + AdaLoRA (ViMD small)"},
    ],
    "split": "test", "resume": True, "flush_every": 10,
}
benchmark_path = config_loader.save_resolved(
    config_loader.from_dict(benchmark_data), OUTPUT_DIR / "benchmark_config.json")
launcher.evaluate(task="benchmark", config_path=benchmark_path,
                  manifest_path=MANIFEST_PATH, nproc_per_node=1)

print("Kết quả:", OUTPUT_DIR)
print("Predictions:", OUTPUT_DIR / "predictions.csv")
print("CER/WER tổng thể:", OUTPUT_DIR / "metrics_overall.csv")
print("Paired comparison:", OUTPUT_DIR / "comparison.json")
print("Theo vùng:", OUTPUT_DIR / "metrics_by_region.csv")
print("Theo tỉnh:", OUTPUT_DIR / "metrics_by_province.csv")
display(pd.read_csv(OUTPUT_DIR / "metrics_overall.csv"))
display(pd.read_csv(OUTPUT_DIR / "metrics_by_province.csv").head(10))
