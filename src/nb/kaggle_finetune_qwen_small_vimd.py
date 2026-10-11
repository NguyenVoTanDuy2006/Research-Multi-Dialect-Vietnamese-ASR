# %% [markdown]
# # Qwen3-ASR 1.7B + AdaLoRA — ViMD small, Kaggle T4 x2
#
# Notebook được lưu dạng Python script với cell markers. Chạy từ trên xuống
# trong Kaggle sau khi attach dataset `small-vimd` dạng Arrow và bật 2 GPU.
# Dataset được đọc trực tiếp từ các Arrow shard, không cần manifest/ZIP riêng.
# Audio là input, transcript là target. Tỉnh/vùng chỉ phục vụ kiểm tra metadata.
# Train 3 epoch; chọn adapter theo valid CER; cuối notebook so sánh base và
# adapter trên test. Kết quả được ghi dưới
# `/kaggle/working/outputs/fine-tuning/vimd_qwen_adalora_small_3epoch`.

# %% [code]
from pathlib import Path
import json
import subprocess
import sys

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
# Nạp config để dataset/output root được resolve theo platform top-level.
from mdv_asr_module.config import ConfigLoader
from mdv_asr_module.data.sources import DatasetSourceFactory
from mdv_asr_module.data.manifest import ManifestRepository
config_loader = ConfigLoader()
config = config_loader.load(
    REPO_DIR / "src/mdv_asr_module/configs/training/qwen_small_vimd_adalora.json"
)
config = runtime.resolve_config_paths(config, create_output=True)
DATASET_ROOT = Path(config.dataset.root)
OUTPUT_DIR = Path(config.output_dir)
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"
repository = ManifestRepository()
source = DatasetSourceFactory.create(config.dataset)
try:
    source_fingerprint = source.fingerprint()
    if repository.is_reusable(MANIFEST_PATH, source_fingerprint):
        index = repository.load(MANIFEST_PATH)
        print("Tái sử dụng manifest:", MANIFEST_PATH)
    else:
        index = source.build_index()
        repository.save(index, MANIFEST_PATH, metadata={
            "source_dataset": config.dataset.dataset_id,
            "source_format": config.dataset.kind,
            "dataset_root": str(DATASET_ROOT),
        })
finally:
    source.close()

for split in ("train", "valid", "test"):
    split_rows = index.for_split(split)
    hours = sum(row.duration_seconds for row in split_rows) / 3600
    speakers = len({row.speaker_id for row in split_rows if row.speaker_id})
    print(f"{split}: {len(split_rows):,} clips | {hours:.2f} h | {speakers:,} speakers")
print("Manifest:", MANIFEST_PATH)

# %% [code]
# Dùng config nhỏ đã nằm trong source repo: AdaLoRA, 3 epoch, batch 1/GPU,
# accumulation 8, q_proj/v_proj ở audio encoder và language decoder.
config_path = config_loader.save_resolved(config, OUTPUT_DIR / "resolved_config.json")
print("Config:", config_path)
print("Method:", config.train.method,
      "| epochs:", config.train.epochs,
      "| lr:", config.train.learning_rate)

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
# Chạy hai model trên cùng split test; lưu prediction, CER/WER tổng thể,
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

# %% [markdown]
# ## Dọn source clone khỏi output Kaggle
#
# Cell cuối chỉ chạy sau khi train và benchmark hoàn tất. Kết quả trong
# OUTPUT_DIR nằm ngoài source clone nên vẫn được giữ để tải xuống.

# %% [code]
from mdv_asr_module.runtime.cleanup import NotebookOutputCleaner

if runtime.platform == "kaggle":
    removed = NotebookOutputCleaner(working_dir=runtime.working_dir).remove_source_checkout(
        REPO_DIR, preserve_paths=[OUTPUT_DIR]
    )
    print("Đã xóa source clone:", REPO_DIR if removed else "đã được xóa trước đó")
else:
    print("Bỏ qua dọn source clone vì runtime không phải Kaggle.")
