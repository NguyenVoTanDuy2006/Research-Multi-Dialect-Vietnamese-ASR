# %% [markdown]
# # Qwen3-ASR 1.7B + PiSSA — ViMD small, Kaggle T4 x2
#
# Giữ luồng cell của notebook fine-tuning cũ: clone source → cấu hình dataset/model
# → tạo hoặc tái dùng manifest → smoke test → train/resume → xem history/checkpoint
# → evaluate best checkpoint → dọn source clone. Dùng small ViMD, PiSSA 3 epoch.
# Audio là input, transcript là target; tỉnh/vùng chỉ dùng để phân tích metadata.

# %%
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/NguyenVoTanDuy2006/Research-Multi-Dialect-Vietnamese-ASR.git"
REPO_BRANCH = "source_code"
REPO_DIR = Path("/kaggle/working/mdv_asr_source")

if not REPO_DIR.is_dir():
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
         REPO_URL, str(REPO_DIR)],
        check=True,
    )
else:
    print("Dùng source clone hiện có:", REPO_DIR)

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r",
     str(REPO_DIR / "requirements-kaggle.txt")],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)],
    check=True,
)
sys.path.insert(0, str(REPO_DIR / "src"))

# %%
import torch

from mdv_asr_module.configs import ConfigLoader
from mdv_asr_module.runtime.environment import RuntimeEnvironment

runtime = RuntimeEnvironment(project_root=REPO_DIR)
if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
    raise RuntimeError(f"Cần bật 2 GPU Kaggle; hiện có {torch.cuda.device_count()}.")
gpu_names = [torch.cuda.get_device_name(i) for i in range(2)]
print("GPUs:", gpu_names)
if not all("T4" in name.upper() for name in gpu_names):
    print("Lưu ý: cấu hình train được thiết kế cho Tesla T4 x2.")
print("HF_TOKEN available:", bool(runtime.get_hf_token()))

# %%
# Chỉnh đường dẫn này để chọn cấu hình fine-tuning muốn chạy.
CONFIG_FILE = Path(REPO_DIR / "src/mdv_asr_module/configs/training/qwen_small_vimd_pissa.json")

# %%
config_loader = ConfigLoader()
config = config_loader.load(CONFIG_FILE)
runtime = RuntimeEnvironment(platform=config.platform, project_root=REPO_DIR)
config = runtime.resolve_config_paths(config, create_output=True)
if config.dataset.root is None:
    raise ValueError("Cần khai báo dataset.root trong file config trước khi train.")
DATASET_ROOT = Path(config.dataset.root)
OUTPUT_DIR = Path(config.output_dir)
CONFIG_PATH = config_loader.save_resolved(config, OUTPUT_DIR / "resolved_config.json")
print("Dataset:", DATASET_ROOT)
print("Output:", OUTPUT_DIR)
print("Method:", config.train.method, "| epochs:", config.train.epochs)
print("PiSSA:", config.train.pissa.to_dict())

# %% [markdown]
# ## Tạo hoặc tái sử dụng manifest
#
# Source đọc metadata và chỉ mục audio trực tiếp từ Arrow/Parquet; không cần
# giải nén hay copy audio. Lần quét đầu có thể mất thời gian; các lần sau dùng
# lại manifest nếu fingerprint của các shard không đổi.

# %%
from mdv_asr_module.data.manifest import ManifestRepository
from mdv_asr_module.data.sources import DatasetSourceFactory

source = DatasetSourceFactory.create(config.dataset)
repository = ManifestRepository()
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"
try:
    fingerprint = source.fingerprint()
    if repository.is_reusable(MANIFEST_PATH, fingerprint):
        index = repository.load(MANIFEST_PATH)
        print("Đã tái sử dụng manifest:", MANIFEST_PATH)
    else:
        index = source.build_index()
        repository.save(index, MANIFEST_PATH)
        print("Đã tạo manifest:", MANIFEST_PATH)

    for split in ("train", "valid", "test"):
        rows = list(index.for_split(split))
        if not rows:
            raise ValueError(f"Split {split} rỗng; hãy kiểm tra các shard small ViMD.")
        hours = sum(row.duration_seconds for row in rows) / 3600
        speakers = len({row.speaker_id for row in rows if row.speaker_id})
        print(f"{split}: {len(rows):,} clips | {hours:.2f} audio hours | {speakers:,} speakers")

    speaker_sets = {
        split: {row.speaker_id for row in index.for_split(split) if row.speaker_id}
        for split in ("train", "valid", "test")
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        print(f"Speaker overlap {left}/{right}:", len(speaker_sets[left] & speaker_sets[right]))
finally:
    source.close()

# %%
# PEFT kiểm tra mọi LoRA dispatcher khi gắn adapter. TorchAO cũ có thể làm
# bước này lỗi dù training không dùng quantization; worker torchrun chạy process mới.
import importlib
import importlib.util

if importlib.util.find_spec("torchao") is not None:
    uninstall = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        check=False,
    )
    if uninstall.returncode != 0:
        raise RuntimeError("Không gỡ được torchao; dừng trước khi khởi tạo launcher.")
    importlib.invalidate_caches()
    if importlib.util.find_spec("torchao") is not None:
        raise RuntimeError("torchao vẫn còn được cài; dừng trước khi khởi tạo launcher.")
    print("Đã gỡ TorchAO không tương thích; tiếp tục dùng LoRA/PiSSA thường.")
else:
    print("TorchAO không được cài; tiếp tục dùng LoRA/PiSSA thường.")

# %%
# Smoke phải chạy thành công trước khi bắt đầu full training.
from mdv_asr_module.runtime.launcher import WorkerLauncher

launcher = WorkerLauncher(environment=runtime)
LAUNCH = {
    "task": "asr",
    "config_path": CONFIG_PATH,
    "manifest_path": MANIFEST_PATH,
    "nproc_per_node": 2,
}
launcher.smoke(**LAUNCH)

# %%
# Có thể resume từ checkpoint .pt ở ranh giới epoch; để None nếu train mới.
# Không truyền thư mục epoch_N_adapter vì đó là adapter inference, không phải resume state.
RESUME_FROM = None
extra_args = ["--resume-from", str(RESUME_FROM)] if RESUME_FROM else []
launcher.train(**LAUNCH, extra_args=extra_args)

# %%
import json
import pandas as pd

history_path = OUTPUT_DIR / "train_history.csv"
best_path = OUTPUT_DIR / "best_checkpoint.json"
if not history_path.is_file() or not best_path.is_file():
    raise FileNotFoundError("Training xong nhưng thiếu train_history.csv hoặc best_checkpoint.json.")

display(pd.read_csv(history_path))
best = json.loads(best_path.read_text(encoding="utf-8"))
print("Best checkpoint:", best)
print("Resume checkpoints:", sorted((OUTPUT_DIR / "training_checkpoints").glob("*.pt")))
print("PiSSA initialization:", OUTPUT_DIR / "pissa_initial")
print("Inference adapters:", sorted(OUTPUT_DIR.glob("epoch_*_adapter")))

# %%
# Đánh giá best PiSSA checkpoint trên split test bằng evaluator/metrics dùng chung.
# Kết quả được ghi dưới OUTPUT_DIR (predictions, CER/WER tổng thể và theo tỉnh/vùng).
launcher.evaluate(**LAUNCH)
for name in (
    "predictions.csv", "metrics_overall.csv", "metrics_by_region.csv",
    "metrics_by_province.csv",
):
    path = OUTPUT_DIR / name
    if path.is_file():
        print(name, "->", path)

# %% [markdown]
# ## Dọn source clone khỏi Kaggle Output
#
# Chạy cell cuối sau khi train và evaluate hoàn tất. Kết quả nằm ngoài clone và
# được giữ lại; chỉ source checkout bị xóa.

# %%
from mdv_asr_module.runtime.cleanup import NotebookOutputCleaner

if runtime.platform == "kaggle":
    removed = NotebookOutputCleaner(working_dir=runtime.working_dir).remove_source_checkout(
        REPO_DIR, preserve_paths=[OUTPUT_DIR]
    )
    print("Đã dọn source clone:", REPO_DIR if removed else "đã được dọn trước đó")
else:
    print("Bỏ qua cleanup vì runtime không phải Kaggle.")
