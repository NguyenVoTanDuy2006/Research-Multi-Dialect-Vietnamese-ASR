# %% [markdown]
# # Qwen3-ASR 1.7B + PiSSA — ViMD, Kaggle T4 x2
#
# Chạy từng cell từ trên xuống. Dataset là Input, source được clone từ repo.
# Không gắn tỉnh/vùng vào input. Audio → transcript; valid chọn adapter theo CER.
# PiSSA train rank 8; adapter xuất rank 16 là LoRA tương đương trên base gốc,
# không phải tăng rank huấn luyện. Smoke và full train chạy ở tiến trình riêng.

# %%
from pathlib import Path
import subprocess
import sys

REPO_URL = "https://github.com/SET_OWNER/SET_REPOSITORY.git"
REPO_BRANCH = "source_code"
REPO_DIR = Path("/kaggle/working/mdv_asr_source")

if "SET_OWNER" in REPO_URL or "SET_REPOSITORY" in REPO_URL:
    raise ValueError("Điền REPO_URL trước khi chạy.")
if not REPO_DIR.is_dir():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
                    REPO_URL, str(REPO_DIR)], check=True)
else:
    print("Dùng clone hiện có:", REPO_DIR, "— không tự ghi đè code.")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(REPO_DIR / "requirements-kaggle.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR / "src"))

# %%
from dataclasses import replace
import torch
from mdv_asr_module.config import ConfigLoader
from mdv_asr_module.runtime.environment import RuntimeEnvironment

runtime = RuntimeEnvironment(platform="kaggle", project_root=REPO_DIR)
if torch.cuda.device_count() != 2:
    raise RuntimeError("Bật GPU T4 x2 trong Kaggle Settings.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(2)])
# HF_TOKEN chỉ đọc từ environment/Kaggle Secrets trong runtime, không lưu vào JSON.

# %%
DATASET_ROOT = runtime.resolve_dataset_root(
    kaggle_root=None,  # Điền Path cụ thể nếu Kaggle dùng đường dẫn khác.
    kaggle_candidates=[Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0"),
                       Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset")],
)
OUTPUT_DIR = Path("/kaggle/working/vimd_qwen_pissa_full_2epoch")
config_loader = ConfigLoader()
config = config_loader.load(REPO_DIR / "configs/training/qwen_full_vimd_pissa_2epoch.json")
config = replace(config, dataset=replace(config.dataset, root=str(DATASET_ROOT), allow_download=False),
                 output_dir=str(OUTPUT_DIR))
CONFIG_PATH = config_loader.save_resolved(config, OUTPUT_DIR / "resolved_config.json")
print("Dataset:", DATASET_ROOT)
print("Train:", config.train.to_dict())

# %% [markdown]
# ## Index một lần, không copy/giải nén audio
#
# Bước đầu có thể mất vài phút vì đọc metadata của toàn bộ split. Hai rank dùng
# cùng manifest sau đó. Giữ nguyên split gốc; in speaker overlap để kiểm tra.

# %%
from mdv_asr_module.data.sources import DatasetSourceFactory
from mdv_asr_module.data.manifest import ManifestRepository

source = DatasetSourceFactory.create(config.dataset)
repository = ManifestRepository()
MANIFEST_PATH = OUTPUT_DIR / "source_manifest.json"
try:
    fingerprint = source.fingerprint()
    if repository.is_reusable(MANIFEST_PATH, fingerprint):
        index = repository.load(MANIFEST_PATH)
    else:
        index = source.build_index()
        repository.save(index, MANIFEST_PATH)
    for split in ("train", "valid", "test"):
        rows = list(index.for_split(split))
        if not rows:
            raise ValueError(f"Split {split} rỗng; kiểm tra DATASET_ROOT.")
        print(split, len(rows), "clips |", round(sum(r.duration_seconds for r in rows) / 3600, 2), "hours")
    speaker_sets = {split: {r.speaker_id for r in index.for_split(split) if r.speaker_id}
                    for split in ("train", "valid", "test")}
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        print(f"Speaker overlap {a}/{b}:", len(speaker_sets[a] & speaker_sets[b]))
finally:
    source.close()

# %%
# WorkerLauncher tạo environment độc lập; không phụ thuộc biến env ở smoke cell.
from mdv_asr_module.runtime.launcher import WorkerLauncher

launcher = WorkerLauncher(environment=runtime)
LAUNCH = dict(task="asr", config_path=CONFIG_PATH, manifest_path=MANIFEST_PATH, nproc_per_node=2)
launcher.smoke(**LAUNCH)  # Lỗi thì dừng, chưa chạy full train.

# %%
# Chỉ điền đường dẫn .pt do repo tạo khi muốn resume chính xác ở biên epoch.
# Không dùng epoch_N_adapter/ (adapter xuất để inference) để resume training.
RESUME_FROM = None
extra = ["--resume-from", str(RESUME_FROM)] if RESUME_FROM else []
launcher.train(**LAUNCH, extra_args=extra)

# %%
import json
import pandas as pd

display(pd.read_csv(OUTPUT_DIR / "train_history.csv"))
print(json.loads((OUTPUT_DIR / "best_checkpoint.json").read_text()))
print("Native resume:", OUTPUT_DIR / "training_checkpoints")
print("Initial PiSSA:", OUTPUT_DIR / "pissa_initial")
print("Benchmark adapters:", sorted(OUTPUT_DIR.glob("epoch_*_adapter")))

# %%
# Đánh giá adapter tốt nhất trên test qua evaluator/metrics dùng chung.
# Để so sánh với base/AdaLoRA, dùng config benchmark có cùng split test và
# adapter_path = epoch_N_adapter của PiSSA; loader vẫn là PeftModel như AdaLoRA.
launcher.evaluate(**LAUNCH)
