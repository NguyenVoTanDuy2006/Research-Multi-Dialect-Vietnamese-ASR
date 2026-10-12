# %% [markdown]
# # Benchmark Qwen3-ASR models on a dataset test split
#
# Nhập một đường dẫn config dataset và danh sách config model. Notebook tự tái sử
# dụng manifest, benchmark trên split `test`, rồi lưu predictions, WER/CER tổng thể,
# theo vùng/tỉnh, coverage và comparison vào `/kaggle/working/outputs/benchmark`.

# %% [code]
from pathlib import Path
import importlib
import importlib.util
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/NguyenVoTanDuy2006/Research-Multi-Dialect-Vietnamese-ASR.git"
REPO_BRANCH = "source_code"
REPO_DIR = Path("/kaggle/working/mdv_asr_source")

if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
    print("Đã xóa source clone cũ:", REPO_DIR)

subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
     REPO_URL, str(REPO_DIR)],
    check=True,
)

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

# %% [code]
import pandas as pd
import torch

from mdv_asr_module.configs import ConfigLoader
from mdv_asr_module.runtime.environment import RuntimeEnvironment

runtime = RuntimeEnvironment(project_root=REPO_DIR)
description = runtime.validate(require_gpu=True)
print("Platform:", description.platform, "| GPUs:", description.gpu_names)
print("HF_TOKEN available:", bool(runtime.get_hf_token()))

# %% [markdown]
# ## Config đầu vào
#
# Đổi `DATASET_CONFIG_FILE` để chọn dataset; thêm/bớt đường dẫn trong
# `MODEL_CONFIG_FILES` để chọn model. Model đầu tiên được dùng làm baseline trong
# file comparison. Adapter Kaggle phải được attach vào notebook dưới dạng Input.

# %% [code]
DATASET_CONFIG_FILE = REPO_DIR / "src/mdv_asr_module/configs/datasets/vimd_full.json"
MODEL_CONFIG_FILES = [
    REPO_DIR / "src/mdv_asr_module/configs/models/qwen_vimd_base.json",
    REPO_DIR / "src/mdv_asr_module/configs/models/qwen_vimd_adalora_small.json",
    REPO_DIR / "src/mdv_asr_module/configs/models/qwen_vimd_adalora_full.json",
    REPO_DIR / "src/mdv_asr_module/configs/models/qwen_vimd_pissa_small.json",
    REPO_DIR / "src/mdv_asr_module/configs/models/qwen_vimd_pissa_full.json",
]
TEST_SPLIT = "test"
RESUME = True
FLUSH_EVERY = 10
OUTPUT_DIR = Path("/kaggle/working/outputs/benchmark/qwen_four_models")

# %% [code]
config_loader = ConfigLoader()
config = config_loader.load_benchmark_configs(
    DATASET_CONFIG_FILE,
    MODEL_CONFIG_FILES,
    output_dir=OUTPUT_DIR,
    split=TEST_SPLIT,
    resume=RESUME,
    flush_every=FLUSH_EVERY,
)
runtime = RuntimeEnvironment(platform=config.platform, project_root=REPO_DIR)
config = runtime.resolve_config_paths(config, create_output=True)
OUTPUT_DIR = Path(config.output_dir)

if not config.dataset.root:
    raise ValueError("Dataset config must define dataset.root.")
runtime.validate(
    require_gpu=True,
    required_paths=[config.dataset.root, *(
        model.adapter_path for model in config.benchmark.models if model.adapter_path
    )],
)
print("Dataset config:", DATASET_CONFIG_FILE)
print("Dataset root:", config.dataset.root)
print("Split:", config.benchmark.split)
print("Models:")
for model in config.benchmark.models:
    print("-", model.display_name or model.key, "|", model.key)
print("Output:", OUTPUT_DIR)

# %% [markdown]
# ## Chuẩn bị dữ liệu
#
# Lần đầu source có thể quét các shard để lập manifest; các lần sau tái sử dụng
# cache theo fingerprint, không giải nén/copy audio.

# %% [code]
from mdv_asr_module.workflows.dataset import DatasetWorkflow

dataset_workflow = DatasetWorkflow.from_config(config.dataset)
index = dataset_workflow.scan(use_cache=True)
test_rows = list(index.for_split(TEST_SPLIT))
if not test_rows:
    dataset_workflow.close()
    raise ValueError(f"Split {TEST_SPLIT!r} rỗng trong dataset config.")
print("Manifest:", len(index.records), "records | test:", len(test_rows), "clips")

# %% [code]
# PEFT 0.21 không tương thích TorchAO 0.10 có sẵn trên một số Kaggle image.
# Benchmark các adapter thường không cần TorchAO; bỏ package lỗi thời nếu có.
if importlib.util.find_spec("torchao") is not None:
    uninstall = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
        check=False,
    )
    if uninstall.returncode != 0:
        dataset_workflow.close()
        raise RuntimeError("Không gỡ được TorchAO không tương thích.")
    importlib.invalidate_caches()
    if importlib.util.find_spec("torchao") is not None:
        dataset_workflow.close()
        raise RuntimeError("TorchAO vẫn còn được cài; dừng trước khi load adapter.")
    print("Đã gỡ TorchAO không tương thích.")

# %% [markdown]
# ## Chạy benchmark
#
# Mỗi model được nạp và giải phóng tuần tự để dùng được trên một GPU. Có thể chạy
# lại cell này; predictions đã lưu sẽ được resume theo cấu hình.

# %% [code]
from mdv_asr_module.workflows.benchmark import ASRBenchmarkWorkflow

benchmark = ASRBenchmarkWorkflow(config, dataset_workflow, output_dir=OUTPUT_DIR)
try:
    results = benchmark.run_test()
finally:
    benchmark.close()
    dataset_workflow.close()

for name in (
    "predictions.csv", "metrics_overall.csv", "metrics_by_region.csv",
    "metrics_by_province.csv", "coverage.json", "comparison.json",
):
    path = OUTPUT_DIR / name
    if path.is_file():
        print(name, "->", path)

overall_path = OUTPUT_DIR / "metrics_overall.csv"
if overall_path.is_file():
    display(pd.read_csv(overall_path))

# %% [markdown]
# ## Dọn source clone khỏi Kaggle Output
#
# Chạy sau khi benchmark xong. Chỉ xóa checkout source; giữ nguyên artifacts trong
# `OUTPUT_DIR` để notebook output tải về không kèm source code.

# %% [code]
from mdv_asr_module.runtime.cleanup import NotebookOutputCleaner

if runtime.platform == "kaggle":
    removed = NotebookOutputCleaner(working_dir=runtime.working_dir).remove_source_checkout(
        REPO_DIR, preserve_paths=[OUTPUT_DIR]
    )
    print("Đã dọn source clone:", REPO_DIR if removed else "đã được dọn trước đó")
