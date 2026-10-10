# %% [markdown]
# # Benchmark Qwen base vs ViMD AdaLoRA adapters
#
# Clones this source repo, installs dependencies, then runs the same ViMD test set through the base model and two adapters.

# %%
from pathlib import Path
import json
import os
import subprocess
import sys

REPO_URL = "https://github.com/SET_OWNER/SET_REPOSITORY.git"
REPO_BRANCH = "main"
REPO_DIR = Path("/kaggle/working/mdv_asr_source")
if "SET_OWNER" in REPO_URL or "SET_REPOSITORY" in REPO_URL:
    raise ValueError("Edit REPO_URL with your GitHub clone URL before running.")
if not REPO_DIR.exists():
    subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
                    REPO_URL, str(REPO_DIR)], check=True)
else:
    print("Using existing clone:", REPO_DIR)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                str(REPO_DIR / "requirements-kaggle.txt")], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
                str(REPO_DIR)], check=True)
sys.path.insert(0, str(REPO_DIR / "src"))

# %%
from kaggle_secrets import UserSecretsClient

env = os.environ.copy()
try:
    env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
except Exception:
    env.pop("HF_TOKEN", None)

# Attach ViMD as a Kaggle Dataset and both fine-tuned models as Kaggle Model inputs.
DATASET_ROOT = Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0")
SMALL_ADAPTER_PATH = Path("/kaggle/input/models/SET_OWNER/SET_SMALL_MODEL/1/epoch_3_adapter")
FULL_ADAPTER_PATH = Path("/kaggle/input/models/SET_OWNER/SET_FULL_MODEL/1/epoch_2_adapter")
OUTPUT_DIR = Path("/kaggle/working/mdv_asr_runs/benchmark_qwen_vimd")

for label, path in [("ViMD dataset", DATASET_ROOT),
                    ("small adapter", SMALL_ADAPTER_PATH),
                    ("full adapter", FULL_ADAPTER_PATH)]:
    if "SET_" in str(path) or not path.exists():
        raise FileNotFoundError(f"Set the Kaggle Input path for {label}: {path}")

CONFIG = json.loads((REPO_DIR / "configs/benchmark/vimd_three_models.json").read_text())
CONFIG["dataset"]["root"] = str(DATASET_ROOT)
CONFIG["dataset"]["allow_download"] = False
CONFIG["output_dir"] = str(OUTPUT_DIR)
CONFIG["benchmark"]["models"][1]["adapter_path"] = str(SMALL_ADAPTER_PATH)
CONFIG["benchmark"]["models"][2]["adapter_path"] = str(FULL_ADAPTER_PATH)
CONFIG_PATH = Path("/kaggle/working/benchmark_qwen_vimd.runtime.json")
CONFIG_PATH.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2))
print("Runtime config:", CONFIG_PATH)
print("Predictions/metrics output:", OUTPUT_DIR)

# %%
subprocess.run([sys.executable, "-m", "mdv_asr.cli", "benchmark", "run",
                "--config", str(CONFIG_PATH)], cwd=REPO_DIR, env=env, check=True)

# %%
import pandas as pd

for name in ("metrics_overall.csv", "metrics_by_region.csv", "metrics_by_province.csv",
             "comparison.json", "coverage.json"):
    path = OUTPUT_DIR / name
    print("\n", name)
    if path.suffix == ".csv" and path.is_file():
        display(pd.read_csv(path))
    elif path.is_file():
        print(path.read_text(encoding="utf-8"))
