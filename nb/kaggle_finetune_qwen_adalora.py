# %% [markdown]
# # Qwen3-ASR 1.7B + AdaLoRA fine-tuning on Kaggle
#
# This notebook source clones the focused source repo, installs runtime dependencies without reinstalling Kaggle's Torch/NumPy, then runs smoke → train → best-checkpoint evaluation.

# %%
from pathlib import Path
import json
import os
import subprocess
import sys

# Set this after creating your own GitHub repository from this folder.
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
import torch
from kaggle_secrets import UserSecretsClient

if not torch.cuda.is_available():
    raise RuntimeError("Enable a GPU accelerator in Kaggle settings.")
if torch.cuda.device_count() != 2:
    raise RuntimeError(f"This config expects 2 GPUs; found {torch.cuda.device_count()}.")
print("GPUs:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

env = os.environ.copy()
try:
    env["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    print("HF_TOKEN loaded from Kaggle Secrets.")
except Exception:
    env.pop("HF_TOKEN", None)
    print("No HF_TOKEN secret found; continuing for public model access.")

# %%
# Select full ViMD; attach it under Kaggle Input. If Kaggle uses a different mount path,
# set DATASET_ROOT explicitly in this cell.
DATASET_ROOT = None
for candidate in [
    Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0"),
    Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset"),
]:
    if candidate.is_dir() and next(candidate.rglob("*.arrow"), None) is not None:
        DATASET_ROOT = candidate.resolve()
        break
if DATASET_ROOT is None:
    raise FileNotFoundError("Attach thanhlamdev/ViMD in Kaggle Input or set DATASET_ROOT.")

OUTPUT_DIR = Path("/kaggle/working/mdv_asr_runs/qwen_full_vimd_2epoch")
CONFIG = json.loads((REPO_DIR / "configs/training/qwen_full_vimd_2epoch.json").read_text())
CONFIG["dataset"]["root"] = str(DATASET_ROOT)
CONFIG["dataset"]["allow_download"] = False
CONFIG["output_dir"] = str(OUTPUT_DIR)
CONFIG["train"]["world_size"] = 2
CONFIG_PATH = Path("/kaggle/working/qwen_full_vimd_2epoch.runtime.json")
CONFIG_PATH.write_text(json.dumps(CONFIG, ensure_ascii=False, indent=2))
print("Dataset:", DATASET_ROOT)
print("Output:", OUTPUT_DIR)
print("Train config:", CONFIG_PATH)

# %% [markdown]
# ## Smoke test
#
# This checks two-GPU distributed initialization, model forward/backward, AdaLoRA updates, and a decode before the full run.

# %%
subprocess.run([sys.executable, "-m", "mdv_asr_module.cli", "train", "smoke",
                "--config", str(CONFIG_PATH)], cwd=REPO_DIR, env=env, check=True)

# %% [markdown]
# ## Train and validate
#
# The trainer selects the best epoch using validation CER and writes adapter checkpoints plus a resumable state to `OUTPUT_DIR`.

# %%
subprocess.run([sys.executable, "-m", "mdv_asr_module.cli", "train", "run",
                "--config", str(CONFIG_PATH)], cwd=REPO_DIR, env=env, check=True)

# %%
history_path = OUTPUT_DIR / "train_history.csv"
if history_path.is_file():
    import pandas as pd
    display(pd.read_csv(history_path))
print("Best checkpoint metadata:", OUTPUT_DIR / "best_checkpoint.json")

# %%
# Evaluate the selected adapter on the configured test split.
subprocess.run([sys.executable, "-m", "mdv_asr_module.cli", "train", "evaluate",
                "--config", str(CONFIG_PATH)], cwd=REPO_DIR, env=env, check=True)
print("Run outputs:", OUTPUT_DIR)
