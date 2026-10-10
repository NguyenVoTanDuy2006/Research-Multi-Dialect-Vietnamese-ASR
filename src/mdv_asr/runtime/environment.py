from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RuntimeDescription:
    platform: str
    python: str
    gpu_count: int
    gpu_names: tuple[str, ...]
    working_free_bytes: int | None


class RuntimeEnvironment:
    def __init__(self, working_dir: str | Path | None = None):
        self.working_dir = Path(working_dir or os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))

    def detect(self) -> RuntimeDescription:
        try:
            import torch

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            names = tuple(torch.cuda.get_device_name(i) for i in range(gpu_count))
        except ImportError:
            gpu_count, names = 0, ()
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            platform = "kaggle"
        elif os.environ.get("COLAB_RELEASE_TAG"):
            platform = "colab"
        else:
            platform = "local"
        free = shutil.disk_usage(self.working_dir).free if self.working_dir.exists() else None
        return RuntimeDescription(platform, sys.version.split()[0], gpu_count, names, free)

    def validate(self, *, require_gpu: bool = False, expected_gpu_count: int | None = None,
                 required_paths: Iterable[str | Path] = ()) -> RuntimeDescription:
        description = self.detect()
        if require_gpu and not description.gpu_count:
            raise RuntimeError("This task requires a CUDA GPU")
        if expected_gpu_count is not None and description.gpu_count != expected_gpu_count:
            raise RuntimeError(f"Expected {expected_gpu_count} GPUs, found {description.gpu_count}")
        missing = [str(path) for path in required_paths if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"Required paths do not exist: {missing}")
        return description

    @staticmethod
    def get_hf_token() -> str | None:
        token = os.environ.get("HF_TOKEN")
        if token:
            return token
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            try:
                from kaggle_secrets import UserSecretsClient

                return UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                return None
        return None

    def subprocess_env(self) -> dict[str, str]:
        result = os.environ.copy()
        token = self.get_hf_token()
        if token:
            result["HF_TOKEN"] = token
        else:
            result.pop("HF_TOKEN", None)
        return result

    def describe(self) -> dict:
        value = self.detect()
        return {"platform": value.platform, "python": value.python,
                "gpu_count": value.gpu_count, "gpu_names": list(value.gpu_names),
                "working_free_bytes": value.working_free_bytes,
                "hf_token_available": bool(self.get_hf_token())}

