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
    """Resolve runtime platform and common input/output paths.

    `platform="auto"` detects the active runtime. Pass `"local"` or `"kaggle"`
    to make notebook/script behavior reproducible when the code is run from a
    different host (for example, local paths while testing Kaggle configs).
    """

    PLATFORMS = {"auto", "local", "kaggle", "colab"}

    def __init__(self, working_dir: str | Path | None = None, *,
                 platform: str = "auto", project_root: str | Path | None = None):
        if platform not in self.PLATFORMS:
            raise ValueError(f"Unsupported runtime platform: {platform!r}")
        self.platform_override = platform
        self.project_root = Path(project_root or Path.cwd()).expanduser().resolve()
        if working_dir is not None:
            self.working_dir = Path(working_dir).expanduser().resolve()
        elif self.platform == "kaggle":
            self.working_dir = Path(
                os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
            ).expanduser()
        else:
            self.working_dir = self.project_root

    @property
    def platform(self) -> str:
        if self.platform_override != "auto":
            return self.platform_override
        if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
            return "kaggle"
        if os.environ.get("COLAB_RELEASE_TAG"):
            return "colab"
        return "local"

    def resolve_dataset_root(self, *, local_root: str | Path | None = None,
                             kaggle_root: str | Path | None = None,
                             kaggle_candidates: Iterable[str | Path] = ()) -> Path:
        """Select and validate the dataset root appropriate for this runtime."""
        configured = kaggle_root if self.platform == "kaggle" else local_root
        candidates = ([configured] if configured is not None else [])
        if self.platform == "kaggle":
            candidates.extend(kaggle_candidates)
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.is_dir():
                return path.resolve()
        searched = [str(path) for path in candidates]
        label = "Kaggle Input" if self.platform == "kaggle" else "local"
        raise FileNotFoundError(
            f"No dataset root found for {label} runtime. Set the corresponding "
            f"dataset root; searched: {searched}"
        )

    def output_path(self, relative_path: str | Path, *,
                    local_root: str | Path | None = None,
                    kaggle_root: str | Path | None = None,
                    create: bool = False) -> Path:
        """Resolve a reusable output location for local or Kaggle execution."""
        relative = Path(relative_path).expanduser()
        if relative.is_absolute():
            destination = relative
        else:
            if self.platform == "kaggle":
                base = Path(kaggle_root).expanduser() if kaggle_root else self.working_dir
            else:
                base = Path(local_root).expanduser() if local_root else self.working_dir
            destination = base / relative
        destination = destination.resolve()
        if create:
            destination.mkdir(parents=True, exist_ok=True)
        return destination

    def detect(self) -> RuntimeDescription:
        try:
            import torch

            gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
            names = tuple(torch.cuda.get_device_name(i) for i in range(gpu_count))
        except ImportError:
            gpu_count, names = 0, ()
        platform = self.platform
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

