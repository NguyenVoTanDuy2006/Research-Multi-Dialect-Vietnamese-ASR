from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, replace
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

    def resolve_config_paths(self, config, *, create_output: bool = False):
        """Return a config with dataset/output paths resolved for its runtime.

        Relative ``dataset.root`` paths use ``/kaggle/input`` on Kaggle and the
        project root elsewhere. Relative ``dataset.cache_dir`` and ``output_dir``
        paths use ``/kaggle/working`` on Kaggle and the runtime working directory
        elsewhere. Absolute paths are preserved. This resolves paths only; it does
        not check that the dataset exists or inspect its contents.
        """
        configured_platform = config.platform
        if (self.platform_override != "auto" and configured_platform != "auto"
                and self.platform_override != configured_platform):
            raise ValueError(
                f"Runtime platform {self.platform_override!r} does not match config "
                f"platform {configured_platform!r}"
            )
        platform = configured_platform if configured_platform != "auto" else self.platform

        if platform == "kaggle":
            default_dataset_base = Path("/kaggle/input")
            default_output_base = Path(
                os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working")
            ).expanduser()
        else:
            default_dataset_base = self.project_root
            default_output_base = self.working_dir

        dataset_base = default_dataset_base.resolve()
        output_base = default_output_base.resolve()
        dataset_root = self._resolve_config_path(config.dataset.root, dataset_base)
        cache_dir = self._resolve_config_path(config.dataset.cache_dir, output_base)
        output_dir = self._resolve_config_path(config.output_dir, output_base)
        if create_output:
            output_dir.mkdir(parents=True, exist_ok=True)

        return replace(
            config,
            dataset=replace(
                config.dataset,
                root=str(dataset_root) if dataset_root else None,
                cache_dir=str(cache_dir) if cache_dir else None,
            ),
            output_dir=str(output_dir),
        )

    @staticmethod
    def _resolve_config_path(value: str | Path | None, base: Path) -> Path | None:
        if value is None:
            return None
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = base / path
        return path.resolve()

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

