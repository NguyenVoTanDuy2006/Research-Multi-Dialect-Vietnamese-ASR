"""Small import helper shared by percent-notebooks; no data/model work at import."""

from __future__ import annotations

import sys
from pathlib import Path


def initialize_project() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents,
                  Path(__file__).resolve().parents[1],
                  *Path(__file__).resolve().parents[1].parents]
    for root in candidates:
        if (root / "src" / "mdv_asr").is_dir():
            if str(root / "src") not in sys.path:
                sys.path.insert(0, str(root / "src"))
            return root
        if (root / "configs").is_dir():
            return root
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for configs in kaggle_input.rglob("configs"):
            if configs.is_dir() and (configs / "datasets").is_dir():
                return configs.parent
    try:
        import mdv_asr  # noqa: F401

        return Path.cwd()
    except ImportError as exc:
        raise RuntimeError("Cannot locate project package/configs; install the wheel and attach its config bundle") from exc
