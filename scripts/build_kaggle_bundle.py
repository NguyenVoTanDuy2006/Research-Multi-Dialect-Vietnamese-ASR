#!/usr/bin/env python3
"""Build a Kaggle-ready project bundle without dataset/model/token payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(output: Path) -> Path:
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing bundle: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    try:
        wheel_dir = staging / "wheels"
        wheel_dir.mkdir()
        subprocess.run([sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps",
                        "--no-build-isolation", "--wheel-dir", str(wheel_dir)], check=True)
        config_dir = staging / "configs"
        shutil.copytree(ROOT / "configs", config_dir)
        # Prevent accidental credentials from entering an upload bundle.
        for path in config_dir.rglob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if _contains_secret(payload):
                raise ValueError(f"Refusing to bundle a credential-like value from {path}")
        (staging / "requirements-kaggle.txt").write_text(
            "# The project wheel is installed with --no-deps to preserve Kaggle's Torch/NumPy.\n"
            "# Install only optional packages missing from the active Kaggle image.\n",
            encoding="utf-8",
        )
        (staging / "README.md").write_text(
            "# ViMD ASR Kaggle code bundle\n\n"
            "Contains the project wheel, experiment configs and checksums only.\n"
            "It intentionally excludes datasets, model checkpoints and credentials.\n\n"
            "Install the wheel with `pip install --no-deps wheels/*.whl`.\n",
            encoding="utf-8",
        )
        files = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {str(path.relative_to(staging)): sha256(path) for path in files}
        (staging / "SHA256SUMS.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output


def _contains_secret(value) -> bool:
    if isinstance(value, dict):
        return any("token" in str(key).casefold() and val not in (None, "")
                   or _contains_secret(val) for key, val in value.items())
    if isinstance(value, list):
        return any(_contains_secret(item) for item in value)
    return isinstance(value, str) and value.startswith("hf_")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/kaggle_code_bundle")
    args = parser.parse_args(argv)
    print(build(Path(args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
