#!/usr/bin/env python3
"""Create and validate a local, upload-ready ViMD AdaLoRA sample package."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdv_asr.cli import main as mdv_main


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/datasets/vimd_full.json")
    parser.add_argument("--sampling-config", default="configs/sampling/vimd_adalora_31_25h.json")
    parser.add_argument("--output", default="outputs/kaggle_vimd_adalora")
    args = parser.parse_args(argv)
    return mdv_main(["data", "package", "--config", args.dataset_config,
                     "--sampling-config", args.sampling_config, "--output", args.output])


if __name__ == "__main__":
    raise SystemExit(main())
