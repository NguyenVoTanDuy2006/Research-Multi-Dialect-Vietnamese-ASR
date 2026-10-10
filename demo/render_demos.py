#!/usr/bin/env python3
"""Render saved ASR predictions with the shared DemoRenderer."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from demo.renderer import DemoRenderer
from demo.settings import DemoConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "demo/configs/asr_comparison_legacy_predictions.json"))
    parser.add_argument("--predictions", default=str(ROOT / "demo/predictions.csv"))
    parser.add_argument("--audio-root", default=str(ROOT))
    parser.add_argument("--output")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args(argv)
    config = DemoConfig.load(args.config)
    renderer = DemoRenderer(config, args.predictions, args.audio_root,
                            args.output or config.output_dir)
    videos = [renderer.render_preview(args.case_id[0] if args.case_id else None)] if args.preview \
        else renderer.render_all(args.case_id)
    print("Rendered:", *videos, sep="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
