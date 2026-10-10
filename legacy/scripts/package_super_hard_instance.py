#!/usr/bin/env python3
"""Extract a video audio track and package it as one external ASR test instance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdv_asr.data.packaging import ExternalInstanceBuilder

TRANSCRIPT = (
    "Hết cải thìa thì gieo cải ngọt. Mà cắt cải ngọt thì gieo cải thìa "
    "với tàng ô với ngò thôi chứ hỗm giờ làm miết cho kịp. Rau xanh là có "
    "mấy món đó thôi. Còn bắp su á thì hỗm cắt trước rồi,mà  bữa nay mưa "
    "có chỗ  chòm hư, rồi có chòm lên trễ."
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", default=str(ROOT / "demo/super hard.mp4"))
    parser.add_argument("--output", default=str(ROOT / "outputs/super_hard_instance"))
    parser.add_argument("--transcript", default=TRANSCRIPT)
    parser.add_argument("--sample-id", default="test_super_hard_000000")
    args = parser.parse_args(argv)
    path = ExternalInstanceBuilder().package_video(
        args.video, args.transcript, args.output,
        metadata={"dataset_id": "super_hard_audio", "sample_id": args.sample_id,
                  "split": "test", "difficulty": "super_hard"},
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
