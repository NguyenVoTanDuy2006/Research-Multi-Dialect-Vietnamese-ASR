"""Render the ViMD and super-hard comparison videos with ManimGL."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEMO_DIR = Path(__file__).resolve().parent
SCENE_FILE = DEMO_DIR / "asr_performance.py"
VIDEO_DIR = DEMO_DIR / "videos"
SCENES = {
    "DemoHaNoi": "30_0309",
    "DemoHoChiMinh": "59_0313",
    "DemoNgheAn": "37_0338",
    "DemoSuperHard": "super_hard",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only", choices=("all", *SCENES.values()), default="all",
        help="Render one named video instead of all four.",
    )
    args = parser.parse_args()
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = DEMO_DIR / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir = cache_dir / "matplotlib"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYGLET_HEADLESS"] = "1"
    env["XDG_CACHE_HOME"] = str(cache_dir)
    env["MPLCONFIGDIR"] = str(mpl_dir)

    for scene, output_name in SCENES.items():
        if args.only != "all" and output_name != args.only:
            continue
        command = [
            sys.executable, "-m", "manimlib", str(SCENE_FILE), scene,
            "-w", "-r", "1920x1080",
            "--video_dir", str(VIDEO_DIR), "--file_name", output_name,
        ]
        print("Rendering:", output_name, flush=True)
        subprocess.run(command, check=True, cwd=DEMO_DIR.parent, env=env)


if __name__ == "__main__":
    main()
