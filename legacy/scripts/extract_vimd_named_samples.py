#!/usr/bin/env python3
"""Extract a fixed set of named ViMD samples using the common dataset reader."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mdv_asr.config.loader import ConfigLoader
from mdv_asr.core.artifacts import ArtifactStore
from mdv_asr.workflows.dataset import DatasetWorkflow


TARGETS = {"easy": ("30_0309", "59_0313"), "hard": ("37_0338",)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/datasets/vimd_full.json")
    parser.add_argument("--output", default="outputs/selected_vimd_instances")
    args = parser.parse_args(argv)
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing extraction: {output}")
    config = ConfigLoader().load(args.config)
    workflow = DatasetWorkflow.from_config(config.dataset)
    try:
        index = workflow.scan()
        records = [index.find_by_filename(name) for values in TARGETS.values() for name in values]
        output.mkdir(parents=True)
        exported = []
        for difficulty, names in TARGETS.items():
            for name in names:
                record = index.find_by_filename(name)
                audio_path = output / difficulty / Path(record.filename).name
                workflow.reader.export(record, workflow.source, audio_path)
                exported.append({**record.to_dict(), "difficulty": difficulty,
                                 "audio_file": str(audio_path)})
        store = ArtifactStore(output)
        store.write_json("manifest.json", {"records": exported})
        store.write_csv("manifest.csv", [
            {key: str(value) if isinstance(value, (dict, list)) else value
             for key, value in row.items()} for row in exported
        ])
        print(f"Exported {len(records)} samples to {output}")
    finally:
        workflow.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
