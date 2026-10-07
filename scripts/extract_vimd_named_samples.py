"""Extract a small, named set of ViMD audio records with source metadata."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import soundfile as sf
from datasets import Audio, Dataset


DEFAULT_CACHE = Path(
    "/home/fernir/.cache/huggingface/datasets/"
    "nguyendv02___vi_md_dataset/default/0.0.0/"
    "3a5b30157034e7eadd5c75fae1a820c6f9383398"
)
CACHE_ROOT = Path(os.environ.get("VIMD_CACHE_DIR", str(DEFAULT_CACHE))).expanduser()
OUTPUT_DIR = Path("outputs/selected_vimd_instances")
TARGETS = {"easy": ["30_0309", "59_0313"], "hard": ["37_0338"]}
SPLITS = ("train", "valid", "test")


def find_records() -> dict[str, dict[str, Any]]:
    wanted = {name for names in TARGETS.values() for name in names}
    hits: dict[str, list[dict[str, Any]]] = {name: [] for name in wanted}
    for split in SPLITS:
        shard_paths = sorted(CACHE_ROOT.glob(f"vi_md_dataset-{split}-*.arrow"))
        if not shard_paths:
            raise FileNotFoundError(f"Không tìm thấy shard {split} tại {CACHE_ROOT}")
        split_offset = 0
        for shard_path in shard_paths:
            part = Dataset.from_file(str(shard_path)).cast_column("audio", Audio(decode=False))
            columns = [
                col for col in (
                    "filename", "region", "province_code", "province_name", "text",
                    "speakerID", "gender", "audio",
                ) if col in part.column_names
            ]
            if "filename" not in columns or "audio" not in columns:
                raise KeyError(f"Shard thiếu filename/audio: {shard_path}")
            metadata = part.select_columns([col for col in columns if col != "audio"])
            for local_index, row in enumerate(metadata):
                filename = Path(str(row.get("filename") or "")).stem
                if filename in wanted:
                    hits[filename].append({
                        "split": split,
                        "shard_path": shard_path,
                        "part": part,
                        "shard_row_index": local_index,
                        "split_row_index": split_offset + local_index,
                        **row,
                    })
            split_offset += len(part)

    missing = [name for name, matches in hits.items() if not matches]
    duplicates = {name: matches for name, matches in hits.items() if len(matches) > 1}
    if missing or duplicates:
        details = {name: [(r["split"], str(r["shard_path"]), r["shard_row_index"])
                          for r in rows] for name, rows in duplicates.items()}
        raise RuntimeError(f"Thiếu tên={missing}; tên trùng={details}")
    return {name: rows[0] for name, rows in hits.items()}


def get_audio_bytes(audio_info: dict[str, Any]) -> bytes:
    if audio_info.get("bytes") is not None:
        return audio_info["bytes"]
    audio_path = audio_info.get("path")
    if audio_path and Path(audio_path).is_file():
        return Path(audio_path).read_bytes()
    raise FileNotFoundError(f"Không đọc được audio field: {audio_info}")


def main() -> None:
    if not CACHE_ROOT.is_dir():
        raise FileNotFoundError(f"VIMD cache không tồn tại: {CACHE_ROOT}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"Thư mục output đã có dữ liệu, không ghi đè: {OUTPUT_DIR}")

    records = find_records()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for difficulty, names in TARGETS.items():
        folder = OUTPUT_DIR / difficulty
        folder.mkdir(parents=True, exist_ok=True)
        for name in names:
            source = records[name]
            row_index = int(source["shard_row_index"])
            audio_info = source["part"][row_index]["audio"]
            audio_bytes = get_audio_bytes(audio_info)
            output_path = folder / f"{name}.wav"
            output_path.write_bytes(audio_bytes)
            audio_info = sf.info(str(output_path))
            manifest_rows.append({
                "difficulty": difficulty,
                "requested_name": name,
                "filename": source.get("filename"),
                "split": source["split"],
                "split_row_index": int(source["split_row_index"]),
                "shard_row_index": row_index,
                "source_shard": str(source["shard_path"]),
                "region": source.get("region"),
                "province_code": source.get("province_code"),
                "province_name": source.get("province_name"),
                "speakerID": source.get("speakerID"),
                "gender": source.get("gender"),
                "text": source.get("text"),
                "audio_file": str(output_path),
                "audio_bytes": len(audio_bytes),
                "duration_seconds": float(audio_info.duration),
                "sample_rate": int(audio_info.samplerate),
                "channels": int(audio_info.channels),
            })

    json_path = OUTPUT_DIR / "manifest.json"
    json_path.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = OUTPUT_DIR / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    for row in manifest_rows:
        print(f"{row['difficulty']:>4} | {row['filename']} | {row['split']} | "
              f"{row['region']} / {row['province_name']} | {row['duration_seconds']:.2f}s | "
              f"{row['audio_file']}")
        print("Transcript:", row["text"])
    print("Manifest JSON:", json_path)
    print("Manifest CSV:", csv_path)


if __name__ == "__main__":
    main()
