from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from .audio import AudioReader
from .index import DatasetIndex
from .records import SampleRecord
from .sources import DatasetSource


class DatasetInspector:
    def __init__(self, reader: AudioReader | None = None):
        self.reader = reader or AudioReader()

    def summarize(self, index: DatasetIndex) -> dict:
        splits: dict[str, list[SampleRecord]] = defaultdict(list)
        for row in index.records:
            splits[row.split].append(row)
        result = {}
        for split, rows in sorted(splits.items()):
            result[split] = {
                "clips": len(rows),
                "hours": sum(row.duration_seconds for row in rows) / 3600,
                "provinces": len({row.province_name for row in rows if row.province_name}),
                "speakers": len({row.speaker_id for row in rows if row.speaker_id}),
                "regions": dict(_counts(row.region for row in rows)),
            }
        return result

    @staticmethod
    def speaker_overlap(index: DatasetIndex) -> dict[str, int]:
        speakers = {split: {row.speaker_id for row in index.for_split(split) if row.speaker_id}
                    for split in ("train", "valid", "test")}
        return {"train_valid": len(speakers["train"] & speakers["valid"]),
                "train_test": len(speakers["train"] & speakers["test"]),
                "valid_test": len(speakers["valid"] & speakers["test"])}

    def validate_records(self, index: DatasetIndex) -> dict[str, int]:
        errors = {"empty_transcript": 0, "invalid_duration": 0, "missing_speaker": 0}
        for row in index.records:
            errors["empty_transcript"] += not bool(row.text.strip())
            errors["invalid_duration"] += row.duration_seconds <= 0
            errors["missing_speaker"] += not bool(row.speaker_id)
        if errors["empty_transcript"] or errors["invalid_duration"]:
            raise ValueError(f"Dataset metadata validation failed: {errors}")
        return errors

    def check_audio(self, index: DatasetIndex, source: DatasetSource, mode: str = "sampled") -> dict:
        if mode not in {"sampled", "all"}:
            raise ValueError("mode must be 'sampled' or 'all'")
        selected = list(index.records)
        if mode == "sampled":
            chosen = {}
            for split in sorted({row.split for row in selected}):
                rows = list(index.for_split(split))
                if rows:
                    chosen[rows[0].key] = rows[0]
                    chosen[max(rows, key=lambda row: row.duration_seconds).key] = max(
                        rows, key=lambda row: row.duration_seconds
                    )
            selected = list(chosen.values())
        checked = 0
        for row in selected:
            audio = self.reader.read(row, source)
            self.reader.validate_duration(row, audio)
            checked += 1
        return {"checked": checked, "mode": mode}

    @staticmethod
    def longest_records(index: DatasetIndex, split: str, n: int = 5) -> list[SampleRecord]:
        if n < 1:
            raise ValueError("n must be positive")
        return sorted(index.for_split(split), key=lambda row: row.duration_seconds, reverse=True)[:n]

    def export_report(self, index: DatasetIndex, path: str) -> str:
        import json
        from pathlib import Path

        payload = {"summary": self.summarize(index), "speaker_overlap": self.speaker_overlap(index),
                   "record_validation": self.validate_records(index)}
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(destination)


def _counts(values: Iterable[str | None]) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[str(value or "Unknown")] += 1
    return dict(sorted(result.items()))

