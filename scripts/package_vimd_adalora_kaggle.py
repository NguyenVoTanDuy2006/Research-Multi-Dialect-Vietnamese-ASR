#!/usr/bin/env python3
"""Build and locally validate a Kaggle-ready ViMD AdaLoRA mini dataset.

This script never invokes Kaggle CLI and never uploads data.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


DATASET_ID = "nguyendv02/ViMD_Dataset"
SPLITS = ("train", "valid", "test")
TARGET_HOURS = {"train": 25.0, "valid": 3.125, "test": 3.125}
GENDER_LABELS = {0: "female", 1: "male"}
DEFAULT_CACHE = Path.home() / ".cache" / "huggingface" / "datasets" / "nguyendv02___vi_md_dataset"


@dataclass
class AudioRecord:
    split: str
    row_index: int
    filename: str
    region: str
    province_code: int | str | None
    province_name: str
    speaker_id: str
    gender_code: int | None
    text: str
    duration_seconds: float
    sample_rate: int
    channels: int
    frames: int
    audio_format: str
    subtype: str
    source_bytes: int

    @property
    def province_key(self) -> str:
        return self.province_name or f"province_{self.province_code}"

    @property
    def gender(self) -> str:
        return GENDER_LABELS.get(self.gender_code, "unknown")


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def discover_cache(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Arrow cache path does not exist: {root}")
        return root
    candidates = sorted(
        DEFAULT_CACHE.rglob("dataset_info.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    ) if DEFAULT_CACHE.exists() else []
    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy Arrow cache ViMD. Truyền --cache-dir với thư mục chứa "
            "vi_md_dataset-{train,valid,test}-*.arrow."
        )
    return candidates[0].parent.resolve()


def load_split_datasets(cache_dir: Path) -> dict[str, Any]:
    try:
        from datasets import Audio, Dataset, concatenate_datasets
    except ImportError as exc:
        raise RuntimeError("Cần cài package 'datasets' để đọc Arrow cache.") from exc

    loaded: dict[str, Any] = {}
    for split in SPLITS:
        shards = sorted(cache_dir.glob(f"vi_md_dataset-{split}-*.arrow"))
        if not shards:
            raise FileNotFoundError(f"Không tìm thấy Arrow shards cho split={split} trong {cache_dir}")
        parts = [Dataset.from_file(str(path)) for path in shards]
        dataset = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
        if "audio" not in dataset.column_names:
            raise ValueError(f"Arrow split {split} không có cột audio.")
        loaded[split] = dataset.cast_column("audio", Audio(decode=False))
    return loaded


def get_audio_bytes(audio_value: dict[str, Any]) -> bytes:
    payload = audio_value.get("bytes")
    if payload is not None:
        return payload
    path = audio_value.get("path")
    if path and Path(path).is_file():
        return Path(path).read_bytes()
    raise ValueError("Audio row không có bytes hoặc path đọc được.")


def inspect_audio(data: bytes) -> tuple[float, int, int, int, str, str]:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("Cần cài package 'soundfile' để đọc WAV header.") from exc
    info = sf.info(io.BytesIO(data))
    if info.frames <= 0 or info.samplerate <= 0 or info.channels <= 0:
        raise ValueError("WAV có frames/sample rate/channel không hợp lệ.")
    return (
        float(info.duration),
        int(info.samplerate),
        int(info.channels),
        int(info.frames),
        str(info.format),
        str(info.subtype),
    )


def normalize_gender(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed in GENDER_LABELS else None


def scan_split(split: str, dataset: Any) -> tuple[list[AudioRecord], list[dict[str, Any]]]:
    records: list[AudioRecord] = []
    failures: list[dict[str, Any]] = []
    for index in range(len(dataset)):
        row = dataset[index]
        try:
            audio = row["audio"]
            data = get_audio_bytes(audio)
            duration, rate, channels, frames, audio_format, subtype = inspect_audio(data)
            filename = str(row.get("filename") or audio.get("path") or f"row_{index}.wav")
            filename = PurePosixPath(filename.replace("\\", "/")).name
            if not filename or filename in {".", ".."}:
                filename = f"row_{index}.wav"
            province_code = row.get("province_code")
            if province_code is not None:
                try:
                    province_code = int(province_code)
                except (TypeError, ValueError):
                    province_code = str(province_code)
            records.append(
                AudioRecord(
                    split=split,
                    row_index=index,
                    filename=filename,
                    region=str(row.get("region") or "Unknown"),
                    province_code=province_code,
                    province_name=str(row.get("province_name") or ""),
                    speaker_id=str(row.get("speakerID") or ""),
                    gender_code=normalize_gender(row.get("gender")),
                    text=str(row.get("text") or ""),
                    duration_seconds=duration,
                    sample_rate=rate,
                    channels=channels,
                    frames=frames,
                    audio_format=audio_format,
                    subtype=subtype,
                    source_bytes=len(data),
                )
            )
        except Exception as exc:
            failures.append({"split": split, "row_index": index, "error": repr(exc)})
    return records, failures


def group_by_province(records: Iterable[AudioRecord]) -> dict[str, list[AudioRecord]]:
    groups: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        groups[record.province_key].append(record)
    return dict(groups)


def split_summary(records: list[AudioRecord]) -> dict[str, Any]:
    return {
        "clips": len(records),
        "duration_seconds": round(sum(r.duration_seconds for r in records), 3),
        "duration_hours": round(sum(r.duration_seconds for r in records) / 3600, 6),
        "source_bytes": sum(r.source_bytes for r in records),
        "speakers": len({r.speaker_id for r in records if r.speaker_id}),
        "provinces": len({r.province_key for r in records}),
    }


def select_eval_split(
    records: list[AudioRecord], target_seconds: float, rng: random.Random
) -> tuple[list[AudioRecord], dict[str, dict[str, float]]]:
    by_province = group_by_province(records)
    province_seconds = {
        province: sum(r.duration_seconds for r in rows)
        for province, rows in by_province.items()
    }
    denominator = sum(province_seconds.values())
    if denominator <= 0:
        raise ValueError(f"Không có thời lượng audio hợp lệ cho {records[0].split if records else 'split'}.")
    selected: list[AudioRecord] = []
    quotas: dict[str, dict[str, float]] = {}
    for province in sorted(by_province):
        quota = target_seconds * province_seconds[province] / denominator
        pool = list(by_province[province])
        rng.shuffle(pool)
        chosen: list[AudioRecord] = []
        total = 0.0
        remaining = quota
        while pool and remaining > 0:
            # Best-fit greedy reduces overshoot caused by a long random final clip.
            # Random shuffle above still randomizes ties / equally suitable clips.
            fitting = [record for record in pool if record.duration_seconds <= remaining]
            if fitting:
                record = max(fitting, key=lambda item: item.duration_seconds)
            else:
                record = min(pool, key=lambda item: item.duration_seconds)
            chosen.append(record)
            pool.remove(record)
            total += record.duration_seconds
            remaining = quota - total
        selected.extend(chosen)
        quotas[province] = {"target_seconds": quota, "actual_seconds": total}
    return selected, quotas


def speaker_gender_groups(
    rows: list[AudioRecord],
) -> tuple[dict[str, list[AudioRecord]], dict[str, int | None], set[str]]:
    by_speaker: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in rows:
        if record.speaker_id:
            by_speaker[record.speaker_id].append(record)
    genders: dict[str, int | None] = {}
    conflicts: set[str] = set()
    for speaker, speaker_rows in by_speaker.items():
        observed = {r.gender_code for r in speaker_rows if r.gender_code is not None}
        if len(observed) == 1:
            genders[speaker] = next(iter(observed))
        else:
            genders[speaker] = None
            if len(observed) > 1:
                conflicts.add(speaker)
    return dict(by_speaker), genders, conflicts


def choose_train_province(
    rows: list[AudioRecord], target_seconds: float, rng: random.Random
) -> tuple[list[AudioRecord], dict[str, Any]]:
    by_speaker, gender_by_speaker, conflicts = speaker_gender_groups(rows)
    gendered = {
        gender: [speaker for speaker, value in gender_by_speaker.items() if value == gender]
        for gender in (0, 1)
    }
    rng.shuffle(gendered[0])
    rng.shuffle(gendered[1])

    # Start with the speaker fraction corresponding to the duration sample ratio.
    duration_pool = sum(r.duration_seconds for r in rows)
    fraction = min(1.0, target_seconds / duration_pool) if duration_pool else 0.0
    known_speakers = len(gendered[0]) + len(gendered[1])
    target_speaker_count = max(2, int(round(known_speakers * fraction))) if known_speakers >= 2 else known_speakers
    target_pairs = max(1, int(round(target_speaker_count / 2))) if known_speakers >= 2 else 0
    max_pairs = min(len(gendered[0]), len(gendered[1]))
    initial_pairs = min(target_pairs, max_pairs)

    selected_speakers: list[str] = []
    selected_rows: list[AudioRecord] = []
    selected_seconds = 0.0
    male_cursor = female_cursor = 0
    pair_count = 0

    def add_pair() -> bool:
        nonlocal male_cursor, female_cursor, selected_seconds, pair_count
        if male_cursor >= len(gendered[1]) or female_cursor >= len(gendered[0]):
            return False
        pair = [gendered[1][male_cursor], gendered[0][female_cursor]]
        male_cursor += 1
        female_cursor += 1
        rng.shuffle(pair)
        for speaker in pair:
            speaker_rows = list(by_speaker[speaker])
            rng.shuffle(speaker_rows)
            # One clip per selected speaker guarantees that speaker-count balance is real.
            representative = speaker_rows.pop()
            selected_rows.append(representative)
            selected_speakers.append(speaker)
            selected_seconds += representative.duration_seconds
        pair_count += 1
        return True

    while pair_count < initial_pairs and selected_seconds < target_seconds:
        add_pair()

    # If representative clips overshoot badly, reduce whole male/female pairs where possible.
    while selected_seconds > target_seconds and pair_count > 1:
        removed = selected_speakers[-2:]
        selected_speakers = selected_speakers[:-2]
        removed_rows = [r for r in selected_rows if r.speaker_id in removed]
        selected_rows = [r for r in selected_rows if r.speaker_id not in removed]
        selected_seconds -= sum(r.duration_seconds for r in removed_rows)
        pair_count -= 1
        male_cursor -= 1
        female_cursor -= 1

    # Fill the quota with clips from the already selected speakers, then add balanced pairs
    # if those speakers do not contain enough audio.
    def add_remaining_for_selected() -> None:
        nonlocal selected_seconds
        selected_set = set(selected_speakers)
        already = {(r.speaker_id, r.row_index) for r in selected_rows}
        extras = [
            r for r in rows
            if r.speaker_id in selected_set and (r.speaker_id, r.row_index) not in already
        ]
        rng.shuffle(extras)
        for record in extras:
            if selected_seconds >= target_seconds:
                break
            selected_rows.append(record)
            selected_seconds += record.duration_seconds

    add_remaining_for_selected()
    while selected_seconds < target_seconds and add_pair():
        add_remaining_for_selected()

    selected_rows.sort(key=lambda r: (r.filename, r.row_index))
    represented = {
        r.speaker_id: gender_by_speaker.get(r.speaker_id)
        for r in selected_rows if r.speaker_id
    }
    female_count = sum(gender == 0 for gender in represented.values())
    male_count = sum(gender == 1 for gender in represented.values())

    # If exact gender parity is impossible in a province, fill the remaining time from
    # resolved-gender speakers and make the deviation explicit in the output summary.
    if selected_seconds < target_seconds and not max_pairs:
        fallback = [r for r in rows if gender_by_speaker.get(r.speaker_id) is not None]
        rng.shuffle(fallback)
        chosen_ids = {(r.speaker_id, r.row_index) for r in selected_rows}
        for record in fallback:
            if selected_seconds >= target_seconds:
                break
            if (record.speaker_id, record.row_index) in chosen_ids:
                continue
            selected_rows.append(record)
            chosen_ids.add((record.speaker_id, record.row_index))
            selected_seconds += record.duration_seconds
            represented[record.speaker_id] = gender_by_speaker[record.speaker_id]
        female_count = sum(g == 0 for g in represented.values())
        male_count = sum(g == 1 for g in represented.values())

    # Conflicted/missing-gender speakers are not used to establish the 50:50 quota,
    # but may be included as unknown-gender examples if the resolved-gender pool is short.
    if selected_seconds < target_seconds:
        chosen_ids = {(r.speaker_id, r.row_index) for r in selected_rows}
        unknown_rows = [r for r in rows if gender_by_speaker.get(r.speaker_id) is None]
        rng.shuffle(unknown_rows)
        for record in unknown_rows:
            if selected_seconds >= target_seconds:
                break
            if (record.speaker_id, record.row_index) in chosen_ids:
                continue
            selected_rows.append(record)
            chosen_ids.add((record.speaker_id, record.row_index))
            selected_seconds += record.duration_seconds
            represented[record.speaker_id] = None

    # If the paired/unknown pools still do not cover the quota, use remaining
    # known-gender speakers. Prefer the currently underrepresented gender, so
    # the best-effort 50:50 speaker balance degrades as little as possible.
    if selected_seconds < target_seconds:
        unselected_speakers = [speaker for speaker in by_speaker if speaker not in represented]
        rng.shuffle(unselected_speakers)
        unselected_speakers.sort(
            key=lambda speaker: (
                0 if gender_by_speaker[speaker] == (0 if female_count <= male_count else 1) else 1,
                gender_by_speaker[speaker] is None,
            )
        )
        for speaker in unselected_speakers:
            speaker_rows = list(by_speaker[speaker])
            rng.shuffle(speaker_rows)
            for record in speaker_rows:
                if selected_seconds >= target_seconds:
                    break
                selected_rows.append(record)
                selected_seconds += record.duration_seconds
            represented[speaker] = gender_by_speaker[speaker]
            female_count = sum(g == 0 for g in represented.values())
            male_count = sum(g == 1 for g in represented.values())
            if selected_seconds >= target_seconds:
                break

    return selected_rows, {
        "target_seconds": target_seconds,
        "actual_seconds": selected_seconds,
        "target_speakers_approx": target_speaker_count,
        "selected_speakers": len(represented),
        "female_speakers": female_count,
        "male_speakers": male_count,
        "gender_balance_exact": female_count == male_count,
        "gender_conflict_speakers_excluded": len(conflicts),
        "available_female_speakers": len(gendered[0]),
        "available_male_speakers": len(gendered[1]),
    }


def select_train_split(
    records: list[AudioRecord], target_seconds: float, rng: random.Random
) -> tuple[list[AudioRecord], dict[str, dict[str, Any]]]:
    by_province = group_by_province(records)
    province_seconds = {
        province: sum(r.duration_seconds for r in rows)
        for province, rows in by_province.items()
    }
    denominator = sum(province_seconds.values())
    if denominator <= 0:
        raise ValueError("Train không có thời lượng audio hợp lệ.")
    selected: list[AudioRecord] = []
    stats: dict[str, dict[str, Any]] = {}
    for province in sorted(by_province):
        quota = target_seconds * province_seconds[province] / denominator
        chosen, details = choose_train_province(by_province[province], quota, rng)
        selected.extend(chosen)
        stats[province] = details
    return selected, stats


def build_selection(
    all_records: dict[str, list[AudioRecord]], seed: int
) -> tuple[dict[str, list[AudioRecord]], dict[str, Any], dict[str, list[AudioRecord]]]:
    rngs = {split: random.Random(seed + i * 1009) for i, split in enumerate(SPLITS)}
    test_speakers = {r.speaker_id for r in all_records["test"] if r.speaker_id}
    valid_candidates = [r for r in all_records["valid"] if not r.speaker_id or r.speaker_id not in test_speakers]
    valid_removed = [r for r in all_records["valid"] if r.speaker_id and r.speaker_id in test_speakers]

    selected: dict[str, list[AudioRecord]] = {}
    quotas: dict[str, Any] = {"valid_test_overlap_removed_from_valid": len(valid_removed)}
    selected["train"], quotas["train"] = select_train_split(
        all_records["train"], TARGET_HOURS["train"] * 3600, rngs["train"]
    )
    selected["valid"], quotas["valid_by_province"] = select_eval_split(
        valid_candidates, TARGET_HOURS["valid"] * 3600, rngs["valid"]
    )
    selected["test"], quotas["test_by_province"] = select_eval_split(
        all_records["test"], TARGET_HOURS["test"] * 3600, rngs["test"]
    )
    return selected, quotas, {"valid_removed_rows": valid_removed}


def record_to_manifest(record: AudioRecord, sample_id: str) -> dict[str, Any]:
    # Keep the source filename in its own metadata column while prefixing the
    # archive path with a sample ID, preventing same-name collisions.
    member = f"{record.split}/audio/{sample_id}__{record.filename}"
    return {
        "sample_id": sample_id,
        "split": record.split,
        "row_index": record.row_index,
        "archive": f"{record.split}.zip",
        "audio_path": member,
        "filename": record.filename,
        "region": record.region,
        "province_code": record.province_code,
        "province_name": record.province_name,
        "speakerID": record.speaker_id,
        "gender": record.gender,
        "gender_code": record.gender_code,
        "sample_rate": record.sample_rate,
        "channels": record.channels,
        "frames": record.frames,
        "format": record.audio_format,
        "subtype": record.subtype,
        "duration_seconds": round(record.duration_seconds, 6),
        "source_bytes": record.source_bytes,
        "text": record.text,
    }


def summarize_selected(records: list[AudioRecord]) -> dict[str, Any]:
    provinces: dict[str, dict[str, Any]] = {}
    for province, rows in sorted(group_by_province(records).items()):
        speakers: dict[str, int | None] = {}
        per_speaker: dict[str, set[int]] = defaultdict(set)
        for row in rows:
            if row.speaker_id and row.gender_code is not None:
                per_speaker[row.speaker_id].add(row.gender_code)
        for speaker, values in per_speaker.items():
            speakers[speaker] = next(iter(values)) if len(values) == 1 else None
        provinces[province] = {
            "clips": len(rows),
            "duration_hours": round(sum(r.duration_seconds for r in rows) / 3600, 6),
            "speakers": len({r.speaker_id for r in rows if r.speaker_id}),
            "female_speakers": sum(v == 0 for v in speakers.values()),
            "male_speakers": sum(v == 1 for v in speakers.values()),
            "unknown_gender_speakers": sum(v is None for v in speakers.values()),
        }
    return {**split_summary(records), "by_province": provinces}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def kaggle_metadata(kaggle_id: str) -> dict[str, Any]:
    return {
        "title": "ViMD AdaLoRA Mini Sample",
        "subtitle": "31.25-hour ViMD sample for Vietnamese ASR research",
        "description": (
            "A stratified 31.25-hour sample of ViMD for non-commercial ASR research. "
            "See README.md for source attribution, sampling details, and license information. "
            "Keep private unless redistribution rights have been confirmed."
        ),
        "id": kaggle_id,
        "licenses": [{"name": "CC-BY-NC-ND-4.0"}],
    }


def package_readme() -> str:
    return """# ViMD AdaLoRA Mini Sample\n\nThis package contains a reproducibly sampled subset of **nguyendv02/ViMD_Dataset** for local research and Kaggle notebooks.\n\n## Contents\n\n- `train.zip`, `valid.zip`, `test.zip`: selected original audio payloads, with paths rooted at `train/audio/`, `valid/audio/`, and `test/audio/`.\n- `sample_metadata.json` and `sample_metadata.csv`: transcript, source split/index, relative audio path, province/region, speaker/gender, and measured audio properties.\n- `dataset-metadata.json`: Kaggle metadata template. Replace `YOUR_KAGGLE_USERNAME` in `id` before creating a dataset.\n\n## Sampling\n\nTargets: 25 hours train, 3.125 hours valid, 3.125 hours test. Clips are not cut; actual durations are recorded in the manifest. Train sampling balances unique male/female speaker IDs per province where available. Validation and test preserve the source gender distribution. Speakers shared between valid and test are removed from valid.\n\n## Source and license\n\nSource: [ViMD Dataset](https://huggingface.co/datasets/nguyendv02/ViMD_Dataset). The source dataset identifies its license as **CC BY-NC-ND 4.0**. Preserve attribution and license notices. This package is intended to remain private; verify redistribution rights before making it public.\n"""


def write_manifest(output_dir: Path, selected: dict[str, list[AudioRecord]], seed: int,
                   failures: dict[str, list[dict[str, Any]]], quotas: dict[str, Any],
                   removed: dict[str, list[AudioRecord]], kaggle_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split in SPLITS:
        sorted_rows = sorted(selected[split], key=lambda r: (r.province_key, r.filename, r.row_index))
        for index, row in enumerate(sorted_rows):
            records.append(record_to_manifest(row, f"{split}_{index:06d}"))
        summaries[split] = summarize_selected(sorted_rows)

    manifest = {
        "schema_version": 1,
        "source_dataset": DATASET_ID,
        "source_license": "CC-BY-NC-ND-4.0",
        "seed": seed,
        "target_hours": TARGET_HOURS,
        "actual_by_split": {split: summaries[split] for split in SPLITS},
        "sampling": {
            "province_time_allocation": "proportional to eligible source duration within each split",
            "train_gender_balance": "unique speaker IDs, female/male by province where feasible",
            "validation_test_gender": "source distribution preserved",
            "valid_test_speaker_overlap_removed_from_valid": len(removed["valid_removed_rows"]),
            "valid_test_overlap_rows_removed": [
                {"filename": r.filename, "speakerID": r.speaker_id, "province_name": r.province_name}
                for r in removed["valid_removed_rows"]
            ],
            "invalid_source_rows_skipped": {split: len(failures[split]) for split in SPLITS},
        },
        "quota_details": quotas,
        "records": records,
    }
    write_json(output_dir / "sample_metadata.json", manifest)

    fields = list(records[0].keys()) if records else []
    with (output_dir / "sample_metadata.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "README.md").write_text(package_readme(), encoding="utf-8")
    write_json(output_dir / "dataset-metadata.json", kaggle_metadata(kaggle_id))
    return records, summaries


def write_archives(output_dir: Path, datasets: dict[str, Any], selected: dict[str, list[AudioRecord]]) -> None:
    for split in SPLITS:
        archive_path = output_dir / f"{split}.zip"
        dataset = datasets[split]
        rows = sorted(selected[split], key=lambda r: (r.province_key, r.filename, r.row_index))
        seen_members: set[str] = set()
        with zipfile.ZipFile(archive_path, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for index, record in enumerate(rows):
                row = dataset[record.row_index]
                data = get_audio_bytes(row["audio"])
                if len(data) != record.source_bytes:
                    raise IOError(f"Audio payload changed while packaging: {split}/{record.filename}")
                sample_id = f"{split}_{index:06d}"
                member = f"{split}/audio/{sample_id}__{record.filename}"
                if member in seen_members:
                    raise ValueError(f"Duplicate archive path: {member}")
                seen_members.add(member)
                info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)


def validate_package(output_dir: Path, selected: dict[str, list[AudioRecord]], manifest_records: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest_records:
        manifest_by_split[record["split"]].append(record)

    report: dict[str, Any] = {}
    archive_speakers: dict[str, set[str]] = {}
    for split in SPLITS:
        path = output_dir / f"{split}.zip"
        sorted_rows = sorted(selected[split], key=lambda r: (r.province_key, r.filename, r.row_index))
        expected = {
            f"{split}/audio/{split}_{index:06d}__{record.filename}"
            for index, record in enumerate(sorted_rows)
        }
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise IOError(f"ZIP CRC failed: {split}.zip::{bad_member}")
            actual = set(archive.namelist())
            if actual != expected:
                missing = sorted(expected - actual)[:3]
                extra = sorted(actual - expected)[:3]
                raise AssertionError(f"{split}.zip member mismatch; missing={missing}, extra={extra}")
            if len(actual) != len(selected[split]):
                raise AssertionError(f"Unexpected duplicate or missing files in {split}.zip")
            # Read audio headers from first/middle/last entries after the full CRC pass.
            members = sorted(actual)
            probe_indices = sorted({0, len(members) // 2, len(members) - 1}) if members else []
            for index in probe_indices:
                info = inspect_audio(archive.read(members[index]))
                if info[0] <= 0:
                    raise AssertionError(f"Invalid audio in {split}.zip::{members[index]}")
        if len(manifest_by_split[split]) != len(selected[split]):
            raise AssertionError(f"Manifest row count mismatch for {split}")
        for row in manifest_by_split[split]:
            if row["audio_path"] not in expected:
                raise AssertionError(f"Manifest path not in archive: {row['audio_path']}")
        archive_speakers[split] = {r.speaker_id for r in selected[split] if r.speaker_id}
        report[split] = {
            **split_summary(selected[split]),
            "zip_bytes": path.stat().st_size,
            "zip_crc": "ok",
            "manifest_rows": len(manifest_by_split[split]),
        }

    valid_test_overlap = archive_speakers["valid"] & archive_speakers["test"]
    train_valid_overlap = archive_speakers["train"] & archive_speakers["valid"]
    train_test_overlap = archive_speakers["train"] & archive_speakers["test"]
    if valid_test_overlap or train_valid_overlap or train_test_overlap:
        raise AssertionError(
            "Speaker overlap across packaged splits: "
            f"train/valid={sorted(train_valid_overlap)[:5]}, "
            f"train/test={sorted(train_test_overlap)[:5]}, "
            f"valid/test={sorted(valid_test_overlap)[:5]}"
        )
    return {
        "splits": report,
        "speaker_overlap": {
            "train_valid": sorted(train_valid_overlap),
            "train_test": sorted(train_test_overlap),
            "valid_test": sorted(valid_test_overlap),
        },
    }


def print_dry_run(records: dict[str, list[AudioRecord]], failures: dict[str, list[dict[str, Any]]],
                  selected: dict[str, list[AudioRecord]], quota_details: dict[str, Any],
                  free_bytes: int) -> None:
    print("\n=== ViMD package dry-run ===")
    print(f"Cache scan: {sum(len(v) for v in records.values()):,} valid rows; "
          f"skipped {sum(len(v) for v in failures.values()):,} invalid rows")
    print(f"Free disk at output: {free_bytes / 1e9:.2f} GB")
    estimated_bytes = 0
    for split in SPLITS:
        chosen = selected[split]
        summary = split_summary(chosen)
        estimated_bytes += summary["source_bytes"]
        print(
            f"{split:5s}: target={TARGET_HOURS[split]:.3f}h, "
            f"selected={summary['duration_hours']:.3f}h, clips={summary['clips']:,}, "
            f"speakers={summary['speakers']:,}, payload≈{summary['source_bytes']/1e9:.2f} GB, "
            f"provinces={summary['provinces']}"
        )
        if split == "train":
            imbalanced = [
                (
                    province,
                    detail["female_speakers"],
                    detail["male_speakers"],
                    detail["available_female_speakers"],
                    detail["available_male_speakers"],
                )
                for province, detail in quota_details["train"].items()
                if not detail["gender_balance_exact"]
            ]
            print(f"  provinces with non-50:50/limited gender pool: {len(imbalanced)}")
            if imbalanced[:8]:
                print("  examples (province, selected_female, selected_male, available_female, available_male):", imbalanced[:8])
    required = estimated_bytes + 256 * 1024 * 1024
    print(f"Estimated package payload: {estimated_bytes/1e9:.2f} GB (ZIP overhead + manifests extra)")
    print(f"Disk preflight: {'PASS' if free_bytes >= required else 'FAIL - insufficient free space'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", help="Folder containing ViMD Arrow shards; defaults to newest local HF cache.")
    parser.add_argument("--output-dir", default=str(project_root() / "outputs" / "kaggle_vimd_adalora"))
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--kaggle-id", default="YOUR_KAGGLE_USERNAME/vimd-adalora-mini")
    parser.add_argument("--dry-run", action="store_true", help="Scan and report selection without writing package files.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cache_dir = discover_cache(args.cache_dir)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir == cache_dir or cache_dir in output_dir.parents:
        raise ValueError("Output directory cannot be inside the source Arrow cache.")
    print(f"Arrow cache: {cache_dir}")
    print(f"Output package: {output_dir}")
    print("Kaggle CLI/upload: disabled; this script only builds and validates local files.")

    datasets = load_split_datasets(cache_dir)
    all_records: dict[str, list[AudioRecord]] = {}
    failures: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        print(f"Scanning {split} ({len(datasets[split]):,} rows)...", flush=True)
        all_records[split], failures[split] = scan_split(split, datasets[split])
        print(
            f"  readable={len(all_records[split]):,}, failed={len(failures[split]):,}, "
            f"duration={sum(r.duration_seconds for r in all_records[split])/3600:.3f}h",
            flush=True,
        )

    selected, quotas, removed = build_selection(all_records, args.seed)
    disk_probe = output_dir.parent
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    free_bytes = shutil.disk_usage(disk_probe).free
    print_dry_run(all_records, failures, selected, quotas, free_bytes)
    if args.dry_run:
        estimated_bytes = sum(r.source_bytes for split in SPLITS for r in selected[split])
        return 0 if free_bytes >= estimated_bytes + 256 * 1024 * 1024 else 2

    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(
                f"Output directory already contains files; refusing to overwrite: {output_dir}. "
                "Choose a new --output-dir."
            )
        raise FileExistsError(f"Output directory exists; choose a new path or remove it yourself: {output_dir}")

    estimated_bytes = sum(r.source_bytes for split in SPLITS for r in selected[split])
    required_bytes = estimated_bytes + 256 * 1024 * 1024
    if free_bytes < required_bytes:
        raise OSError(
            f"Không đủ dung lượng trống: cần khoảng {required_bytes/1e9:.2f} GB, "
            f"còn {free_bytes/1e9:.2f} GB."
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_parent = output_dir.parent
    staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=staging_parent))
    try:
        manifest_records, summaries = write_manifest(
            staging_dir, selected, args.seed, failures, quotas, removed, args.kaggle_id
        )
        print("Writing three split ZIPs (audio payloads are stored without recompression)...", flush=True)
        write_archives(staging_dir, datasets, selected)
        validation = validate_package(staging_dir, selected, manifest_records)
        write_json(staging_dir / "validation_report.json", validation)
        os.replace(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print("\n=== Package validation: PASS ===")
    for split in SPLITS:
        result = validation["splits"][split]
        print(
            f"{split}: {result['duration_hours']:.3f}h, {result['clips']:,} clips, "
            f"{result['speakers']:,} speakers, ZIP={result['zip_bytes']/1e9:.2f} GB, CRC OK"
        )
    print(f"Package: {output_dir}")
    print("Files: train.zip, valid.zip, test.zip, sample_metadata.json/csv, README.md, dataset-metadata.json, validation_report.json")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; source cache was not modified.", file=sys.stderr)
        raise SystemExit(130)
