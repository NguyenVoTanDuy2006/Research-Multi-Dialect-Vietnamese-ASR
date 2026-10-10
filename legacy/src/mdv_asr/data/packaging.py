from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

from .records import AudioLocator, SampleRecord, SelectionResult
from .sources import DatasetSource


class DatasetPackager:
    """Build a validated, local dataset package. This class never uploads it."""

    def estimate(self, selection: SelectionResult, source: DatasetSource) -> dict[str, Any]:
        total_seconds = sum(row.duration_seconds for row in selection.records)
        return {
            **selection.summary(),
            "source_fingerprint": source.fingerprint(),
            "estimated_audio_hours": total_seconds / 3600,
            "record_count": len(selection.records),
        }

    def write_audio(self, selection: SelectionResult, source: DatasetSource,
                    staging_dir: Path) -> dict[tuple[str, str], tuple[str, str]]:
        archives: dict[str, zipfile.ZipFile] = {}
        locations: dict[tuple[str, str], tuple[str, str]] = {}
        try:
            for record in selection.records:
                archive_name = f"{record.split}.zip"
                if archive_name not in archives:
                    archives[archive_name] = zipfile.ZipFile(
                        staging_dir / archive_name, "w", compression=zipfile.ZIP_STORED
                    )
                safe_name = Path(record.filename).name.replace("/", "_").replace("\\", "_")
                member = f"audio/{record.sample_id}__{safe_name}"
                payload = source.read_bytes(record.audio_locator)
                archives[archive_name].writestr(member, payload)
                locations[record.key] = archive_name, member
        finally:
            for archive in archives.values():
                archive.close()
        return locations

    def write_metadata(self, selection: SelectionResult, locations: dict,
                       staging_dir: Path, source: DatasetSource,
                       extra_metadata: dict[str, Any] | None = None) -> Path:
        rows = []
        for record in selection.records:
            archive, member = locations[record.key]
            rows.append({
                "dataset_id": record.dataset_id,
                "sample_id": record.sample_id,
                "source_sample_id": record.source_sample_id,
                "split": record.split,
                "filename": record.filename,
                "text": record.text,
                "duration_seconds": record.duration_seconds,
                "region": record.region,
                "province_name": record.province_name,
                "province_code": record.province_code,
                "speaker_id": record.speaker_id,
                "gender": record.gender,
                "source_gender": record.source_gender,
                "audio_archive": archive,
                "audio_member": member,
                "audio_path": f"{archive}::{member}",
                "metadata": record.metadata,
            })
        payload = {
            "schema_version": 1,
            "source_fingerprint": source.fingerprint(),
            "seed": selection.seed,
            "requested_seconds": selection.requested_seconds,
            "summary": selection.summary(),
            "metadata": extra_metadata or {},
            "records": rows,
        }
        manifest = staging_dir / "sample_metadata.json"
        manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        flat_rows = [{key: (json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value)
                      for key, value in row.items()} for row in rows]
        if flat_rows:
            with (staging_dir / "sample_metadata.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
                writer.writeheader()
                writer.writerows(flat_rows)
        (staging_dir / "README.md").write_text(
            "# ViMD sample package\n\n"
            "Audio is stored in one ZIP archive per split. The JSON manifest contains "
            "the transcript, sample identity, source locator, and duration.\n\n"
            "Source dataset and license: see the source dataset card and the metadata "
            "included with this package. ViMD is distributed under CC BY-NC-ND 4.0; "
            "respect the non-commercial/no-derivatives terms. This package was built locally; it has not "
            "been uploaded automatically.\n",
            encoding="utf-8",
        )
        return manifest

    def write_documentation(self, staging_dir: Path, metadata: dict[str, Any] | None = None) -> None:
        if metadata:
            (staging_dir / "package_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    def validate(self, staging_dir: str | Path) -> dict[str, Any]:
        staging_dir = Path(staging_dir)
        manifest_path = staging_dir / "sample_metadata.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        seen = set()
        checked = 0
        archives = {}
        try:
            for row in payload.get("records", []):
                key = (row["dataset_id"], row["sample_id"])
                if key in seen:
                    raise ValueError(f"Duplicate package sample: {key}")
                seen.add(key)
                archive_path = staging_dir / row["audio_archive"]
                if archive_path not in archives:
                    archive = zipfile.ZipFile(archive_path)
                    if archive.testzip() is not None:
                        archive.close()
                        raise ValueError(f"Corrupt ZIP archive: {archive_path}")
                    archives[archive_path] = archive
                info = archives[archive_path].getinfo(row["audio_member"])
                if info.file_size <= 0:
                    raise ValueError(f"Empty audio entry: {row['audio_member']}")
                if not row.get("text", "").strip():
                    raise ValueError(f"Empty transcript: {key}")
                checked += 1
        finally:
            for archive in archives.values():
                archive.close()
        return {"records_checked": checked, "archives": sorted(
            path.name for path in staging_dir.glob("*.zip"))}

    def package(self, selection: SelectionResult, source: DatasetSource,
                output_dir: str | Path, *, metadata: dict[str, Any] | None = None,
                validate: bool = True) -> Path:
        selection.validate()
        output_dir = Path(output_dir).expanduser().resolve()
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite package directory: {output_dir}")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
        try:
            locations = self.write_audio(selection, source, staging)
            self.write_metadata(selection, locations, staging, source, metadata)
            self.write_documentation(staging, metadata)
            if validate:
                report = self.validate(staging)
                (staging / "validation_report.json").write_text(
                    json.dumps(report, indent=2), encoding="utf-8"
                )
            os.replace(staging, output_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return output_dir


class ExternalInstanceBuilder:
    def extract_audio(self, video_path: str | Path, destination: str | Path) -> Path:
        import subprocess

        video_path, destination = Path(video_path), Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        command = ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
                   "-ar", "16000", "-c:a", "pcm_s16le", str(destination)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-2000:]}")
        return destination

    def build_record(self, audio_path: str | Path, transcript: str,
                     metadata: dict[str, Any]) -> SampleRecord:
        import soundfile as sf

        audio_path = Path(audio_path).resolve()
        info = sf.info(str(audio_path))
        record = SampleRecord(
            dataset_id=str(metadata.get("dataset_id", "external")),
            sample_id=str(metadata["sample_id"]), split=str(metadata.get("split", "test")),
            filename=audio_path.name, text=transcript, duration_seconds=float(info.duration),
            audio_locator=AudioLocator("file", str(audio_path)),
            region=metadata.get("region"), province_name=metadata.get("province_name"),
            province_code=metadata.get("province_code"), speaker_id=metadata.get("speaker_id"),
            gender=str(metadata.get("gender") or "unknown"), metadata=dict(metadata),
        )
        record.validate()
        return record

    @staticmethod
    def source_checksum(path: str | Path) -> str:
        import hashlib

        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def package_video(self, video_path: str | Path, transcript: str,
                      output_dir: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """Extract and package one external test instance; never overwrites output."""
        import csv
        import json
        import os
        import shutil
        import tempfile

        video_path = Path(video_path).expanduser().resolve()
        output_dir = Path(output_dir).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite package: {output_dir}")
        metadata = dict(metadata or {})
        sample_id = str(metadata.get("sample_id", "test_external_000000"))
        split = str(metadata.get("split", "test"))
        if Path(sample_id).name != sample_id or split not in {"train", "valid", "test"}:
            raise ValueError("sample_id must be a safe basename and split must be train/valid/test")
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-",
                                        dir=output_dir.parent))
        try:
            audio_dir = staging / "audio"
            audio_dir.mkdir()
            audio_path = self.extract_audio(video_path, audio_dir / f"{sample_id}.wav")
            record = self.build_record(audio_path, transcript, {
                "dataset_id": metadata.get("dataset_id", "external"),
                "sample_id": sample_id, "split": split,
                **{key: value for key, value in metadata.items()
                   if key not in {"dataset_id", "sample_id", "split"}},
            })
            archive_name = f"{split}.zip"
            member = f"{split}/audio/{sample_id}__{audio_path.name}"
            with zipfile.ZipFile(staging / archive_name, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.write(audio_path, member)
            shutil.rmtree(audio_dir)
            row = {**record.to_dict(), "audio_archive": archive_name,
                   "audio_member": member, "audio_path": f"{archive_name}::{member}",
                   "source_video": str(video_path),
                   "source_video_sha256": self.source_checksum(video_path)}
            row["audio_locator"] = None
            (staging / "sample_metadata.json").write_text(
                json.dumps({"schema_version": 1, "records": [row]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            with (staging / "sample_metadata.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list))
                                 else value for key, value in row.items()})
            (staging / "README.md").write_text(
                "# External ASR test instance\n\n"
                "This is a one-sample external evaluation package, not part of the ViMD source dataset.\n",
                encoding="utf-8",
            )
            os.replace(staging, output_dir)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return output_dir

