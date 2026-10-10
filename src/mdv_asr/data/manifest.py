from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mdv_asr.core.errors import DatasetIntegrityError

from .index import DatasetIndex
from .records import AudioLocator, SampleRecord
from .sources import DatasetSource


class ManifestRepository:
    schema_version = 1

    def save(self, index: DatasetIndex, path: str | Path,
             metadata: dict[str, Any] | None = None) -> Path:
        import os
        import tempfile

        path = Path(path)
        payload = {
            "schema_version": self.schema_version,
            "source_fingerprint": index.source_fingerprint,
            "metadata": metadata or {},
            "records": [record.to_dict() for record in index.records],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    def load(self, path: str | Path) -> DatasetIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return DatasetIndex([self._legacy_record(row, i) for i, row in enumerate(payload)])
        version = int(payload.get("schema_version", 0))
        if version != self.schema_version:
            raise DatasetIntegrityError(f"Unsupported manifest schema_version={version}")
        records = [SampleRecord.from_dict(row) for row in payload.get("records", [])]
        return DatasetIndex(records, source_fingerprint=payload.get("source_fingerprint"))

    def import_legacy(self, path: str | Path, source_config=None) -> DatasetIndex:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", payload.get("samples", []))
        return DatasetIndex([self._legacy_record(row, i, source_config) for i, row in enumerate(rows)])

    @staticmethod
    def _legacy_record(row: dict[str, Any], index: int, source_config=None) -> SampleRecord:
        split = str(row.get("split") or "test")
        audio_value = row.get("audio_locator")
        if isinstance(audio_value, dict):
            locator = AudioLocator.from_dict(audio_value)
        elif row.get("source_file") and row.get("source_row_index") is not None:
            source_path = str(row["source_file"])
            kind = "parquet" if source_path.casefold().endswith(".parquet") else "arrow"
            locator = AudioLocator(kind, source_path, row_index=int(row["source_row_index"]))
        else:
            path = row.get("audio_file") or row.get("audio_path") or row.get("filename")
            if not path:
                raise DatasetIntegrityError(f"Legacy record {index} has no audio path")
            locator = AudioLocator("file", str(path))
        dataset_id = (source_config.dataset_id if source_config else row.get("dataset_id", "legacy"))
        duration = row.get("duration_seconds")
        if duration is None:
            raise DatasetIntegrityError(f"Legacy record {index} has no duration_seconds")
        raw_gender = row.get("resolved_speaker_gender", row.get("gender"))
        gender_map = {"0": "female", "female": "female", "f": "female", "nữ": "female", "nu": "female",
                      "1": "male", "male": "male", "m": "male", "nam": "male"}
        gender = gender_map.get(str(raw_gender).strip().casefold(), "unknown")
        return SampleRecord(
            dataset_id=str(dataset_id), sample_id=str(row.get("sample_id") or f"{split}_{index:07d}"),
            split=split, filename=str(row.get("filename") or Path(locator.path).name),
            text=str(row.get("text") or ""), duration_seconds=float(duration),
            audio_locator=locator, source_sample_id=row.get("source_sample_id"),
            region=row.get("region"), province_name=row.get("province_name"),
            province_code=row.get("province_code"),
            speaker_id=row.get("speakerID", row.get("speaker_id")),
            gender=gender,
            source_gender=str(raw_gender) if raw_gender is not None else None,
            metadata={key: value for key, value in row.items()
                      if key not in {"text", "filename", "sample_id", "audio_path", "audio_file"}},
        )

    def validate_source(self, index: DatasetIndex, source: DatasetSource) -> None:
        if index.source_fingerprint and index.source_fingerprint != source.fingerprint():
            raise DatasetIntegrityError("Manifest fingerprint does not match dataset source")

    def is_reusable(self, path: str | Path, source_fingerprint: str) -> bool:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            return (payload.get("schema_version") == self.schema_version
                    and payload.get("source_fingerprint") == source_fingerprint)
        except (OSError, ValueError, TypeError):
            return False

