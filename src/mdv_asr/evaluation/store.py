from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from mdv_asr.core.artifacts import hash_payload
from mdv_asr.core.errors import ResumeMismatchError

from .records import PredictionRecord


class PredictionStore:
    def __init__(self, path: str | Path, *, flush_every: int = 10):
        if flush_every < 1:
            raise ValueError("flush_every must be positive")
        self.path = Path(path)
        self.flush_every = flush_every
        self._fingerprint = None
        self._rows: dict[tuple[str, str, str], PredictionRecord] = {}
        self._pending = 0

    def open(self, run_config: dict, *, resume: bool = True) -> "PredictionStore":
        self._fingerprint = hash_payload(run_config)
        metadata_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        if self.path.exists() and resume:
            if not metadata_path.is_file():
                raise ResumeMismatchError(f"Prediction file has no fingerprint metadata: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("fingerprint") != self._fingerprint:
                raise ResumeMismatchError("Prediction cache belongs to a different dataset/model/decode config")
            self._read_rows()
        elif self.path.exists() and not resume:
            raise FileExistsError(f"Refusing to overwrite existing predictions: {self.path}")
        else:
            self._write_metadata(metadata_path)
        return self

    def _write_metadata(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fingerprint": self._fingerprint}, indent=2), encoding="utf-8")

    def _read_rows(self) -> None:
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                row = PredictionRecord.from_dict(json.loads(line))
            except (ValueError, TypeError, KeyError):
                continue
            self._rows[row.key] = row

    def completed_keys(self, model_key: str | None = None) -> set[tuple[str, str, str]]:
        return {key for key, row in self._rows.items()
                if not row.error and (model_key is None or row.model_key == model_key)}

    def pending(self, records, model_key: str) -> list:
        completed = self.completed_keys(model_key)
        return [record for record in records if (record.dataset_id, record.sample_id, model_key) not in completed]

    def upsert(self, prediction: PredictionRecord) -> None:
        self._rows[prediction.key] = prediction
        self._pending += 1
        if self._pending >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        if self._fingerprint is None:
            raise RuntimeError("PredictionStore.open() must be called first")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                for key in sorted(self._rows):
                    stream.write(json.dumps(self._rows[key].to_dict(), ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        self._write_metadata(self.path.with_suffix(self.path.suffix + ".meta.json"))
        self._pending = 0

    def load_all(self) -> list[PredictionRecord]:
        if not self._rows and self.path.is_file():
            self._read_rows()
        return [self._rows[key] for key in sorted(self._rows)]

    def validate_expected_ids(self, expected_keys, model_key: str) -> None:
        expected = {(dataset_id, sample_id, model_key) for dataset_id, sample_id in expected_keys}
        actual = self.completed_keys(model_key)
        if actual != expected:
            raise ResumeMismatchError(
                f"Prediction coverage mismatch: missing={len(expected-actual)}, unexpected={len(actual-expected)}"
            )

