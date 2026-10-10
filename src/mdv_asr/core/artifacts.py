from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping


def hash_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_payload(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class ArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def prepare_run(self, run_id: str, *, allow_existing: bool = True) -> Path:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("run_id must be a single safe path component")
        path = self.root / run_id
        if path.exists() and not allow_existing and any(path.iterdir()):
            raise FileExistsError(f"Run output is not empty: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _atomic_write(path: Path, writer) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as stream:
                writer(stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        return self._atomic_write(self.root / name, lambda stream: stream.write(data))

    def read_json(self, name: str) -> Any:
        return json.loads((self.root / name).read_text(encoding="utf-8"))

    def write_csv(self, name: str, rows: Iterable[Mapping[str, Any]],
                  *, fieldnames: Iterable[str] | None = None) -> Path:
        values = list(rows)
        columns = list(fieldnames) if fieldnames is not None else (list(values[0]) if values else [])
        if not columns:
            raise ValueError("Cannot write a CSV without rows/field names")
        import io

        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=columns)
        writer.writeheader()
        writer.writerows(values)
        data = buffer.getvalue().encode("utf-8-sig")
        return self._atomic_write(self.root / name, lambda stream: stream.write(data))

    def write_run_manifest(self, config: Any, provenance: Mapping[str, Any]) -> Path:
        config_value = config.to_dict() if hasattr(config, "to_dict") else config
        return self.write_json("run_manifest.json", {
            "config": config_value,
            "config_fingerprint": hash_payload(config_value),
            "provenance": dict(provenance),
        })

    def path(self, name: str) -> Path:
        return self.root / name

