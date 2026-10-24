"""Durable JSONL cache for completed semantic scoring batches."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class SemanticScoreStore:
    """Persist per-pair semantic scores so interrupted runs can resume."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._validated_for_append = False

    @staticmethod
    def _validate_record(record: Any, label: str) -> dict[str, Any]:
        if not isinstance(record, dict) or not isinstance(record.get("metric_fingerprint"), str):
            raise ValueError(f"Invalid semantic score cache record on {label}")
        scores = record.get("scores")
        if not isinstance(scores, dict) or any(
            not isinstance(pair_hash, str) or not isinstance(score, dict)
            for pair_hash, score in scores.items()
        ):
            raise ValueError(f"Invalid semantic scores on {label}")
        return record

    def _read_batches(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []

        raw = self.path.read_bytes()
        lines = raw.splitlines(keepends=True)
        nonempty = [index for index, line in enumerate(lines) if line.strip()]
        last_nonempty = nonempty[-1] if nonempty else None
        batches: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            payload = line.rstrip(b"\r\n")
            if not payload.strip():
                continue
            try:
                record = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                if line_number - 1 == last_nonempty:
                    return batches
                raise ValueError(
                    f"Malformed semantic score cache JSON on line {line_number}"
                ) from exc

            batches.append(self._validate_record(record, f"line {line_number}"))
        return batches

    def _repair_trailing_record(self) -> None:
        """Inspect only the final nonblank JSONL row and truncate it if partial."""
        if not self.path.is_file():
            return
        with self.path.open("r+b") as stream:
            stream.seek(0, os.SEEK_END)
            cursor = stream.tell()
            if cursor == 0:
                return

            reverse_line = bytearray()
            tail_record = None
            while cursor:
                block_start = max(0, cursor - 8192)
                stream.seek(block_start)
                block = stream.read(cursor - block_start)
                for index in range(len(block) - 1, -1, -1):
                    byte = block[index]
                    if byte == ord("\n"):
                        if reverse_line.strip():
                            tail_record = (
                                block_start + index + 1,
                                bytes(reversed(reverse_line)).rstrip(b"\r"),
                            )
                            break
                        reverse_line.clear()
                    else:
                        reverse_line.append(byte)
                if tail_record is not None:
                    break
                cursor = block_start

            if tail_record is None and reverse_line.strip():
                tail_record = (0, bytes(reversed(reverse_line)).rstrip(b"\r"))
            if tail_record is None:
                return

            record_start, payload = tail_record
            try:
                record = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                stream.truncate(record_start)
                stream.flush()
                os.fsync(stream.fileno())
                return
            self._validate_record(record, "final row")

    def load(self, metric_fingerprint: str) -> dict[str, dict[str, object]]:
        if not metric_fingerprint.strip():
            raise ValueError("metric_fingerprint cannot be empty")
        scores: dict[str, dict[str, object]] = {}
        for batch in self._read_batches():
            if batch["metric_fingerprint"] == metric_fingerprint:
                scores.update(batch["scores"])
        self._validated_for_append = True
        return scores

    def append_batch(
        self,
        metric_fingerprint: str,
        scores: Mapping[str, Mapping[str, object]],
    ) -> None:
        if not metric_fingerprint.strip():
            raise ValueError("metric_fingerprint cannot be empty")
        if not isinstance(scores, Mapping):
            raise TypeError("scores must be a mapping of pair hashes to score mappings")
        normalized: dict[str, dict[str, object]] = {}
        for pair_hash, score in scores.items():
            if not isinstance(pair_hash, str) or not pair_hash:
                raise ValueError("pair hashes must be non-empty strings")
            if not isinstance(score, Mapping):
                raise TypeError(f"Score for {pair_hash!r} must be a mapping")
            normalized[pair_hash] = dict(score)
        if not normalized:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self._validated_for_append:
            self._read_batches()
            self._validated_for_append = True
        self._repair_trailing_record()
        record = {
            "metric_fingerprint": metric_fingerprint,
            "scores": {key: normalized[key] for key in sorted(normalized)},
        }
        line = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n"
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            if self.path.stat().st_size:
                with self.path.open("rb") as reader:
                    reader.seek(-1, os.SEEK_END)
                    ends_with_newline = reader.read(1) == b"\n"
                if not ends_with_newline:
                    stream.write("\n")
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
