from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


def prediction_row(
    record: Any,
    hypothesis: str,
    *,
    raw_hypothesis: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "sample_id": str(record.sample_id),
        "split": str(record.split),
        "reference": str(record.text),
        "hypothesis": hypothesis,
    }
    if raw_hypothesis is not None:
        row["hypothesis_raw"] = raw_hypothesis
    for key in ("region", "province_name", "province_code"):
        value = getattr(record, key, None)
        if value is not None:
            row[key] = value
    return row


def write_predictions(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Atomically write one UTF-8 JSON object per line."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for row in rows:
                output.write(json.dumps(row, ensure_ascii=False, default=str))
                output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return destination


def adapter_prediction_path(output_dir: Path, adapter_path: str | Path) -> Path:
    """Build a readable, stable filename that distinguishes curriculum rounds."""
    normalized = str(adapter_path).replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    label = "-".join(parts[-2:]) if parts else "adapter"
    safe = "".join(character if character.isalnum() or character in "._-" else "-"
                   for character in label)
    return output_dir / "predictions" / f"{safe or 'adapter'}.jsonl"
