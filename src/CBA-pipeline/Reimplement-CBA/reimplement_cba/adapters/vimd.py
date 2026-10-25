from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from reimplement_cba.config import CBAConfig


@dataclass
class OpenViMD:
    source: Any
    index: Any


class ViMDAdapter:
    """Thin adapter over the workspace's mdv_asr_module ViMD interfaces."""

    def __init__(self, config: CBAConfig):
        self.config = config
        self._opened: OpenViMD | None = None

    def _dataset_values(self) -> dict[str, Any]:
        payload = json.loads(self.config.dataset_config_path.read_text(encoding="utf-8"))
        dataset_values = dict(payload["dataset"])
        override = self.config.values["dataset"].get("root_override")
        if override:
            root = Path(override).expanduser()
            if not root.is_absolute():
                root = (self.config.path.parent / root).resolve()
            dataset_values["root"] = str(root)
        return dataset_values

    def validate_config(self) -> dict[str, Any]:
        dataset_path = self.config.dataset_config_path
        dataset_values = self._dataset_values()

        from mdv_asr_module.configs.types import DatasetConfig

        dataset_config = DatasetConfig(**dataset_values)
        dataset_config.validate()
        return {
            "dataset_id": dataset_config.dataset_id,
            "source_kind": dataset_config.kind,
            "configured_root": dataset_config.root,
            "dataset_config": str(dataset_path),
        }

    def open(self) -> OpenViMD:
        if self._opened is not None:
            return self._opened
        dataset_values = self._dataset_values()

        from mdv_asr_module.configs.types import DatasetConfig
        from mdv_asr_module.data.sources import DatasetSourceFactory

        dataset_config = DatasetConfig(**dataset_values)
        dataset_config.validate()
        source = DatasetSourceFactory.create(dataset_config)
        index = source.build_index()
        self._opened = OpenViMD(source=source, index=index)
        return self._opened

    def validate_splits(self) -> dict[str, Any]:
        opened = self.open()
        available = sorted({record.split.casefold() for record in opened.index.records})
        aliases = {
            "train": {"train"},
            "validation": {"validation", "valid", "dev"},
            "test": {"test"},
        }
        missing = []
        for required in self.config.values["dataset"].get("required_splits", []):
            accepted = aliases.get(str(required).casefold(), {str(required).casefold()})
            if not any(split in accepted for split in available):
                missing.append(str(required))
        if missing:
            raise ValueError(
                f"ViMD source is missing required splits {missing}; available: {available}"
            )
        return {"record_count": len(opened.index), "available_splits": available}

    def read_audio(self, sample_record: Any):
        opened = self.open()
        from mdv_asr_module.data.audio import AudioReader

        reader = AudioReader(target_sample_rate=int(self.config.values["audio"]["sample_rate"]))
        return reader.read(sample_record, opened.source)

    def close(self) -> None:
        if self._opened is not None:
            self._opened.source.close()
            self._opened = None

    def __enter__(self) -> "ViMDAdapter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
