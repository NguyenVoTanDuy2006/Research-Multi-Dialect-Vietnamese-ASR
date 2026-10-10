from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AudioLocator:
    kind: str
    path: str
    row_index: int | None = None
    member: str | None = None

    def validate(self) -> None:
        if self.kind not in {"arrow", "parquet", "zip", "file"}:
            raise ValueError(f"Unsupported audio locator kind: {self.kind}")
        if not self.path:
            raise ValueError("Audio locator path is required")
        if self.kind in {"arrow", "parquet"} and self.row_index is None:
            raise ValueError(f"{self.kind} locator requires row_index")
        if self.kind == "zip" and not self.member:
            raise ValueError("ZIP locator requires a member path")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AudioLocator":
        locator = cls(**data)
        locator.validate()
        return locator


@dataclass(frozen=True)
class SampleRecord:
    dataset_id: str
    sample_id: str
    split: str
    filename: str
    text: str
    duration_seconds: float
    audio_locator: AudioLocator
    source_sample_id: str | None = None
    region: str | None = None
    province_name: str | None = None
    province_code: str | int | None = None
    speaker_id: str | None = None
    gender: str = "unknown"
    source_gender: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str]:
        return self.dataset_id, self.sample_id

    def validate(self) -> None:
        if not self.dataset_id or not self.sample_id or not self.split:
            raise ValueError("dataset_id, sample_id and split are required")
        if not self.filename or not self.text.strip():
            raise ValueError(f"Sample {self.sample_id} requires filename and transcript")
        if self.duration_seconds <= 0:
            raise ValueError(f"Invalid duration for {self.sample_id}")
        if self.gender not in {"female", "male", "unknown"}:
            raise ValueError(f"Invalid resolved gender: {self.gender}")
        self.audio_locator.validate()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["audio_locator"] = self.audio_locator.to_dict()
        return value

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleRecord":
        value = dict(data)
        value["audio_locator"] = AudioLocator.from_dict(value["audio_locator"])
        record = cls(**value)
        record.validate()
        return record


@dataclass
class AudioSample:
    waveform: np.ndarray
    sample_rate: int
    sample_id: str | None = None

    @property
    def duration_seconds(self) -> float:
        return len(self.waveform) / self.sample_rate if self.sample_rate else 0.0

    def validate(self) -> None:
        if self.sample_rate <= 0 or self.waveform.ndim != 1 or self.waveform.size == 0:
            raise ValueError("Audio must be a nonempty mono waveform with a valid sample rate")
        if not np.isfinite(self.waveform).all():
            raise ValueError("Audio contains NaN or Inf")
        if self.waveform.dtype != np.float32:
            raise ValueError("Audio waveform must be float32")

    def __repr__(self) -> str:
        return (f"AudioSample(sample_id={self.sample_id!r}, sample_rate={self.sample_rate}, "
                f"duration_seconds={self.duration_seconds:.2f})")

