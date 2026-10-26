from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Literal

from reimplement_cba.config import CBAConfig


@dataclass(frozen=True)
class RejectedTrainingRecord:
    sample_id: str
    split: str
    duration_seconds: float | None
    reason: str


@dataclass(frozen=True)
class TrainingRecordSelection:
    kept: tuple[object, ...]
    rejected: tuple[RejectedTrainingRecord, ...]


class AudioPreprocessor:
    """Check ViMD audio and choose records compatible with Whisper training."""

    def __init__(self, config: CBAConfig):
        audio = config.values["audio"]
        self.sample_rate = int(audio["sample_rate"])
        self.min_duration_seconds = float(audio["min_duration_seconds"])
        self.training_max_duration_seconds = float(audio["max_duration_seconds"])
        self.inference_max_duration_seconds = float(
            audio.get("long_form_max_duration_seconds", self.training_max_duration_seconds)
        )

    @staticmethod
    def rejected_record(record: object, reason: str) -> RejectedTrainingRecord:
        raw_duration = getattr(record, "duration_seconds", None)
        try:
            duration = float(raw_duration)
        except (TypeError, ValueError):
            duration = math.nan
        return RejectedTrainingRecord(
            sample_id=str(getattr(record, "sample_id", "")),
            split=str(getattr(record, "split", "")),
            duration_seconds=duration if math.isfinite(duration) else None,
            reason=reason,
        )

    def select_training_records(self, records: Iterable[object]) -> TrainingRecordSelection:
        """Use indexed durations to exclude clips unsuited to 30-second training."""
        kept: list[object] = []
        rejected: list[RejectedTrainingRecord] = []
        for record in records:
            raw_duration = getattr(record, "duration_seconds", None)
            try:
                duration = float(raw_duration)
            except (TypeError, ValueError):
                duration = math.nan
            if not math.isfinite(duration):
                reason = "invalid_duration"
            elif duration < self.min_duration_seconds:
                reason = "below_min_duration"
            elif duration > self.training_max_duration_seconds:
                reason = "above_max_duration"
            else:
                kept.append(record)
                continue
            rejected.append(self.rejected_record(record, reason))
        return TrainingRecordSelection(tuple(kept), tuple(rejected))

    def process(
        self,
        audio: object,
        *,
        purpose: Literal["training", "inference"] = "inference",
    ) -> object:
        """Validate an already decoded ViMD sample without changing its waveform."""
        if purpose not in {"training", "inference"}:
            raise ValueError(f"Unknown audio purpose: {purpose!r}")
        sample_id = getattr(audio, "sample_id", None)
        validate = getattr(audio, "validate", None)
        if not callable(validate):
            raise TypeError(f"Audio for sample {sample_id!r} must provide validate()")
        try:
            validate()
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid audio for sample {sample_id!r}: {error}") from error
        if getattr(audio, "sample_rate", None) != self.sample_rate:
            raise ValueError(
                f"Audio {sample_id!r} must be {self.sample_rate} Hz; "
                f"got {getattr(audio, 'sample_rate', None)!r} Hz"
            )
        try:
            duration = float(getattr(audio, "duration_seconds", math.nan))
        except (TypeError, ValueError):
            duration = math.nan
        maximum = (
            self.training_max_duration_seconds
            if purpose == "training"
            else self.inference_max_duration_seconds
        )
        if not math.isfinite(duration) or not self.min_duration_seconds <= duration <= maximum:
            raise ValueError(
                f"Audio {sample_id!r} duration {duration:.3f}s is outside "
                f"{purpose} limits [{self.min_duration_seconds:g}, {maximum:g}]s"
            )
        return audio

    def read(
        self,
        record: object,
        audio_reader: Callable[[object], object],
        *,
        purpose: Literal["training", "inference"],
    ) -> object:
        """Keep the record ID in errors raised while reading or validating audio."""
        sample_id = getattr(record, "sample_id", None)
        try:
            return self.process(audio_reader(record), purpose=purpose)
        except Exception as error:
            raise RuntimeError(
                f"Could not prepare {purpose} audio for sample {sample_id!r}: {error}"
            ) from error
