from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from mdv_asr_module.data.records import AudioSample


@dataclass(frozen=True)
class SampleScore:
    sample_id: str
    previous_score: float
    current_score: float


@dataclass(frozen=True)
class CurriculumDecision:
    sample_id: str
    domain: str
    category: str
    keep: bool
    previous_score: float
    current_score: float
    reason: str


@dataclass(frozen=True)
class CheckpointMetrics:
    checkpoint: str
    wer: float
    semscore: float

    def validate(self) -> None:
        if not self.checkpoint:
            raise ValueError("checkpoint path is required")
        if not math.isfinite(self.wer) or self.wer < 0:
            raise ValueError("WER must be a finite non-negative number")
        if not math.isfinite(self.semscore) or not 0 <= self.semscore <= 100:
            raise ValueError("SemScore must be between 0 and 100")


class Transcriber(Protocol):
    def load(self) -> None:
        """Load the configured model and optional adapter."""

    def transcribe(self, audio: AudioSample) -> str:
        """Return one transcript for an audio sample."""

    def metadata(self) -> dict[str, Any]:
        """Describe the loaded checkpoint and runtime settings."""

    def close(self) -> None:
        """Release model resources."""


class PairScorer(Protocol):
    def score(self, reference: str, hypothesis: str) -> float:
        """Return a per-sample semantic score in the range 0..100."""
