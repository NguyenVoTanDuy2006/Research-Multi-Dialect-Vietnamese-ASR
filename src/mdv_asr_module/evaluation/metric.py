"""Abstract interface for sentence-level evaluation metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, Generic, TypeVar

from .helper import validate_text_pairs

ScoreT = TypeVar("ScoreT")


class Metric(ABC, Generic[ScoreT]):
    """Common score/batch/aggregate/metadata interface for ASR metrics."""

    @abstractmethod
    def score(self, reference: str, prediction: str) -> ScoreT:
        """Score one transcript against its reference."""

    def score_batch(self, references: Sequence[str], predictions: Sequence[str]) -> list[ScoreT]:
        """Score aligned pairs. Specialized metrics may batch model inference."""
        pairs = validate_text_pairs(references, predictions)
        return [self.score(reference, prediction) for reference, prediction in pairs]

    @abstractmethod
    def aggregate(self, scores: Sequence[ScoreT]) -> dict[str, Any]:
        """Summarize per-sample scores; empty inputs should report ``None`` rates."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Describe the metric settings and the model revision actually used."""
