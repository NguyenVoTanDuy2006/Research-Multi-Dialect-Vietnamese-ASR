from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    dataset_id: str
    sample_id: str
    model_key: str
    reference: str
    prediction: str
    split: str
    filename: str
    region: str | None = None
    province_name: str | None = None
    speaker_id: str | None = None
    inference_seconds: float | None = None
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.dataset_id, self.sample_id, self.model_key

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionRecord":
        return cls(**data)


@dataclass(frozen=True)
class ErrorCounts:
    word_errors: int = 0
    reference_words: int = 0
    char_errors: int = 0
    reference_chars: int = 0

    @property
    def wer(self) -> float:
        return self.word_errors / self.reference_words if self.reference_words else 0.0

    @property
    def cer(self) -> float:
        return self.char_errors / self.reference_chars if self.reference_chars else 0.0

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            self.word_errors + other.word_errors,
            self.reference_words + other.reference_words,
            self.char_errors + other.char_errors,
            self.reference_chars + other.reference_chars,
        )


@dataclass
class EvaluationResult:
    predictions: list[PredictionRecord]
    counts: ErrorCounts
    expected_keys: set[tuple[str, str, str]] = field(default_factory=set)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float:
        if not self.expected_keys:
            return 1.0
        return len({row.key for row in self.predictions if not row.error}) / len(self.expected_keys)

