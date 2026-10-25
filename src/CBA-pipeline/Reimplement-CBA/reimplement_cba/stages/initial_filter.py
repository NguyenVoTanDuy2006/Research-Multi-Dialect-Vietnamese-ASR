from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class InitialFilterResult:
    kept: tuple[object, ...]
    rejected: tuple[object, ...]


class InitialTranscriptFilter:
    """Apply the paper's initial transcript-marker rule to record objects."""

    def __init__(self, rejected_markers: Iterable[str]):
        self.rejected_markers = tuple(rejected_markers)

    def apply(self, records: Iterable[object]) -> InitialFilterResult:
        kept, rejected = [], []
        for record in records:
            transcript = str(getattr(record, "text", ""))
            destination = rejected if any(marker in transcript for marker in self.rejected_markers) else kept
            destination.append(record)
        return InitialFilterResult(tuple(kept), tuple(rejected))
