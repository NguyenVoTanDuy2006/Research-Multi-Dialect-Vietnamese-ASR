from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class InitialFilterResult:
    kept: tuple[object, ...]
    rejected: tuple[object, ...]


class InitialTranscriptFilter:
    """Apply initial transcript and reviewed sample-ID exclusions."""

    def __init__(
        self,
        rejected_markers: Iterable[str] = (),
        rejected_sample_ids: Iterable[str] = (),
    ):
        self.rejected_markers = tuple(rejected_markers)
        self.rejected_sample_ids = frozenset(
            str(sample_id).strip()
            for sample_id in rejected_sample_ids
            if str(sample_id).strip()
        )

    def apply(self, records: Iterable[object]) -> InitialFilterResult:
        kept, rejected = [], []
        for record in records:
            destination = rejected if self.rejection_reason(record) else kept
            destination.append(record)
        return InitialFilterResult(tuple(kept), tuple(rejected))

    def rejection_reason(self, record: object) -> str | None:
        sample_id = str(getattr(record, "sample_id", "")).strip()
        if sample_id in self.rejected_sample_ids:
            return "approved_english_sample_id"
        transcript = str(getattr(record, "text", ""))
        if any(marker in transcript for marker in self.rejected_markers):
            return "transcript_marker"
        return None
