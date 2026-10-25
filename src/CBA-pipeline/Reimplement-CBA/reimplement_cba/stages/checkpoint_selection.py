from __future__ import annotations

from collections.abc import Iterable

from reimplement_cba.contracts import CheckpointMetrics


def select_best_checkpoint(metrics: Iterable[CheckpointMetrics]) -> CheckpointMetrics:
    """Rank validation checkpoints by SemScore descending, then WER ascending."""
    candidates = list(metrics)
    if not candidates:
        raise ValueError("Cannot select a checkpoint from an empty metric list")
    for candidate in candidates:
        candidate.validate()
    return sorted(candidates, key=lambda item: (-item.semscore, item.wer, item.checkpoint))[0]
