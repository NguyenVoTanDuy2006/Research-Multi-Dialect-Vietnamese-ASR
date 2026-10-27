from __future__ import annotations


def resolve_training_world_size(
    device: str,
    *,
    distributed_world_size: int,
    visible_gpu_count: int,
) -> int:
    """Return the replica count used by Trainer for optimizer-step planning."""
    distributed_world_size = max(1, int(distributed_world_size))
    if distributed_world_size > 1:
        return distributed_world_size
    if device == "cuda":
        return max(1, int(visible_gpu_count))
    return 1
