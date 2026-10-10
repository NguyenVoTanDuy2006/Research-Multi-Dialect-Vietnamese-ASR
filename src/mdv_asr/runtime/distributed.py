from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class DistributedContext:
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    initialized: bool = False

    @classmethod
    def from_environment(cls) -> "DistributedContext":
        return cls(world_size=int(os.environ.get("WORLD_SIZE", "1")),
                   rank=int(os.environ.get("RANK", "0")),
                   local_rank=int(os.environ.get("LOCAL_RANK", "0")))

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def device(self):
        import torch

        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        return torch.device("cpu")

    def initialize(self, backend: str | None = None) -> "DistributedContext":
        if self.initialized or not self.is_distributed:
            return self
        import torch.distributed as dist

        if backend is None:
            backend = "nccl" if self.device.type == "cuda" else "gloo"
        if self.device.type == "cuda":
            import torch

            torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend=backend, device_id=self.device if backend == "nccl" else None)
        self.initialized = True
        return self

    def barrier(self) -> None:
        if self.is_distributed and self.initialized:
            import torch.distributed as dist

            dist.barrier()

    def gather_objects(self, value):
        if not self.is_distributed:
            return [value]
        import torch.distributed as dist

        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, value)
        return gathered

    def all_ranks_agree(self, flag: bool) -> bool:
        if not self.is_distributed:
            return bool(flag)
        import torch
        import torch.distributed as dist

        device = self.device if dist.get_backend() == "nccl" else torch.device("cpu")
        value = torch.tensor([1 if flag else 0], dtype=torch.int32, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MIN)
        return bool(value.item())

    def reduce_sum(self, value):
        if not self.is_distributed:
            return value
        import torch.distributed as dist

        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    def close(self) -> None:
        if self.initialized:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
            self.initialized = False

    def __enter__(self):
        return self.initialize()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

