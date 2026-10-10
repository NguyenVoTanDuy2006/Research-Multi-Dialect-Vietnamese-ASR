"""Architecture-independent selection of ordinary, unquantized Linear layers."""
from __future__ import annotations
import re


class ModuleTargetSelector:
    def __init__(self, targets: tuple[str, ...] | list[str] | str):
        if not targets:
            raise ValueError("At least one target suffix or a regular expression is required")
        self.targets = targets

    def select(self, model, *, rank: int) -> list[str]:
        import torch
        def matches(name):
            if isinstance(self.targets, str):
                return re.fullmatch(self.targets, name) is not None
            return any(name == target or name.endswith("." + target) for target in self.targets)
        names = []
        for name, module in model.named_modules():
            if not name or not matches(name):
                continue
            if type(module) is not torch.nn.Linear:
                raise TypeError(f"{name}: only unquantized torch.nn.Linear is supported, got {type(module).__name__}")
            if rank > min(module.weight.shape):
                raise ValueError(f"{name}: rank {rank} exceeds weight dimensions {tuple(module.weight.shape)}")
            names.append(name)
        if not names:
            raise ValueError(f"No Linear modules matched {self.targets!r}")
        return names
