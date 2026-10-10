from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StepResult:
    attempted: bool
    successful: bool
    skipped: bool
    grad_norm: float
    old_scale: float
    new_scale: float


@dataclass
class TrainProgress:
    attempted_steps: int = 0
    successful_steps: int = 0
    skipped_steps: int = 0
    micro_batches: int = 0

    def record_step(self, result: StepResult) -> None:
        if not result.attempted:
            raise ValueError("Cannot record a step that was not attempted")
        self.attempted_steps += 1
        self.successful_steps += int(result.successful)
        self.skipped_steps += int(result.skipped)
        self.validate()

    def validate(self) -> None:
        if self.successful_steps + self.skipped_steps != self.attempted_steps:
            raise ValueError("Train step counters do not reconcile")

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict) -> "TrainProgress":
        result = cls(**value)
        result.validate()
        return result


class OptimizerStepController:
    def __init__(self, optimizer, *, scaler=None, scheduler=None,
                 gradient_clip_norm: float = 1.0, context=None):
        self.optimizer = optimizer
        self.scaler = scaler
        self.scheduler = scheduler
        self.gradient_clip_norm = gradient_clip_norm
        self.context = context

    def backward(self, loss, accumulation_group_size: int) -> None:
        if accumulation_group_size < 1:
            raise ValueError("accumulation_group_size must be positive")
        scaled_loss = loss / accumulation_group_size
        if self.scaler is not None:
            self.scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

    def step(self, model, ada_manager=None) -> StepResult:
        import math
        import torch

        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        grads_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                           for parameter in parameters)
        if self.context is not None:
            grads_finite = self.context.all_ranks_agree(grads_finite)
        if parameters:
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(parameters, self.gradient_clip_norm)
            grad_norm = float(grad_norm_tensor.detach().cpu())
        else:
            grad_norm = 0.0
        if not math.isfinite(grad_norm):
            grads_finite = False
        old_scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
        if not grads_finite:
            if self.scaler is not None:
                self.scaler.update(new_scale=max(old_scale / 2, 1.0))
            self.optimizer.zero_grad(set_to_none=True)
            new_scale = float(self.scaler.get_scale()) if self.scaler is not None else 1.0
            return StepResult(True, False, True, grad_norm, old_scale, new_scale)
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            new_scale = float(self.scaler.get_scale())
            successful = new_scale >= old_scale
        else:
            self.optimizer.step()
            new_scale, successful = old_scale, True
        if successful:
            if self.scheduler is not None:
                self.scheduler.step()
            if ada_manager is not None:
                next_step = ada_manager.successful_steps + 1
                ada_manager.update_after_step(next_step)
        self.optimizer.zero_grad(set_to_none=True)
        return StepResult(True, successful, not successful, grad_norm, old_scale, new_scale)

    def zero_grad(self) -> None:
        self.optimizer.zero_grad(set_to_none=True)

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler": self.scaler.state_dict() if self.scaler is not None else None}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if self.scaler is not None and state.get("scaler") is not None:
            self.scaler.load_state_dict(state["scaler"])

