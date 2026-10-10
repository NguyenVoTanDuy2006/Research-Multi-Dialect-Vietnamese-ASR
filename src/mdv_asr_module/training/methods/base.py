from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import asdict
from pathlib import Path
from .targets import ModuleTargetSelector


class FineTuningMethod(ABC):
    """Model-independent PEFT lifecycle; trainer owns AMP/optimizer/scheduler.

    Call after_optimizer_step only after a successful update, before zero_grad.
    """
    name: str

    def __init__(self, settings):
        settings.validate()
        self.settings = settings
        self.model = None
        self.target_names = []
        self.successful_steps = 0

    @abstractmethod
    def attach(self, model, total_steps: int | None = None, *, initialization_state=None):
        """Wrap a fresh pretrained base, optionally reconstructing saved initialization."""

    def _select(self, model, rank):
        if self.model is not None or getattr(model, "peft_config", None):
            raise ValueError("Attach requires a fresh base model without an adapter")
        self.target_names = ModuleTargetSelector(self.settings.target_modules).select(model, rank=rank)
        return self.target_names

    def after_optimizer_step(self, successful_step: int) -> None:
        if self.model is None:
            raise RuntimeError("attach() must be called first")
        if successful_step != self.successful_steps + 1:
            raise ValueError("Method step must advance by exactly one successful update")
        self.successful_steps = successful_step

    def update_after_step(self, successful_step):
        self.after_optimizer_step(successful_step)

    def summary(self):
        if self.model is None:
            raise RuntimeError("attach() must be called first")
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {"method": self.name, "trainable_parameters": trainable, "total_parameters": total,
                "trainable_ratio": trainable / total if total else 0,
                "target_module_count": len(self.target_names),
                "audio_or_encoder_targets": sum("audio" in n.lower() or "encoder" in n.lower()
                                                for n in self.target_names)}

    def trainable_summary(self):
        return self.summary()

    def state_dict(self):
        return {"name": self.name, "settings": asdict(self.settings), "successful_steps": self.successful_steps}

    def load_state_dict(self, state):
        if state.get("name", self.name) != self.name or state.get("settings") != asdict(self.settings):
            raise ValueError("Fine-tuning method/settings differ from checkpoint")
        self.successful_steps = int(state["successful_steps"])

    def save_initialization(self, destination: str | Path):
        return None

    def save_adapter(self, destination: str | Path):
        if self.model is None:
            raise RuntimeError("attach() must be called first")
        self.model.save_pretrained(destination, safe_serialization=True, save_embedding_layers=False)
        return Path(destination)

    def rank_report(self):
        ranks = {}
        if self.model is None:
            raise RuntimeError("attach() must be called first")
        for name, module in self.model.named_modules():
            rank = getattr(module, "r", None)
            if isinstance(rank, dict) and "default" in rank:
                ranks[name] = int(rank["default"])
        return {"method": self.name, "successful_steps": self.successful_steps, "active_ranks": ranks}

    def validate_rank_consistency(self, context):
        report = self.rank_report()
        reports = context.gather_objects(report)
        if any(item != reports[0] for item in reports[1:]):
            raise RuntimeError("Adapter rank/step state differs across distributed processes")
        return {"world_size": len(reports), "consistent": True, "report": report}
