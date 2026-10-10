from __future__ import annotations

from dataclasses import asdict
from typing import Any

from mdv_asr.config.types import AdaLoraSettings


class AdaLoraManager:
    def __init__(self, settings: AdaLoraSettings):
        settings.validate()
        self.settings = settings
        self.tuner = None
        self.model = None
        self.target_names: list[str] = []
        self.successful_steps = 0

    def find_targets(self, model) -> list[str]:
        target_suffixes = tuple(self.settings.target_modules)
        names = [name for name, module in model.named_modules()
                 if module.__class__.__name__ == "Linear" and name.endswith(target_suffixes)]
        audio = [name for name in names if "audio" in name.casefold() or "encoder" in name.casefold()]
        decoder = [name for name in names if name not in audio]
        if not names or not audio or not decoder:
            raise RuntimeError("AdaLoRA target scan did not find modules in both audio encoder and decoder")
        self.target_names = names
        return names

    def attach(self, model, total_steps: int):
        from peft import AdaLoraConfig, TaskType, get_peft_model

        if total_steps < 1:
            raise ValueError("total_steps must be positive")
        self.find_targets(model)
        settings = self.settings
        config = AdaLoraConfig(
            init_r=settings.init_r, target_r=settings.target_r,
            target_modules=list(settings.target_modules), lora_alpha=settings.alpha,
            lora_dropout=settings.dropout, bias="none", task_type=TaskType.CAUSAL_LM,
            total_step=total_steps,
            tinit=max(1, round(settings.tinit_ratio * total_steps)),
            tfinal=max(1, round(settings.tfinal_ratio * total_steps)),
            deltaT=settings.delta_t,
        )
        self.model = get_peft_model(model, config)
        self.tuner = self._find_tuner(self.model)
        return self.model

    @staticmethod
    def _find_tuner(model):
        candidates = (model, getattr(model, "base_model", None),
                      getattr(getattr(model, "base_model", None), "model", None))
        for candidate in candidates:
            if candidate is not None and hasattr(candidate, "update_and_allocate"):
                return candidate
        raise AttributeError("PEFT AdaLoRA wrapper does not expose update_and_allocate")

    def trainable_summary(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("attach() must be called before trainable_summary()")
        trainable = sum(parameter.numel() for parameter in self.model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.model.parameters())
        return {"trainable_parameters": trainable, "total_parameters": total,
                "trainable_ratio": trainable / total if total else 0,
                "target_module_count": len(self.target_names),
                "audio_or_encoder_targets": sum("audio" in name.casefold() or "encoder" in name.casefold()
                                                 for name in self.target_names)}

    def update_after_step(self, successful_step: int) -> None:
        if self.tuner is None:
            raise RuntimeError("attach() must be called before updating AdaLoRA")
        if successful_step <= self.successful_steps:
            raise ValueError("AdaLoRA steps must increase monotonically")
        self.tuner.update_and_allocate(successful_step)
        self.successful_steps = successful_step

    def rank_report(self) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("attach() must be called first")
        patterns = {}
        for name, module in self.model.named_modules():
            mask = getattr(module, "rank_mask", None)
            if mask is not None:
                patterns[name] = int(mask.detach().float().sum().item())
                continue
            rank = getattr(module, "r", None)
            if isinstance(rank, dict):
                rank = rank.get("default")
            if rank is not None:
                patterns[name] = int(rank)
        return {"successful_steps": self.successful_steps,
                "target_module_count": len(self.target_names), "active_ranks": patterns}

    def state_dict(self) -> dict[str, Any]:
        state = {"settings": asdict(self.settings), "successful_steps": self.successful_steps}
        allocator = None
        for name in ("rankallocator", "rank_allocator"):
            allocator = getattr(self.tuner, name, None) if self.tuner is not None else None
            if allocator is not None:
                break
        if allocator is not None:
            fields = ("ipt", "exp_avg_ipt", "exp_avg_unc", "rank_pattern", "budget",
                      "init_bgt", "target_bgt", "tinit", "tfinal", "total_step", "deltaT")
            state["allocator"] = {name: getattr(allocator, name) for name in fields
                                   if hasattr(allocator, name)}
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("settings") != asdict(self.settings):
            raise ValueError("AdaLoRA settings differ from checkpoint")
        allocator_state = state.get("allocator", {})
        if allocator_state and self.tuner is not None:
            allocator = getattr(self.tuner, "rankallocator", None)
            if allocator is None:
                allocator = getattr(self.tuner, "rank_allocator", None)
            if allocator is None:
                raise ValueError("Checkpoint contains AdaLoRA allocator state but wrapper has no allocator")
            for name, value in allocator_state.items():
                setattr(allocator, name, value)
        self.successful_steps = int(state.get("successful_steps", 0))

    def validate_rank_consistency(self, context) -> dict[str, Any]:
        report = self.rank_report()
        reports = context.gather_objects(report)
        if any(item != reports[0] for item in reports[1:]):
            raise RuntimeError(f"AdaLoRA rank state differs across distributed processes: {reports}")
        return {"world_size": len(reports), "consistent": True, "report": report}

