from __future__ import annotations
from copy import deepcopy
from pathlib import Path

from mdv_asr_module.config.types import AdaLoraSettings
from .base import FineTuningMethod


class AdaLoRAMethod(FineTuningMethod):
    name = "adalora"

    def __init__(self, settings: AdaLoraSettings | None = None):
        super().__init__(settings or AdaLoraSettings())
        self.tuner = None

    def find_targets(self, model):
        return self._select(model, self.settings.init_r)

    def attach(self, model, total_steps=None, *, initialization_state=None):
        from peft import AdaLoraConfig
        from .peft_wrappers import AdaLoRAPeftModel
        if total_steps is None or total_steps < 3:
            raise ValueError("AdaLoRA requires at least 3 planned optimizer steps")
        self.find_targets(model)
        s = self.settings
        tinit, tfinal = max(1, round(s.tinit_ratio * total_steps)), max(1, round(s.tfinal_ratio * total_steps))
        if tinit + tfinal >= total_steps:
            raise ValueError("AdaLoRA warmup/final periods leave no rank-allocation steps")
        config = AdaLoraConfig(
            init_r=s.init_r, target_r=s.target_r, target_modules=self.target_names,
            lora_alpha=s.alpha, lora_dropout=s.dropout, bias="none", task_type=None,
            total_step=total_steps, tinit=tinit, tfinal=tfinal, deltaT=s.delta_t,
        )
        config.base_model_name_or_path = getattr(model, "name_or_path", None)
        self.model = AdaLoRAPeftModel(model, config)
        self.tuner = self.model.base_model
        return self.model

    def after_optimizer_step(self, successful_step):
        if self.tuner is None or successful_step != self.successful_steps + 1:
            raise ValueError("Attach first and advance exactly one successful step")
        self.tuner.update_and_allocate(successful_step)
        super().after_optimizer_step(successful_step)

    def state_dict(self):
        state = super().state_dict()
        if self.tuner is not None:
            allocator = self.tuner.rankallocator
            fields = ("ipt", "exp_avg_ipt", "exp_avg_unc", "init_bgt", "target_bgt")
            state["allocator"] = {k: deepcopy(getattr(allocator, k)) for k in fields if hasattr(allocator, k)}
            state["rank_pattern"] = deepcopy(self.model.peft_config["default"].rank_pattern)
            state["total_steps"] = self.model.peft_config["default"].total_step
        return state

    def load_state_dict(self, state):
        import torch
        super().load_state_dict(state)
        if self.tuner is None:
            raise RuntimeError("Attach before restoring allocator")
        config = self.model.peft_config["default"]
        if state.get("total_steps", config.total_step) != config.total_step:
            raise ValueError("AdaLoRA total step schedule differs from checkpoint")
        parameters = dict(self.tuner.model.named_parameters())
        for key, value in state.get("allocator", {}).items():
            if isinstance(value, dict):
                value = {n: v.to(parameters[n].device) if torch.is_tensor(v) and n in parameters else v
                         for n, v in value.items()}
            setattr(self.tuner.rankallocator, key, value)
        config.rank_pattern = deepcopy(state.get("rank_pattern", state.get("allocator", {}).get("rank_pattern")))

    def rank_report(self):
        report = super().rank_report()
        pattern = self.model.peft_config["default"].rank_pattern
        # Use the actual allocation mask, not init_r or zero singular values at init.
        if pattern:
            report["active_ranks"] = {key: int(sum(mask)) for key, mask in pattern.items()}
        return report

    def save_adapter(self, destination):
        import json
        if self.model is None:
            raise RuntimeError("attach() must be called first")
        config = self.model.peft_config["default"]
        pattern = deepcopy(config.rank_pattern)
        try:
            # Keep full-shaped factors and masked E: equivalent inference without
            # rank-zero tensors or ambiguous rank_pattern key remapping on reload.
            config.rank_pattern = None
            super().save_adapter(destination)
        finally:
            config.rank_pattern = pattern
        Path(destination, "training_method.json").write_text(json.dumps({
            "method": self.name, "export": "full_factors_masked_singular_values",
            "rank_report": self.rank_report(),
        }, indent=2), encoding="utf-8")
        return Path(destination)
