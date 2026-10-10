from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import os
import shutil
import tempfile

from mdv_asr_module.config.types import PiSSASettings
from .base import FineTuningMethod


class PiSSAMethod(FineTuningMethod):
    """PEFT PiSSA initialization, fixed-rank training, and portable LoRA export.

    Native resume stores initial factors AND targeted residual weights. The
    residual is necessary for exact fp16/bf16 resume (rounded factors alone do
    not reconstruct PEFT's rounded residual bit-for-bit). Export is ordinary
    rank-2r LoRA, to be loaded onto the ORIGINAL pretrained base.
    """
    name = "pissa"

    def __init__(self, settings: PiSSASettings | None = None):
        super().__init__(settings or PiSSASettings())
        self.initial_adapter_state = {}
        self.residual_state = {}
        self.initialization_path = None

    def attach(self, model, total_steps=None, *, initialization_state=None):
        import torch
        from peft import LoraConfig, get_peft_model

        self._select(model, self.settings.r)
        s = self.settings
        if initialization_state is not None:
            if initialization_state.get("name") != self.name or initialization_state.get("settings") != asdict(s):
                raise ValueError("PiSSA initialization settings do not match resume checkpoint")
        devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
        # Shared initialization seed on all DDP ranks; preserve per-rank training RNG.
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(s.init_seed)
            config = LoraConfig(
                r=s.r, lora_alpha=s.alpha, lora_dropout=s.dropout, target_modules=self.target_names,
                bias="none", task_type=None,
                init_lora_weights=s.init_lora_weights if initialization_state is None else True,
            )
            self.model = get_peft_model(model, config)
        current = self.model.state_dict()
        adapter_keys = {k for k in current if ".lora_A.default." in k or ".lora_B.default." in k}
        residual_keys = {k.replace(".lora_A.default.weight", ".base_layer.weight")
                         for k in adapter_keys if ".lora_A." in k}
        if initialization_state is not None:
            initial = initialization_state.get("initial_adapter_state", {})
            residual = initialization_state.get("residual_state", {})
            if set(initial) != adapter_keys or set(residual) != residual_keys:
                raise ValueError("PiSSA resume is missing initial factors or residual weights")
            for key, value in {**initial, **residual}.items():
                if value.shape != current[key].shape or value.dtype != current[key].dtype:
                    raise ValueError(f"PiSSA initialization tensor shape/dtype mismatch: {key}")
                current[key].copy_(value.to(current[key].device))
            # Annotation only; no second decomposition of the original weights.
            self.model.peft_config["default"].init_lora_weights = s.init_lora_weights
        self.initial_adapter_state = {k: current[k].detach().cpu().clone() for k in adapter_keys}
        self.residual_state = {k: current[k].detach().cpu().clone() for k in residual_keys}
        return self.model

    def state_dict(self):
        return {**super().state_dict(), "initial_adapter_state": self.initial_adapter_state,
                "residual_state": self.residual_state}

    def load_state_dict(self, state):
        import torch
        for field in ("initial_adapter_state", "residual_state"):
            current, saved = getattr(self, field), state.get(field, {})
            if current.keys() != saved.keys() or any(not torch.equal(current[k], saved[k].cpu()) for k in current):
                raise ValueError(f"PiSSA {field} differs: attach with checkpoint initialization_state before restoring")
        super().load_state_dict(state)

    def save_initialization(self, destination):
        import torch
        from peft import get_peft_model_state_dict
        from safetensors.torch import load_file

        if self.model is None:
            raise RuntimeError("attach() must be called first")
        destination = Path(destination).resolve()
        if destination.name == "default":
            raise ValueError("Initial adapter directory must not be named 'default'")
        if destination.exists():
            weights = load_file(str(destination / "adapter_model.safetensors"))
            expected = get_peft_model_state_dict(self.model, state_dict=self.initial_adapter_state)
            if weights.keys() != expected.keys() or any(not torch.equal(weights[k], expected[k]) for k in weights):
                raise ValueError(f"Existing PiSSA initialization does not match: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".pissa-initial-", dir=destination.parent))
            config = self.model.peft_config["default"]
            old_init = config.init_lora_weights
            try:
                config.init_lora_weights = True
                self.model.save_pretrained(staging, state_dict=self.initial_adapter_state,
                                           safe_serialization=True, save_embedding_layers=False)
                os.replace(staging, destination)
            except Exception:
                shutil.rmtree(staging, ignore_errors=True)
                raise
            finally:
                config.init_lora_weights = old_init
        self.initialization_path = destination
        return destination

    def save_adapter(self, destination):
        import torch
        if self.initialization_path is None:
            raise RuntimeError("Save PiSSA initialization before exporting an inference adapter")
        model = self.model
        configs = deepcopy(model.peft_config)
        active = model.active_adapter
        train_flags = {n: m.training for n, m in model.named_modules()}
        requires_grad = {n: p.requires_grad for n, p in model.named_parameters()}
        devices = sorted({p.device.index for p in model.parameters() if p.device.type == "cuda"})
        try:
            # PEFT temporarily loads the initial adapter. Preserve training RNG,
            # active adapters, requires_grad, and configs for the next epoch.
            with torch.random.fork_rng(devices=devices):
                model.save_pretrained(
                    destination, selected_adapters=["default"], safe_serialization=True,
                    save_embedding_layers=False,
                    path_initial_model_for_weight_conversion=str(self.initialization_path),
                )
        finally:
            for name, config in configs.items():
                model.peft_config[name] = config
            model.set_adapter(active)
            for n, p in model.named_parameters():
                if n in requires_grad:
                    p.requires_grad_(requires_grad[n])
            for n, m in model.named_modules():
                if n in train_flags:
                    m.training = train_flags[n]
        Path(destination, "training_method.json").write_text(json.dumps({
            "method": self.name, "export": "converted_standard_lora",
            "training_rank": self.settings.r, "export_rank": 2 * self.settings.r,
            "load_on": "original_pretrained_base", "settings": asdict(self.settings),
            "note": "Low-precision residual rounding can cause small numeric differences after conversion.",
        }, indent=2), encoding="utf-8")
        return Path(destination)
