from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mdv_asr_module.core.errors import CheckpointMismatchError


class AdapterAuditor:
    def inspect_files(self, adapter_path: str | Path) -> dict[str, Any]:
        adapter_path = Path(adapter_path)
        required = ["adapter_config.json", "adapter_model.safetensors"]
        missing = [name for name in required if not (adapter_path / name).is_file()]
        if missing:
            raise FileNotFoundError(f"Adapter is missing files: {missing}")
        config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
        return {"path": str(adapter_path.resolve()), "files": required,
                "base_model_name_or_path": config.get("base_model_name_or_path"),
                "peft_type": config.get("peft_type"), "rank_pattern": config.get("rank_pattern", {})}

    def inspect_tensors(self, adapter_path: str | Path) -> dict[str, Any]:
        from safetensors import safe_open

        path = Path(adapter_path) / "adapter_model.safetensors"
        tensors = {}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                shape = tuple(handle.get_slice(key).get_shape())
                tensors[key] = {"shape": shape, "zero_shape": any(dim == 0 for dim in shape)}
        return {"tensor_count": len(tensors),
                "zero_shape_count": sum(value["zero_shape"] for value in tensors.values()),
                "tensors": tensors}

    def check_rank_pattern(self, adapter_path: str | Path) -> dict[str, Any]:
        info = self.inspect_files(adapter_path)
        tensors = self.inspect_tensors(adapter_path)["tensors"]
        patterns = info["rank_pattern"] or {}
        zero_keys = [key for key, value in tensors.items() if value["zero_shape"]]
        mismatches = []
        for key, value in tensors.items():
            if ".lora_A." not in key and ".lora_B." not in key:
                continue
            pattern_match = next((rank for module, rank in patterns.items() if module in key), None)
            if pattern_match is None:
                continue
            rank_dim = value["shape"][0] if ".lora_A." in key else value["shape"][-1]
            expected_rank = sum(pattern_match) if isinstance(pattern_match, list) else int(pattern_match)
            if rank_dim != expected_rank:
                mismatches.append({"key": key, "tensor_rank": rank_dim, "pattern_rank": pattern_match})
        return {"zero_shape_tensors": len(zero_keys), "rank_mismatches": mismatches,
                "rank_pattern_key_count": len(patterns), "zero_shape_keys": zero_keys}

    def compare_loaded_state(self, adapter_path: str | Path, model,
                             *, atol: float = 0.0, rtol: float = 0.0) -> dict[str, Any]:
        import torch
        from safetensors.torch import load_file

        path = Path(adapter_path) / "adapter_model.safetensors"
        saved = load_file(str(path), device="cpu")
        if getattr(model, "peft_config", None):
            from peft import get_peft_model_state_dict
            # PEFT files omit adapter names (e.g. .default) from canonical keys.
            # Compare canonical state, not a guessed live-module prefix.
            loaded = get_peft_model_state_dict(model, adapter_name="default")
        else:
            loaded = model.state_dict()
        compared, missing, shape_mismatch, value_mismatch = [], [], [], []
        for key, expected in saved.items():
            candidates = [key]
            if key.startswith("base_model.model."):
                candidates.append(key.removeprefix("base_model.model."))
            actual_key = next((candidate for candidate in candidates if candidate in loaded), None)
            if actual_key is None:
                missing.append(key)
                continue
            actual = loaded[actual_key].detach().cpu()
            if tuple(actual.shape) != tuple(expected.shape):
                shape_mismatch.append({"key": key, "saved": tuple(expected.shape), "loaded": tuple(actual.shape)})
                continue
            if not torch.allclose(actual.to(expected.dtype), expected, atol=atol, rtol=rtol, equal_nan=False):
                value_mismatch.append(key)
            else:
                compared.append(key)
        report = {"saved_tensor_count": len(saved), "compared_tensor_count": len(compared),
                  "missing_keys": missing, "shape_mismatches": shape_mismatch,
                  "value_mismatches": value_mismatch}
        if missing or shape_mismatch or value_mismatch:
            raise CheckpointMismatchError(f"Loaded adapter state differs from file: {report}")
        return report

    def check_forward(self, model, batch: dict[str, Any]) -> float:
        import torch

        model.eval()
        with torch.inference_mode():
            result = model.generate(**batch, max_new_tokens=8, do_sample=False)
        if not torch.is_tensor(result) or result.numel() == 0:
            raise CheckpointMismatchError("Adapter model generated an empty output")
        return float(result.numel())

    def audit(self, adapter_path: str | Path) -> dict[str, Any]:
        return {"files": self.inspect_files(adapter_path), **self.check_rank_pattern(adapter_path)}

