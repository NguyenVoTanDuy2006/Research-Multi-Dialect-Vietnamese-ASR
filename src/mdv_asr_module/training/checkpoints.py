from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mdv_asr_module.core.artifacts import ArtifactStore
from mdv_asr_module.core.errors import ResumeMismatchError


class CheckpointManager:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.store = ArtifactStore(self.output_dir)

    def save_epoch_adapter(self, model, processor, epoch: int, *, method=None) -> Path:
        import shutil

        destination = self.output_dir / f"epoch_{epoch}_adapter"
        if destination.exists():
            raise FileExistsError(f"Adapter checkpoint already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-",
                                        dir=destination.parent))
        try:
            if method is None:
                model.save_pretrained(staging)
            else:
                if method.model is not model:
                    raise ValueError("Checkpoint method does not own the supplied model")
                method.save_adapter(staging)
            processor.save_pretrained(staging)
            os.replace(staging, destination)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return destination

    @staticmethod
    def capture_rng_state() -> dict[str, Any]:
        return {"python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

    @staticmethod
    def restore_rng_state(state: dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(state["cuda"])

    @staticmethod
    def adapter_state(model) -> dict[str, torch.Tensor]:
        state = model.state_dict()
        return {key: value.detach().cpu().clone() for key, value in state.items()
                if "lora_" in key or "ranknum" in key or "rank_mask" in key}

    def save_training_state(self, *, epoch: int, model, optimizer_controller,
                            progress, history, rng_by_rank,
                            config_fingerprint: str, model_revision: str | None,
                            method=None, ada_manager=None) -> Path:
        method = method if method is not None else ada_manager
        if method is None:
            raise ValueError("A fine-tuning method is required for resumable checkpoints")
        destination = self.output_dir / "training_checkpoints" / f"epoch_{epoch:04d}.pt"
        if destination.exists():
            raise FileExistsError(f"Training checkpoint already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "epoch": epoch,
            "adapter_state": self.adapter_state(model),
            "optimizer_controller": optimizer_controller.state_dict(),
            "method_name": method.name,
            "method_state": method.state_dict(),
            "progress": progress.to_dict(),
            "history": history,
            "rng_by_rank": rng_by_rank,
            "config_fingerprint": config_fingerprint,
            "model_revision": model_revision,
        }
        fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
        os.close(fd)
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        except Exception:
            Path(temporary).unlink(missing_ok=True)
            raise
        return destination

    def load_training_state(self, path: str | Path, *, model, optimizer_controller,
                            context, config_fingerprint: str,
                            model_revision: str | None, method=None, ada_manager=None,
                            compatible_config_fingerprints=()) -> dict[str, Any]:
        method = method if method is not None else ada_manager
        payload = self.read_training_state(path)
        allowed = {config_fingerprint}
        if payload["schema_version"] == 1 and method is not None and method.name == "adalora":
            allowed.update(compatible_config_fingerprints)
        if payload.get("config_fingerprint") not in allowed:
            raise ResumeMismatchError("Training configuration does not match checkpoint")
        if payload.get("model_revision") != model_revision:
            raise ResumeMismatchError("Base model revision does not match checkpoint")
        if method is None or payload["method_name"] != method.name:
            raise ResumeMismatchError("Checkpoint fine-tuning method does not match")
        model_state = model.state_dict()
        saved = payload["adapter_state"]
        expected_keys = set(self.adapter_state(model))
        missing = sorted(set(saved) ^ expected_keys)
        shape_errors = [key for key in saved if key in model_state
                        and tuple(saved[key].shape) != tuple(model_state[key].shape)]
        if missing or shape_errors:
            raise ResumeMismatchError(f"Adapter state mismatch: missing={missing[:5]}, shape={shape_errors[:5]}")
        model_state.update(saved)
        model.load_state_dict(model_state, strict=True)
        optimizer_controller.load_state_dict(payload["optimizer_controller"])
        method.load_state_dict(payload["method_state"])
        rank = context.rank if context is not None else 0
        rng_states = payload.get("rng_by_rank", [])
        expected_world = context.world_size if context is not None else 1
        if len(rng_states) != expected_world:
            raise ResumeMismatchError("Exact resume requires RNG state for every current DDP rank")
        if rank < len(rng_states):
            self.restore_rng_state(rng_states[rank])
        return {"epoch": int(payload["epoch"]), "history": payload.get("history", []),
                "progress": payload.get("progress", {})}

    @staticmethod
    def read_training_state(path: str | Path) -> dict[str, Any]:
        # Only load checkpoints created by this project: torch pickle is not safe
        # for arbitrary/untrusted input artifacts.
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") not in {1, 2}:
            raise ResumeMismatchError("Unsupported training checkpoint schema")
        if payload["schema_version"] == 1:
            payload["method_name"] = "adalora"
            payload["method_state"] = payload["adalora_state"]
        return payload

    def update_best(self, epoch: int, metrics: dict[str, Any], adapter_path: str | Path) -> Path:
        return self.store.write_json("best_checkpoint.json", {
            "epoch": epoch, "metrics": metrics, "adapter_path": str(adapter_path),
        })

    def get_best_adapter(self) -> Path:
        metadata = self.store.read_json("best_checkpoint.json")
        path = Path(metadata["adapter_path"])
        if not path.is_dir():
            raise FileNotFoundError(f"Best adapter path is missing: {path}")
        return path

    def validate_resume(self, path: str | Path, *, config_fingerprint: str,
                        model_revision: str | None) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("config_fingerprint") != config_fingerprint:
            raise ResumeMismatchError("Resume config fingerprint mismatch")
        if payload.get("model_revision") != model_revision:
            raise ResumeMismatchError("Resume model revision mismatch")
        return payload

