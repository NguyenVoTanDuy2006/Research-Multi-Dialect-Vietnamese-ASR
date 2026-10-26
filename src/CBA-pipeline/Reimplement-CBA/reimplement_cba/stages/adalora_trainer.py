from __future__ import annotations

import gc
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from peft import AdaLoraConfig, PeftModel, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from reimplement_cba.adapters.whisper_data import (
    PhoWhisperDataCollator,
    PhoWhisperTrainingDataset,
)
from reimplement_cba.config import CBAConfig
from reimplement_cba.stages.adalora_checkpoint import full_rank_save_config


def _find_adalora_model(model: Any) -> Any | None:
    """Find the PEFT AdaLoRA wrapper through common DDP/PEFT wrappers."""
    pending = [model]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if callable(getattr(current, "update_and_allocate", None)):
            return current
        for attribute in ("module", "base_model"):
            nested = getattr(current, attribute, None)
            if nested is not None:
                pending.append(nested)
    return None


class AdaLoRAAllocationCallback(TrainerCallback):
    """Update AdaLoRA rank allocation after each optimizer update."""

    def __init__(self, total_steps: int) -> None:
        if total_steps < 1:
            raise ValueError("AdaLoRA total_steps must be positive")
        self.total_steps = int(total_steps)
        self.last_step: int | None = None

    def on_optimizer_step(self, args, state, control, model=None, **kwargs):
        # Trainer calls this after optimizer.step() and before it increments
        # global_step or clears gradients, matching PEFT's Whisper example.
        optimizer_step = int(state.global_step)
        if optimizer_step >= self.total_steps:
            return control
        if optimizer_step == self.last_step:
            raise RuntimeError(f"AdaLoRA allocation hook ran twice for optimizer step {optimizer_step}")

        adalora_model = _find_adalora_model(model)
        if adalora_model is None:
            raise RuntimeError(
                "Trainer could not find AdaLoRA.update_and_allocate(); "
                "dynamic rank allocation cannot run with this Trainer/PEFT model"
            )
        adalora_model.update_and_allocate(optimizer_step)
        self.last_step = optimizer_step
        return control


class AdaLoRASeq2SeqTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer with PEFT's required adaptive-rank update hook."""

    def __init__(self, *args, allocation_total_steps: int, callbacks=None, **kwargs) -> None:
        trainer_callbacks = list(callbacks or [])
        trainer_callbacks.append(AdaLoRAAllocationCallback(allocation_total_steps))
        super().__init__(*args, callbacks=trainer_callbacks, **kwargs)

    def save_model(self, *args, **kwargs):
        # Trainer calls this for epoch checkpoints. Keep their AdaLoRA factor
        # shapes resumable so validation may select any saved epoch.
        with full_rank_save_config(self.model):
            return super().save_model(*args, **kwargs)


class AdaLoRATrainer:
    """Train or continue one AdaLoRA curriculum round on selected ViMD records."""

    def __init__(self, config: CBAConfig):
        self.config = config

    def _runtime_device_dtype(self) -> tuple[str, torch.dtype, str]:
        model_values = self.config.values["model"]
        requested_device = str(model_values.get("device", "auto"))
        if requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("model.device is 'cuda' but CUDA is not available")
            device = "cuda"
        else:
            device = "cpu"

        compute_type = str(model_values.get("compute_type", "float16"))
        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return device, dtype_map[compute_type], compute_type

    @staticmethod
    def _set_total_steps(model: Any, total_steps: int) -> None:
        configs = getattr(model, "peft_config", None)
        if not isinstance(configs, dict) or not configs:
            raise RuntimeError("Loaded model does not expose a PEFT adapter configuration")
        for adapter_config in configs.values():
            peft_type = getattr(adapter_config, "peft_type", None)
            peft_type = getattr(peft_type, "value", peft_type)
            if str(peft_type).upper() not in {"ADALORA", "PEFTTYPE.ADALORA"}:
                raise ValueError(f"Expected an AdaLoRA adapter, got PEFT type {peft_type!r}")
            adapter_config.total_step = total_steps

    @staticmethod
    def _write_run_metadata(round_dir: Path, metadata: dict[str, Any]) -> None:
        (round_dir / "resolved-config.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _save_resumable_adapter(trainer: Any, model: Any, destination: Path) -> dict[str, Any]:
        """Save full AdaLoRA factor shapes so later CL rounds can resume allocation.

        PEFT normally compacts saved factor tensors using ``rank_pattern``. For
        curriculum continuation, retain the initialized rank dimensions and the
        learned zeroed singular values; a new allocator can then continue from
        the prior weights without loading compact tensors into a full-rank
        allocator state.
        """
        with full_rank_save_config(model) as rank_patterns:
            trainer.save_model(str(destination))
        return rank_patterns

    def run(
        self,
        train_records,
        validation_records,
        *,
        audio_reader,
        round_index: int,
        resume_adapter: str | None = None,
    ) -> str:
        if round_index < 1:
            raise ValueError("round_index must be at least 1")
        if not train_records:
            raise ValueError("AdaLoRA training requires at least one training record")
        if not validation_records:
            raise ValueError("AdaLoRA training requires at least one validation record")

        model_values = self.config.values["model"]
        training_values = self.config.values["training"]
        model_checkpoint = self.config.resolve_model_reference(model_values["base_model"])
        processor_reference = model_values.get("processor_model") or model_values["base_model"]
        processor_checkpoint = self.config.resolve_model_reference(processor_reference)
        revision = model_values.get("revision")
        revision_kwargs = {"revision": revision} if revision else {}
        audio_values = self.config.values["audio"]
        audio_limits = {
            "min_duration_seconds": float(audio_values["min_duration_seconds"]),
            "max_duration_seconds": float(audio_values["max_duration_seconds"]),
        }

        processor = WhisperProcessor.from_pretrained(
            processor_checkpoint,
            language=model_values["language"],
            task=model_values["task"],
            **revision_kwargs,
        )
        train_dataset = PhoWhisperTrainingDataset(
            train_records,
            audio_reader=audio_reader,
            processor=processor,
            **audio_limits,
        )
        validation_dataset = PhoWhisperTrainingDataset(
            validation_records,
            audio_reader=audio_reader,
            processor=processor,
            **audio_limits,
        )

        epochs = float(training_values["epochs"])
        batch_size = int(training_values["batch_size"])
        accumulation_steps = int(training_values.get("gradient_accumulation_steps", 1))
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        updates_per_epoch = math.ceil(
            len(train_dataset) / (batch_size * world_size * accumulation_steps)
        )
        total_steps = max(1, math.ceil(updates_per_epoch * epochs))

        device, dtype, resolved_compute_type = self._runtime_device_dtype()
        model = WhisperForConditionalGeneration.from_pretrained(
            model_checkpoint,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        model.config.use_cache = False

        if resume_adapter:
            adapter_reference = self.config.resolve_model_reference(resume_adapter)
            model = PeftModel.from_pretrained(
                model,
                adapter_reference,
                is_trainable=True,
            )
            self._set_total_steps(model, total_steps)
        else:
            adalora_config = AdaLoraConfig(
                # Keep PEFT's generic wrapper so Whisper's input_features
                # signature is passed through unchanged. AdaLoraModel still
                # supplies the AdaLoRA forward pass and orthogonal regularizer.
                init_r=int(training_values["init_r"]),
                target_r=int(training_values["target_r"]),
                target_modules=list(training_values["target_modules"]),
                total_step=total_steps,
            )
            model = get_peft_model(model, adalora_config)
            self._set_total_steps(model, total_steps)

        if _find_adalora_model(model) is None:
            raise RuntimeError("The active PEFT model does not provide AdaLoRA rank allocation")

        round_dir = self.config.output_dir / "adapters" / f"round-{round_index}"
        trainer_output_dir = round_dir / "trainer"
        final_adapter_dir = round_dir / "checkpoint-final"
        trainer_output_dir.mkdir(parents=True, exist_ok=True)

        use_cpu = device == "cpu"
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(trainer_output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=accumulation_steps,
            learning_rate=float(training_values["learning_rate"]),
            warmup_steps=int(training_values["warmup_steps"]),
            num_train_epochs=epochs,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=max(
                math.ceil(epochs), int(training_values.get("save_total_limit", 10))
            ),
            load_best_model_at_end=False,
            fp16=(device == "cuda" and resolved_compute_type == "float16"),
            bf16=(device == "cuda" and resolved_compute_type == "bfloat16"),
            use_cpu=use_cpu,
            report_to=[],
            logging_strategy="epoch",
            dataloader_num_workers=0,
            remove_unused_columns=False,
            label_names=["labels"],
        )
        trainer = AdaLoRASeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=PhoWhisperDataCollator(
                processor,
                decoder_start_token_id=getattr(model.config, "decoder_start_token_id", None),
            ),
            processing_class=processor,
            allocation_total_steps=total_steps,
        )
        round_dir.mkdir(parents=True, exist_ok=True)
        run_metadata = {
            "status": "running",
            "round_index": round_index,
            "model_checkpoint": model_checkpoint,
            "processor_checkpoint": processor_checkpoint,
            "revision": revision,
            "resume_adapter": str(resume_adapter) if resume_adapter else None,
            "train_records": len(train_dataset),
            "validation_records": len(validation_dataset),
            "optimizer_steps_planned": total_steps,
            "compute_type": resolved_compute_type,
            "device": device,
            "training_config": training_values,
            "peft_config": {
                name: configuration.to_dict()
                for name, configuration in model.peft_config.items()
            },
            "final_adapter": str(final_adapter_dir),
        }
        self._write_run_metadata(round_dir, run_metadata)
        trainer.train()
        model.config.use_cache = True
        rank_patterns = self._save_resumable_adapter(trainer, model, final_adapter_dir)
        run_metadata.update(
            {
                "status": "complete",
                "optimizer_steps_completed": int(trainer.state.global_step),
                "allocated_rank_pattern": rank_patterns,
                "adapter_save_policy": "keep_init_rank_with_pruned_singular_values",
            }
        )
        self._write_run_metadata(round_dir, run_metadata)
        adapter_path = str(final_adapter_dir)
        del trainer, model, processor, train_dataset, validation_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return adapter_path
