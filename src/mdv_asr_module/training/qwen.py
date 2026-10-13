from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import get_linear_schedule_with_warmup

from mdv_asr_module.configs.types import DecodeConfig, ModelSpec, TrainConfig
from mdv_asr_module.core.artifacts import hash_payload
from mdv_asr_module.data.sources import DatasetSourceFactory
from mdv_asr_module.evaluation.evaluator import ASREvaluator
from mdv_asr_module.evaluation.error_metrics import MetricAggregator
from mdv_asr_module.models.base import ModelBundle
from mdv_asr_module.models.qwen import QwenASRBackend, QwenModelLoader
from mdv_asr_module.runtime.distributed import DistributedContext

from .checkpoints import CheckpointManager
from .collator import QwenTrainingCollator
from .dataset import ASRDataLoaderFactory
from .optimization import OptimizerStepController, StepResult, TrainProgress
from .methods import FineTuningMethod, create_method, AdaLoRAMethod


class QwenTrainer:
    def __init__(self, *, records_by_split, dataset_config, model_spec: ModelSpec,
                 train_config: TrainConfig, decode_config: DecodeConfig,
                 output_dir: str | Path, context: DistributedContext | None = None,
                 token: str | None = None, method: FineTuningMethod | None = None):
        from dataclasses import replace
        if method is not None:
            train_config = replace(train_config, method=method.name, **{method.name: method.settings})
        train_config.validate()
        model_spec.validate()
        self.records_by_split = records_by_split
        self.dataset_config = dataset_config
        self.model_spec = model_spec
        self.train_config = train_config
        self.decode_config = decode_config
        self.output_dir = Path(output_dir)
        self.context = context or DistributedContext.from_environment()
        self.loader = QwenModelLoader(token=token)
        self.checkpoints = CheckpointManager(self.output_dir)
        self.metric_aggregator = MetricAggregator()
        self.bundle: ModelBundle | None = None
        self.model = None
        self.processor = None
        self.method = method or create_method(train_config)
        self.optimizer_controller = None
        self.progress = TrainProgress()
        self.history: list[dict] = []
        self.train_loader = None
        self.train_sampler = None
        self.total_steps = 0
        self.config_fingerprint = hash_payload({
            "dataset": dataset_config.to_dict(), "model": model_spec.to_dict(),
            "train": train_config.to_dict(), "decode": decode_config.to_dict(),
        })
        legacy_train = train_config.to_dict()
        legacy_train.pop("method")
        legacy_train.pop("pissa")
        self.legacy_config_fingerprints = (hash_payload({
            "dataset": dataset_config.to_dict(), "model": model_spec.to_dict(),
            "train": legacy_train, "decode": decode_config.to_dict(),
        }),) if self.method.name == "adalora" else ()

    def setup(self, *, total_steps_override: int | None = None, initialization_state=None) -> None:
        self.context.initialize()
        seed = self.train_config.seed + self.context.rank
        torch.manual_seed(seed)
        np.random.seed(seed % (2**32))
        random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processor = self.loader.load_processor(self.model_spec)
        collator = QwenTrainingCollator(self.processor, self.decode_config.language)
        loader_factory = ASRDataLoaderFactory(self.train_config, self.context)
        self.train_loader, self.train_sampler = loader_factory.create(
            self.records_by_split["train"], self.dataset_config, "train", collator, shuffle=True
        )
        steps_per_epoch = loader_factory.steps_per_epoch(
            self.train_loader, self.train_config.gradient_accumulation_steps
        )
        self.total_steps = int(total_steps_override or (steps_per_epoch * self.train_config.epochs))
        model_dtype = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}[
            self.train_config.precision
        ]
        self.bundle = self.loader.load_base(self.model_spec, str(self.context.device), dtype=model_dtype)
        base = self.bundle.model
        self.model = self.method.attach(base, self.total_steps, initialization_state=initialization_state)
        self.model.peft_config["default"].revision = self.bundle.revision
        if self.context.is_main:
            print("Fine-tuning method:", self.method.summary(), flush=True)
        self.model.config.use_cache = False
        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            self.model.gradient_checkpointing_enable()
        if self.context.is_distributed:
            self.model = DistributedDataParallel(
                self.model, device_ids=[self.context.local_rank] if self.context.device.type == "cuda" else None,
                output_device=self.context.local_rank if self.context.device.type == "cuda" else None,
                find_unused_parameters=False,
            )
        parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=self.train_config.learning_rate,
                                      weight_decay=self.train_config.weight_decay)
        warmup_steps = int(self.train_config.warmup_ratio * self.total_steps)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, self.total_steps)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.context.device.type == "cuda"
                                                       and self.train_config.precision == "fp16"))
        if not scaler.is_enabled():
            scaler = None
        self.optimizer_controller = OptimizerStepController(
            optimizer, scaler=scaler, scheduler=scheduler,
            gradient_clip_norm=self.train_config.gradient_clip_norm, context=self.context,
        )

    def _unwrap_model(self):
        return self.model.module if isinstance(self.model, DistributedDataParallel) else self.model

    def _autocast(self):
        enabled = self.context.device.type == "cuda" and self.train_config.precision in {"fp16", "bf16"}
        dtype = torch.float16 if self.train_config.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    def _move_batch(self, batch):
        device = self.context.device
        dtype = next(self._unwrap_model().parameters()).dtype
        moved = {}
        for key, value in batch.items():
            if key == "records":
                continue
            if torch.is_tensor(value):
                moved[key] = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            else:
                moved[key] = value
        return moved

    def _backward_loss(self, batch):
        inputs = self._move_batch(batch)
        with self._autocast():
            output = self.model(**inputs)
            loss = output.get("loss") if isinstance(output, dict) else output.loss
            if isinstance(loss, dict):
                loss = loss.get("loss")
            if loss is None or not torch.is_tensor(loss):
                raise TypeError("Qwen forward did not return a tensor loss")
        finite = bool(torch.isfinite(loss).all().item())
        finite = self.context.all_ranks_agree(finite)
        if not finite:
            raise FloatingPointError("Non-finite loss occurred on at least one DDP rank")
        return loss

    def smoke(self) -> dict:
        if self.model is None:
            from dataclasses import replace

            if self.method.name == "adalora":
                self.method = AdaLoRAMethod(replace(self.train_config.adalora, delta_t=1))
            self.setup(total_steps_override=3)
        dataset = self.train_loader.dataset
        collator = QwenTrainingCollator(self.processor, self.decode_config.language)
        self.model.train()
        reports = []
        for index in range(3):
            item = dataset[index % len(dataset)]
            batch = collator([item])
            loss = self._backward_loss(batch)
            self.optimizer_controller.backward(loss, 1)
            result = self.optimizer_controller.step(self.model, self.method)
            self.progress.record_step(result)
            reports.append({"loss": float(loss.detach().cpu()), "step": result.__dict__})
        if self.progress.successful_steps < 1:
            raise RuntimeError("Smoke test did not complete a successful optimizer step")
        self.method.validate_rank_consistency(self.context)
        return {"rank": self.context.rank, "steps": reports,
                "targets": len(self.method.target_names),
                "method": self.method.name, "rank_report": self.method.rank_report()}

    def train_accumulation_group(self, batches: list[dict]) -> StepResult:
        if not batches:
            raise ValueError("Accumulation group cannot be empty")
        size = len(batches)
        losses = []
        for batch in batches:
            loss = self._backward_loss(batch)
            self.optimizer_controller.backward(loss, size)
            losses.append(float(loss.detach().cpu()))
            self.progress.micro_batches += 1
        self._last_group_mean_loss = float(np.mean(losses))
        result = self.optimizer_controller.step(self.model, self.method)
        self.progress.record_step(result)
        return result

    def train_epoch(self, epoch: int) -> dict:
        if self.model is None:
            self.setup()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch - 1)
        self.model.train()
        batch_group = []
        losses = []
        accumulation = self.train_config.gradient_accumulation_steps
        for batch_index, batch in enumerate(self.train_loader):
            batch_group.append(batch)
            if len(batch_group) == accumulation or batch_index + 1 == len(self.train_loader):
                result = self.train_accumulation_group(batch_group)
                losses.append(self._last_group_mean_loss)
                if result.skipped:
                    print(f"[rank {self.context.rank}] GradScaler skipped optimizer update at "
                          f"epoch={epoch}, batch={batch_index + 1}/{len(self.train_loader)}", flush=True)
                batch_group.clear()
            if (self.context.is_main and self.progress.attempted_steps > 0
                    and self.progress.attempted_steps % 50 == 0
                    and (batch_index + 1) % accumulation == 0):
                print(f"epoch={epoch}/{self.train_config.epochs} "
                      f"step={self.progress.successful_steps}/{self.total_steps} "
                      f"batch={batch_index + 1}/{len(self.train_loader)}", flush=True)
        if batch_group:
            raise RuntimeError("Final gradient accumulation group was not processed")
        gathered = self.context.gather_objects({"group_losses": losses})
        all_losses = [value for chunk in gathered for value in chunk["group_losses"]]
        mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
        return {"epoch": epoch, "train_loss": mean_loss,
                "mean_loss_across_ranks": mean_loss,
                "attempted_steps": self.progress.attempted_steps,
                "successful_steps": self.progress.successful_steps,
                "skipped_steps": self.progress.skipped_steps}

    def validate(self, epoch: int) -> dict:
        model = self._unwrap_model()
        model.eval()
        source = DatasetSourceFactory.create(self.dataset_config)
        bundle = ModelBundle(model=model, processor=self.processor,
                             revision=self.bundle.revision, device=str(self.context.device),
                             dtype=str(next(model.parameters()).dtype))
        backend = QwenASRBackend.from_loaded(self.model_spec, bundle)
        backend.decode_config = self.decode_config
        evaluator = ASREvaluator(context=self.context)
        try:
            result = evaluator.evaluate_model(backend, self.records_by_split["valid"], source)
            evaluator.validate_coverage(result)
        finally:
            source.close()
        metrics = self.metric_aggregator.overall(result.predictions)
        metrics["epoch"] = epoch
        metrics["optimizer_steps"] = self.progress.successful_steps
        return metrics

    def save_epoch(self, epoch: int, train_metrics: dict, valid_metrics: dict) -> Path | None:
        self.context.barrier()
        adapter_path = None
        rng_by_rank = self.context.gather_objects(self.checkpoints.capture_rng_state())
        if self.context.is_main:
            model = self._unwrap_model()
            adapter_path = self.checkpoints.save_epoch_adapter(model, self.processor, epoch, method=self.method)
            row = {**train_metrics, **{f"valid_{key.lower()}": value
                                      for key, value in valid_metrics.items()}}
            self.history.append(row)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.checkpoints.store.write_csv("train_history.csv", self.history)
            valid_cer = valid_metrics["CER"]
            best_path = self.output_dir / "best_checkpoint.json"
            previous = self.checkpoints.store.read_json("best_checkpoint.json") if best_path.exists() else None
            if previous is None or valid_cer < previous["metrics"]["CER"]:
                self.checkpoints.update_best(epoch, valid_metrics, adapter_path)
                (self.output_dir / "best_epoch.txt").write_text(str(epoch), encoding="utf-8")
            checkpoint = self.checkpoints.save_training_state(
                epoch=epoch, model=self._unwrap_model(),
                optimizer_controller=self.optimizer_controller,
                method=self.method, progress=self.progress,
                history=self.history, rng_by_rank=rng_by_rank,
                config_fingerprint=self.config_fingerprint,
                model_revision=self.bundle.revision,
            )
            print(f"Saved epoch adapter: {adapter_path}", flush=True)
            print(f"Saved resumable training state: {checkpoint}", flush=True)
        self.context.barrier()
        return adapter_path

    def fit(self, resume_from: str | Path | None = None) -> list[dict]:
        initialization_state = None
        if resume_from is not None and self.method.name == "pissa":
            if self.model is not None:
                raise ValueError("Resume PiSSA on a fresh trainer so its residual can be restored exactly")
            initialization_state = self.checkpoints.read_training_state(resume_from)["method_state"]
        if self.model is None:
            self.setup(initialization_state=initialization_state)
        if self.context.is_main:
            self.method.save_initialization(self.output_dir / "pissa_initial")
        self.context.barrier()
        start_epoch = 1
        if resume_from is not None:
            state = self.checkpoints.load_training_state(
                resume_from, model=self._unwrap_model(),
                optimizer_controller=self.optimizer_controller,
                method=self.method, context=self.context,
                config_fingerprint=self.config_fingerprint,
                model_revision=self.bundle.revision,
                compatible_config_fingerprints=self.legacy_config_fingerprints,
            )
            self.progress = TrainProgress.from_dict(state["progress"])
            self.history = list(state["history"])
            start_epoch = int(state["epoch"]) + 1
        for epoch in range(start_epoch, self.train_config.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            valid_metrics = self.validate(epoch)
            if self.context.is_main:
                print(f"epoch={epoch} train={train_metrics} valid={valid_metrics}", flush=True)
            self.save_epoch(epoch, train_metrics, valid_metrics)
        planned = self.total_steps
        self.progress.validate()
        if self.progress.attempted_steps != planned:
            raise RuntimeError(f"Attempted {self.progress.attempted_steps} optimizer steps; planned {planned}")
        return self.history

    def close(self) -> None:
        if self.bundle is not None:
            self.loader.release(self.bundle)
            self.bundle = None
        self.context.close()

