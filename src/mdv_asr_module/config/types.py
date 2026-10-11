from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ValidatedConfig:
    def validate(self) -> None:
        return None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DatasetConfig(ValidatedConfig):
    kind: str = "auto"
    root: str | None = None
    handle: str | None = None
    dataset_id: str = "vimd"
    cache_dir: str | None = None
    allow_download: bool = False

    def validate(self) -> None:
        if self.kind not in {"auto", "arrow", "parquet"}:
            raise ValueError(f"Unsupported dataset source kind: {self.kind}")
        if self.root is not None and not str(self.root).strip():
            raise ValueError("Dataset root cannot be an empty string")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id is required")


@dataclass(frozen=True)
class ModelSpec(ValidatedConfig):
    key: str
    backend: str
    model_id: str
    revision: str | None = None
    adapter_path: str | None = None
    display_name: str | None = None
    dtype: str = "float16"
    trust_remote_code: bool = False

    def validate(self) -> None:
        if not self.key or not self.backend or not self.model_id:
            raise ValueError("Model key, backend and model_id are required")
        if self.dtype not in {"float16", "bfloat16", "float32", "auto"}:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.adapter_path is not None and not str(self.adapter_path).strip():
            raise ValueError("adapter_path cannot be empty")


@dataclass(frozen=True)
class AdaLoraSettings(ValidatedConfig):
    init_r: int = 16
    target_r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    tinit_ratio: float = 0.05
    tfinal_ratio: float = 0.05
    delta_t: int = 10

    def validate(self) -> None:
        if self.init_r < 1 or self.target_r < 1 or self.target_r > self.init_r:
            raise ValueError("AdaLoRA requires 1 <= target_r <= init_r")
        if self.alpha < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid AdaLoRA alpha/dropout")
        if not self.target_modules or self.delta_t < 1:
            raise ValueError("AdaLoRA target modules and delta_t are required")
        if min(self.tinit_ratio, self.tfinal_ratio) < 0 or self.tinit_ratio + self.tfinal_ratio >= 1:
            raise ValueError("AdaLoRA warmup/final ratios must be nonnegative and sum to less than 1")


@dataclass(frozen=True)
class PiSSASettings(ValidatedConfig):
    r: int = 8
    alpha: int = 16
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    init_lora_weights: str = "pissa_niter_4"
    init_seed: int = 24127159

    def validate(self) -> None:
        import re
        if self.r < 1 or self.alpha < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid PiSSA rank/alpha/dropout")
        if not self.target_modules:
            raise ValueError("PiSSA target_modules are required")
        if self.init_lora_weights != "pissa" and not re.fullmatch(r"pissa_niter_[1-9][0-9]*", self.init_lora_weights):
            raise ValueError("Use 'pissa' or 'pissa_niter_<positive integer>'")
        if self.init_seed < 0:
            raise ValueError("PiSSA initialization seed must be nonnegative")


@dataclass(frozen=True)
class DecodeConfig(ValidatedConfig):
    language: str = "Vietnamese"
    max_new_tokens: int = 768
    do_sample: bool = False
    num_beams: int = 1

    def validate(self) -> None:
        if self.max_new_tokens < 1 or self.num_beams < 1:
            raise ValueError("Decode limits must be positive")


@dataclass(frozen=True)
class TrainConfig(ValidatedConfig):
    epochs: int = 2
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    gradient_clip_norm: float = 1.0
    precision: str = "fp16"
    num_workers: int = 2
    world_size: int = 2
    seed: int = 24127159
    checkpoint_every_epoch: bool = True
    adalora: AdaLoraSettings = field(default_factory=AdaLoraSettings)
    method: str = "adalora"
    pissa: PiSSASettings = field(default_factory=PiSSASettings)

    def validate(self) -> None:
        if min(self.epochs, self.per_device_batch_size,
               self.gradient_accumulation_steps, self.world_size) < 1:
            raise ValueError("Training counts and world_size must be positive")
        if self.num_workers < 0 or self.learning_rate <= 0:
            raise ValueError("Invalid num_workers or learning_rate")
        if not 0 <= self.warmup_ratio < 1 or self.gradient_clip_norm <= 0:
            raise ValueError("Invalid warmup ratio or gradient clip norm")
        if self.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        self.adalora.validate()
        self.pissa.validate()
        if self.method not in {"adalora", "pissa"}:
            raise ValueError(f"Unsupported fine-tuning method: {self.method}")


@dataclass(frozen=True)
class BenchmarkConfig(ValidatedConfig):
    models: tuple[ModelSpec, ...] = ()
    split: str = "test"
    resume: bool = True
    flush_every: int = 10

    def validate(self) -> None:
        if not self.split or self.flush_every < 1:
            raise ValueError("Benchmark split and flush interval are required")
        keys = [model.key for model in self.models]
        if len(keys) != len(set(keys)):
            raise ValueError("Model keys must be unique")
        if not self.models:
            raise ValueError("Benchmark requires at least one model")
        for model in self.models:
            model.validate()


@dataclass(frozen=True)
class ExperimentConfig(ValidatedConfig):
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    platform: str = "auto"
    model: ModelSpec | None = None
    decode: DecodeConfig = field(default_factory=DecodeConfig)
    train: TrainConfig | None = None
    benchmark: BenchmarkConfig | None = None
    output_dir: str = "outputs/runs/default"
    run_id: str = "default"

    def validate(self) -> None:
        self.dataset.validate()
        if self.platform not in {"auto", "local", "kaggle", "colab"}:
            raise ValueError(f"Unsupported runtime platform: {self.platform}")
        self.decode.validate()
        if self.train is not None and self.model is None:
            raise ValueError("Training configuration requires a model section")
        if self.benchmark is not None and not self.benchmark.models:
            raise ValueError("Benchmark configuration requires at least one model")
        if self.model:
            self.model.validate()
        if self.train:
            self.train.validate()
        if self.benchmark:
            self.benchmark.validate()
        if not self.run_id.strip() or not str(Path(self.output_dir)):
            raise ValueError("run_id and output_dir are required")

