from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, TypeVar, get_origin, get_type_hints

from .types import BenchmarkConfig, ExperimentConfig

T = TypeVar("T")


def _construct(cls: type[T], value: Any) -> T:
    if not is_dataclass(cls):
        raise TypeError(f"Expected dataclass type, got {cls}")
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object for {cls.__name__}")
    allowed = {item.name for item in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} fields: {sorted(unknown)}")
    normalized = dict(value)
    hints = get_type_hints(cls)
    for item in fields(cls):
        if item.name not in normalized:
            continue
        annotation = hints.get(item.name)
        if get_origin(annotation) is tuple and isinstance(normalized[item.name], list):
            normalized[item.name] = tuple(normalized[item.name])
    return cls(**normalized)


class ConfigLoader:
    def load(self, path: str | Path) -> ExperimentConfig:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        config = self.from_dict(data)
        config.validate()
        return config

    def load_sections(self, path: str | Path, sections: tuple[str, ...] | list[str]) -> ExperimentConfig:
        """Load only named top-level sections from a JSON config.

        This is useful for small notebook entrypoints that need a dataset/model
        subset without carrying unrelated run settings into the active config.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Experiment config must be a JSON object")
        missing = set(sections) - set(data)
        if missing:
            raise KeyError(f"Config sections not found in {path}: {sorted(missing)}")
        config = self.from_dict({key: data[key] for key in sections})
        config.validate()
        return config

    def load_benchmark_configs(
        self,
        dataset_config_path: str | Path,
        model_config_paths: Iterable[str | Path],
        *,
        output_dir: str | Path,
        split: str = "test",
        resume: bool = True,
        flush_every: int = 10,
    ) -> ExperimentConfig:
        """Combine one dataset config and separate model configs for benchmarking."""
        dataset_data = json.loads(Path(dataset_config_path).read_text(encoding="utf-8"))
        if not isinstance(dataset_data, dict) or "dataset" not in dataset_data:
            raise ValueError(f"Dataset config must contain a dataset section: {dataset_config_path}")
        dataset_config = self.from_dict({
            key: dataset_data[key]
            for key in ("dataset", "platform", "decode")
            if key in dataset_data
        })
        dataset_config.validate()
        model_specs = []
        for path in model_config_paths:
            model_config = self.load_sections(path, ("model",))
            if model_config.model is None:
                raise ValueError(f"Model config has no model section: {path}")
            model_specs.append(model_config.model)
        if not model_specs:
            raise ValueError("Benchmark requires at least one model config path")

        config = replace(
            dataset_config,
            benchmark=BenchmarkConfig(
                models=tuple(model_specs), split=split, resume=resume,
                flush_every=flush_every,
            ),
            output_dir=str(output_dir),
            run_id=f"{dataset_config.dataset.dataset_id}_benchmark",
        )
        config.validate()
        return config

    def from_dict(self, data: dict[str, Any]) -> ExperimentConfig:
        from .types import (
            AdaLoraSettings, BenchmarkConfig, DatasetConfig, DecodeConfig,
            ModelSpec, PiSSASettings, SemanticMetricsConfig, TrainConfig,
        )

        allowed = {
            "dataset", "platform", "model", "decode", "train", "benchmark",
            "semantic_metrics", "output_dir", "run_id",
        }
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"Unknown ExperimentConfig fields: {sorted(unknown)}")

        train_data = data.get("train")
        if train_data is not None and isinstance(train_data.get("adalora"), dict):
            train_data = dict(train_data)
            train_data["adalora"] = _construct(AdaLoraSettings, train_data["adalora"])
        if train_data is not None and isinstance(train_data.get("pissa"), dict):
            train_data = dict(train_data)
            train_data["pissa"] = _construct(PiSSASettings, train_data["pissa"])
        benchmark_data = data.get("benchmark")
        if benchmark_data is not None:
            benchmark_data = dict(benchmark_data)
            benchmark_data["models"] = tuple(
                _construct(ModelSpec, model) for model in benchmark_data.get("models", ())
            )
        semantic_metrics_data = data.get("semantic_metrics", {})
        return ExperimentConfig(
            dataset=_construct(DatasetConfig, data.get("dataset", {})),
            platform=data.get("platform", "auto"),
            model=_construct(ModelSpec, data["model"]) if data.get("model") is not None else None,
            decode=_construct(DecodeConfig, data.get("decode", {})),
            train=_construct(TrainConfig, train_data) if train_data is not None else None,
            benchmark=_construct(BenchmarkConfig, benchmark_data) if benchmark_data is not None else None,
            semantic_metrics=_construct(SemanticMetricsConfig, semantic_metrics_data),
            output_dir=data.get("output_dir", "outputs/runs/default"),
            run_id=data.get("run_id", "default"),
        )

    def with_overrides(self, config: ExperimentConfig, overrides: dict[str, Any]) -> ExperimentConfig:
        data = config.to_dict()
        for key, value in overrides.items():
            if key not in data:
                raise ValueError(f"Unknown ExperimentConfig override: {key}")
            data[key] = value
        resolved = self.from_dict(data)
        resolved.validate()
        return resolved

    def save_resolved(self, config: ExperimentConfig, path: str | Path) -> Path:
        config.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return destination

