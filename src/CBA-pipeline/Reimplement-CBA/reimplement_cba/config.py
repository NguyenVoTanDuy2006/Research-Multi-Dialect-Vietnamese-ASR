from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = (
    "dataset", "model", "audio", "initial_rule", "training",
    "curriculum", "checkpoint_selection",
)


@dataclass(frozen=True)
class CBAConfig:
    """Validated JSON run config with paths resolved relative to its file."""

    path: Path
    values: dict[str, Any]

    @classmethod
    def from_file(cls, path: str | Path) -> "CBAConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"CBA config does not exist: {config_path}")
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("CBA config must be a JSON object")
        missing = [section for section in REQUIRED_SECTIONS if section not in payload]
        if missing:
            raise ValueError(f"CBA config is missing sections: {', '.join(missing)}")
        config = cls(config_path, payload)
        config.validate()
        return config

    def validate(self) -> None:
        for section in REQUIRED_SECTIONS:
            if not isinstance(self.values[section], dict):
                raise ValueError(f"Config section '{section}' must be an object")

        dataset_config = self.dataset_config_path
        if not dataset_config.is_file():
            raise FileNotFoundError(f"ViMD dataset config does not exist: {dataset_config}")
        dataset_payload = json.loads(dataset_config.read_text(encoding="utf-8"))
        if not isinstance(dataset_payload, dict) or not isinstance(dataset_payload.get("dataset"), dict):
            raise ValueError(f"ViMD dataset config must contain a 'dataset' object: {dataset_config}")

        model = self.values["model"]
        if not str(model.get("base_model", "")).strip():
            raise ValueError("model.base_model is required")
        if model.get("task") != "transcribe":
            raise ValueError("This scaffold currently supports task='transcribe' only")
        if not str(model.get("language", "")).strip():
            raise ValueError("model.language is required")

        audio = self.values["audio"]
        sample_rate = int(audio.get("sample_rate", 0))
        min_duration = float(audio.get("min_duration_seconds", 0))
        max_duration = float(audio.get("max_duration_seconds", 0))
        if sample_rate <= 0 or min_duration < 0 or max_duration <= min_duration:
            raise ValueError("Audio sample rate/duration limits are invalid")

        curriculum = self.values["curriculum"]
        if int(curriculum.get("max_rounds", 0)) < 1:
            raise ValueError("curriculum.max_rounds must be at least 1")
        if curriculum.get("domain") not in {"in_domain", "out_of_domain"}:
            raise ValueError("curriculum.domain must be 'in_domain' or 'out_of_domain'")
        threshold = float(curriculum.get("threshold", -1))
        if not 0 <= threshold <= 100:
            raise ValueError("curriculum.threshold must be between 0 and 100")
        if curriculum.get("round_initialization") not in {
            "resume_previous_adapter", "restart_from_base"
        }:
            raise ValueError("Unsupported curriculum.round_initialization")

        training = self.values["training"]
        if training.get("method") != "adalora":
            raise ValueError("This scaffold currently supports method='adalora' only")
        if int(training.get("init_r", 0)) <= 0 or int(training.get("target_r", 0)) <= 0:
            raise ValueError("AdaLoRA ranks must be positive")
        if int(training["target_r"]) > int(training["init_r"]):
            raise ValueError("AdaLoRA target_r cannot exceed init_r")
        if not training.get("target_modules"):
            raise ValueError("training.target_modules cannot be empty")

        selection = self.values["checkpoint_selection"]
        if (selection.get("primary_metric"), selection.get("tie_break_metric")) != (
            "semscore", "wer"
        ):
            raise ValueError("Checkpoint policy must rank by SemScore, then WER")
        if (selection.get("primary_direction"), selection.get("tie_break_direction")) != (
            "max", "min"
        ):
            raise ValueError("Checkpoint policy directions must be SemScore=max, WER=min")

    @property
    def dataset_config_path(self) -> Path:
        raw_path = Path(str(self.values["dataset"].get("config_path", ""))).expanduser()
        return raw_path.resolve() if raw_path.is_absolute() else (self.path.parent / raw_path).resolve()

    @property
    def output_dir(self) -> Path:
        raw_path = Path(str(self.values.get("output_dir", "outputs/cba_run"))).expanduser()
        return raw_path.resolve() if raw_path.is_absolute() else (self.path.parent / raw_path).resolve()

    @property
    def run_id(self) -> str:
        return str(self.values.get("run_id") or self.output_dir.name)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_config": str(self.dataset_config_path),
            "dataset_root_override": self.values["dataset"].get("root_override"),
            "base_model": self.values["model"]["base_model"],
            "language": self.values["model"]["language"],
            "task": self.values["model"]["task"],
            "curriculum_rounds": int(self.values["curriculum"]["max_rounds"]),
            "curriculum_domain": self.values["curriculum"]["domain"],
            "curriculum_threshold": float(self.values["curriculum"]["threshold"]),
            "output_dir": str(self.output_dir),
        }
