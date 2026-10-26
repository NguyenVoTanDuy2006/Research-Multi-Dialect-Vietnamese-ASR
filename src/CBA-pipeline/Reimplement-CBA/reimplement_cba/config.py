from __future__ import annotations

import json
import math
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

        initial_rule = self.values["initial_rule"]
        markers = initial_rule.get("reject_transcript_markers", [])
        if not isinstance(markers, list) or any(
            not isinstance(marker, str) or not marker for marker in markers
        ):
            raise ValueError("initial_rule.reject_transcript_markers must be a list of nonempty strings")
        manifest_path = self.english_exclusion_manifest_path
        if manifest_path is not None:
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"English exclusion manifest does not exist: {manifest_path}"
                )
            self.english_exclusion_sample_ids

        model = self.values["model"]
        base_model = model.get("base_model")
        if not isinstance(base_model, str) or not base_model.strip():
            raise ValueError("model.base_model is required")
        processor_model = model.get("processor_model")
        if processor_model is not None and (
            not isinstance(processor_model, str) or not processor_model.strip()
        ):
            raise ValueError("model.processor_model must be null or a nonempty model ID/path")
        revision = model.get("revision")
        if revision is not None and (not isinstance(revision, str) or not revision.strip()):
            raise ValueError("model.revision must be null or a nonempty revision")
        if model.get("task") != "transcribe":
            raise ValueError("This scaffold currently supports task='transcribe' only")
        language = model.get("language")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("model.language is required")
        if int(model.get("max_new_tokens", 128)) <= 0:
            raise ValueError("model.max_new_tokens must be positive")
        if int(model.get("num_beams", 1)) <= 0:
            raise ValueError("model.num_beams must be positive")
        device = model.get("device", "auto")
        if not isinstance(device, str) or device not in {"auto", "cpu", "cuda"}:
            raise ValueError("model.device must be 'auto', 'cpu', or 'cuda'")
        compute_type = model.get("compute_type", "float16")
        if not isinstance(compute_type, str) or compute_type not in {
            "float16", "bfloat16", "float32"
        }:
            raise ValueError("model.compute_type must be float16, bfloat16, or float32")

        inference = self.values.get("inference", {})
        if not isinstance(inference, dict):
            raise ValueError("Config section 'inference' must be an object")
        backend = inference.get("backend", "transformers")
        if backend not in {"transformers", "faster_whisper", "whisperx"}:
            raise ValueError(
                "inference.backend must be 'transformers', 'faster_whisper', or 'whisperx'"
            )
        ct2_model_path = inference.get("ct2_model_path")
        if ct2_model_path is not None and (
            not isinstance(ct2_model_path, str) or not ct2_model_path.strip()
        ):
            raise ValueError("inference.ct2_model_path must be null or a nonempty path")
        batch_size = int(inference.get("batch_size", 8))
        if batch_size < 1:
            raise ValueError("inference.batch_size must be positive")
        vad = inference.get("vad", {})
        if not isinstance(vad, dict):
            raise ValueError("inference.vad must be an object")
        if vad.get("method", "pyannote") not in {"pyannote", "silero"}:
            raise ValueError("inference.vad.method must be 'pyannote' or 'silero'")
        chunk_size = float(vad.get("chunk_size_seconds", 30))
        onset = float(vad.get("onset", 0.3))
        offset = float(vad.get("offset", 0.3))
        if (
            not math.isfinite(chunk_size)
            or not 0 < chunk_size <= 30
            or not math.isfinite(onset)
            or not 0 <= onset <= 1
            or not math.isfinite(offset)
            or not 0 <= offset <= 1
        ):
            raise ValueError("inference VAD chunk size/onset/offset are invalid")

        audio = self.values["audio"]
        sample_rate = int(audio.get("sample_rate", 0))
        min_duration = float(audio.get("min_duration_seconds", 0))
        max_duration = float(audio.get("max_duration_seconds", 0))
        long_form_max_duration = float(
            audio.get("long_form_max_duration_seconds", max_duration)
        )
        if (
            sample_rate != 16000
            or not all(
                math.isfinite(value)
                for value in (min_duration, max_duration, long_form_max_duration)
            )
            or min_duration < 0.5
            or max_duration <= min_duration
            or max_duration > 30
            or long_form_max_duration < max_duration
        ):
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
        if float(training.get("epochs", 0)) <= 0:
            raise ValueError("training.epochs must be positive")
        if int(training.get("batch_size", 0)) <= 0:
            raise ValueError("training.batch_size must be positive")
        if float(training.get("learning_rate", 0)) <= 0:
            raise ValueError("training.learning_rate must be positive")
        if int(training.get("warmup_steps", 0)) < 0:
            raise ValueError("training.warmup_steps cannot be negative")
        if int(training.get("init_r", 0)) <= 0 or int(training.get("target_r", 0)) <= 0:
            raise ValueError("AdaLoRA ranks must be positive")
        if int(training["target_r"]) > int(training["init_r"]):
            raise ValueError("AdaLoRA target_r cannot exceed init_r")
        if int(training.get("gradient_accumulation_steps", 1)) < 1:
            raise ValueError("training.gradient_accumulation_steps must be positive")
        target_modules = training.get("target_modules")
        if not isinstance(target_modules, list) or not target_modules or any(
            not isinstance(module, str) or not module.strip() for module in target_modules
        ):
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
    def english_exclusion_manifest_path(self) -> Path | None:
        raw_value = self.values.get("initial_rule", {}).get("english_sample_ids_manifest")
        if raw_value is None:
            return None
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(
                "initial_rule.english_sample_ids_manifest must be null or a nonempty path"
            )
        raw_path = Path(raw_value).expanduser()
        return raw_path.resolve() if raw_path.is_absolute() else (self.path.parent / raw_path).resolve()

    @property
    def english_exclusion_sample_ids(self) -> tuple[str, ...]:
        manifest_path = self.english_exclusion_manifest_path
        if manifest_path is None:
            return ()
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid English exclusion manifest JSON: {manifest_path}") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("sample_ids"), list):
            raise ValueError(
                "English exclusion manifest must be an object containing a 'sample_ids' list"
            )
        sample_ids = payload["sample_ids"]
        if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in sample_ids):
            raise ValueError("English exclusion manifest sample_ids must be nonempty strings")
        normalized = tuple(sample_id.strip() for sample_id in sample_ids)
        if len(normalized) != len(set(normalized)):
            raise ValueError("English exclusion manifest contains duplicate sample IDs")
        return normalized

    @property
    def output_dir(self) -> Path:
        raw_path = Path(str(self.values.get("output_dir", "outputs/cba_run"))).expanduser()
        return raw_path.resolve() if raw_path.is_absolute() else (self.path.parent / raw_path).resolve()

    def resolve_model_reference(self, reference: str | Path) -> str:
        """Resolve explicit local paths while preserving Hugging Face repo IDs."""
        raw = str(reference).strip()
        path = Path(raw).expanduser()
        if path.is_absolute():
            return str(path.resolve())
        if raw.startswith(("./", "../", "~")):
            return str((self.path.parent / path).resolve())
        config_relative = self.path.parent / path
        if config_relative.exists():
            return str(config_relative.resolve())
        return raw

    @property
    def run_id(self) -> str:
        return str(self.values.get("run_id") or self.output_dir.name)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "dataset_config": str(self.dataset_config_path),
            "dataset_root_override": self.values["dataset"].get("root_override"),
            "english_exclusion_manifest": (
                str(self.english_exclusion_manifest_path)
                if self.english_exclusion_manifest_path is not None
                else None
            ),
            "approved_english_sample_ids": len(self.english_exclusion_sample_ids),
            "base_model": self.values["model"]["base_model"],
            "processor_model": self.values["model"].get("processor_model")
            or self.values["model"]["base_model"],
            "model_revision": self.values["model"].get("revision"),
            "language": self.values["model"]["language"],
            "task": self.values["model"]["task"],
            "device": self.values["model"].get("device", "auto"),
            "compute_type": self.values["model"].get("compute_type", "float16"),
            "inference_backend": self.values.get("inference", {}).get(
                "backend", "transformers"
            ),
            "ct2_model_path": self.values.get("inference", {}).get("ct2_model_path"),
            "training_max_duration_seconds": float(
                self.values["audio"]["max_duration_seconds"]
            ),
            "long_form_max_duration_seconds": float(
                self.values["audio"].get(
                    "long_form_max_duration_seconds",
                    self.values["audio"]["max_duration_seconds"],
                )
            ),
            "curriculum_rounds": int(self.values["curriculum"]["max_rounds"]),
            "curriculum_domain": self.values["curriculum"]["domain"],
            "curriculum_threshold": float(self.values["curriculum"]["threshold"]),
            "output_dir": str(self.output_dir),
        }
