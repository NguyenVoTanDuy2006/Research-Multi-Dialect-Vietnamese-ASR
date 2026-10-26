from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from reimplement_cba.config import CBAConfig


def prepare_ct2_checkpoint(
    config: CBAConfig,
    output_dir: str | Path,
    *,
    adapter_path: str | Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    """Merge an optional PEFT adapter and convert the Whisper checkpoint to CT2."""
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"CTranslate2 output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    converter = shutil.which("ct2-transformers-converter")
    if converter is None:
        raise RuntimeError(
            "ct2-transformers-converter is unavailable; install the inference extra "
            "and Transformers conversion dependencies"
        )

    model_values = config.values["model"]
    base_model = config.resolve_model_reference(model_values["base_model"])
    processor_reference = model_values.get("processor_model") or model_values["base_model"]
    processor_checkpoint = config.resolve_model_reference(processor_reference)
    resolved_adapter = None
    if adapter_path is not None:
        resolved_adapter = config.resolve_model_reference(adapter_path)
        if not Path(resolved_adapter).is_dir():
            raise FileNotFoundError(f"AdaLoRA adapter directory does not exist: {resolved_adapter}")

    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError as error:
        raise RuntimeError(
            "PyTorch and Transformers are required to prepare a CTranslate2 checkpoint"
        ) from error

    with tempfile.TemporaryDirectory(
        prefix=".ct2-prepare-", dir=str(destination.parent)
    ) as temporary_root:
        temporary_root = Path(temporary_root)
        transformers_dir = temporary_root / "transformers"
        converted_dir = temporary_root / "converted"
        transformers_dir.mkdir()

        revision = model_values.get("revision")
        revision_kwargs = {"revision": revision} if revision else {}
        processor = WhisperProcessor.from_pretrained(
            processor_checkpoint,
            language=str(model_values["language"]),
            task=str(model_values.get("task", "transcribe")),
            **revision_kwargs,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            base_model,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )
        if resolved_adapter is not None:
            try:
                from peft import PeftModel
            except ImportError as error:
                raise RuntimeError("PEFT is required to merge the AdaLoRA adapter") from error
            model = PeftModel.from_pretrained(
                model,
                resolved_adapter,
                is_trainable=False,
            )
            try:
                model = model.merge_and_unload(safe_merge=True)
            except (AttributeError, NotImplementedError) as error:
                raise RuntimeError(
                    "This PEFT adapter does not support merging into a standalone Whisper model"
                ) from error

        model.save_pretrained(transformers_dir, safe_serialization=True)
        processor.save_pretrained(transformers_dir)
        required_processor_files = ("tokenizer.json", "preprocessor_config.json")
        missing_files = [
            filename for filename in required_processor_files
            if not (transformers_dir / filename).is_file()
        ]
        if missing_files:
            raise RuntimeError(
                "Saved Whisper processor is missing files required by CTranslate2: "
                + ", ".join(missing_files)
            )

        command = [
            converter,
            "--model", str(transformers_dir),
            "--output_dir", str(converted_dir),
            "--copy_files", "tokenizer.json", "preprocessor_config.json",
            "--quantization", "float16",
        ]
        try:
            runner(command, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or error.stdout or str(error)).strip()
            raise RuntimeError(f"CTranslate2 conversion failed: {detail}") from error
        if not (converted_dir / "model.bin").is_file():
            raise RuntimeError("CTranslate2 converter completed without creating model.bin")

        metadata = {
            "base_model": base_model,
            "processor_model": processor_checkpoint,
            "model_revision": revision,
            "adapter_path": resolved_adapter,
            "backend_format": "ctranslate2",
            "quantization": "float16",
            "language": str(model_values["language"]),
            "task": str(model_values.get("task", "transcribe")),
        }
        (converted_dir / "reimplement_cba.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        converted_dir.rename(destination)

        del model, processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return destination
