from __future__ import annotations

from pathlib import Path
from typing import Any

from mdv_asr_module.data.records import AudioSample

from reimplement_cba.config import CBAConfig


class PhoWhisperAdapter:
    """Load any configured Hugging Face Whisper-compatible checkpoint.

    The current config defaults to PhoWhisper-medium. A PEFT adapter is an
    optional, separate checkpoint layered on top of ``model.base_model``.
    Heavy model libraries are imported only when :meth:`load` is called.
    """

    def __init__(self, config: CBAConfig, *, adapter_path: str | Path | None = None):
        self.config = config
        self.adapter_path = (
            config.resolve_model_reference(adapter_path) if adapter_path is not None else None
        )
        model_config = config.values["model"]
        self.model_checkpoint = config.resolve_model_reference(model_config["base_model"])
        processor_reference = model_config.get("processor_model") or model_config["base_model"]
        self.processor_checkpoint = config.resolve_model_reference(processor_reference)
        self.revision = model_config.get("revision")
        self.language = str(model_config["language"])
        self.task = str(model_config.get("task", "transcribe"))
        self.max_new_tokens = int(model_config.get("max_new_tokens", 128))
        self.num_beams = int(model_config.get("num_beams", 1))
        self.requested_device = str(model_config.get("device", "auto"))
        self.requested_compute_type = str(model_config.get("compute_type", "float16"))
        audio_config = config.values["audio"]
        self.min_duration_seconds = float(audio_config["min_duration_seconds"])
        self.max_duration_seconds = float(
            audio_config.get(
                "long_form_max_duration_seconds",
                audio_config["max_duration_seconds"],
            )
        )

        self.model = None
        self.processor = None
        self.device: str | None = None
        self.compute_type: str | None = None
        self.model_revision: str | None = None

    def _runtime_settings(self, torch) -> tuple[str, Any, str]:
        if self.requested_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        elif self.requested_device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("model.device is 'cuda' but CUDA is not available")
            device = "cuda"
        else:
            device = "cpu"

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        compute_type = self.requested_compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "float32"
        return device, dtype_map[compute_type], compute_type

    def load(self) -> None:
        """Load configured model and processor, optionally attaching a PEFT adapter."""
        if self.model is not None and self.processor is not None:
            return

        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        device, dtype, resolved_compute_type = self._runtime_settings(torch)
        revision_kwargs = {"revision": self.revision} if self.revision else {}
        processor = WhisperProcessor.from_pretrained(
            self.processor_checkpoint,
            language=self.language,
            task=self.task,
            **revision_kwargs,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            self.model_checkpoint,
            dtype=dtype,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        )
        model.to(device)

        if self.adapter_path is not None:
            try:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model,
                    self.adapter_path,
                    is_trainable=False,
                )
                model.to(device)
            except Exception as error:
                raise RuntimeError(
                    f"Could not load PEFT adapter '{self.adapter_path}' on base checkpoint "
                    f"'{self.model_checkpoint}': {error}"
                ) from error

        model.eval()
        self.model = model
        self.processor = processor
        self.device = device
        self.compute_type = resolved_compute_type
        self.model_revision = getattr(getattr(model, "config", None), "_commit_hash", None)

    def transcribe(self, audio: AudioSample) -> str:
        """Transcribe one mono 16 kHz sample and preserve the decoded text."""
        if self.model is None or self.processor is None or self.device is None:
            raise RuntimeError("Call PhoWhisperAdapter.load() before transcribe()")
        audio.validate()
        if audio.sample_rate != 16000:
            raise ValueError(
                f"PhoWhisper expects 16000 Hz audio; received {audio.sample_rate} Hz "
                f"for sample {audio.sample_id!r}"
            )
        if not self.min_duration_seconds <= audio.duration_seconds <= self.max_duration_seconds:
            raise ValueError(
                f"PhoWhisper inference audio duration must be between {self.min_duration_seconds:g} "
                f"and {self.max_duration_seconds:g} seconds; sample {audio.sample_id!r} is "
                f"{audio.duration_seconds:.3f} seconds"
            )

        import torch

        is_long_form = audio.duration_seconds > 30.0
        processor_kwargs = {
            "sampling_rate": audio.sample_rate,
            "return_tensors": "pt",
        }
        if is_long_form:
            # Whisper's long-form generate loop needs the complete feature
            # sequence and an attention mask to step through timestamped chunks.
            processor_kwargs.update(
                {
                    "truncation": False,
                    "padding": "longest",
                    "return_attention_mask": True,
                }
            )
        encoded = self.processor(audio.waveform, **processor_kwargs)
        input_features = encoded["input_features"].to(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device=self.device)
        generation_kwargs = {
            "input_features": input_features,
            "language": self.language,
            "task": self.task,
            "num_beams": self.num_beams,
        }
        if attention_mask is not None:
            generation_kwargs["attention_mask"] = attention_mask
        if is_long_form:
            generation_kwargs["return_timestamps"] = True
        else:
            generation_kwargs["max_new_tokens"] = self.max_new_tokens

        # Whisper checkpoints may ship a legacy generation_config.max_length
        # (e.g. 448). It conflicts with max_new_tokens even though the latter
        # takes precedence. Clear that stored cap only for short-form calls;
        # long-form Whisper generation keeps the checkpoint's original value.
        generation_config = getattr(self.model, "generation_config", None)
        restore_max_length = not is_long_form and generation_config is not None
        original_max_length = (
            generation_config.max_length if restore_max_length else None
        )
        if restore_max_length:
            generation_config.max_length = None
        try:
            with torch.inference_mode():
                token_ids = self.model.generate(**generation_kwargs)
        finally:
            if restore_max_length:
                generation_config.max_length = original_max_length
        decoded = self.processor.batch_decode(token_ids, skip_special_tokens=True)
        return decoded[0].strip() if decoded else ""

    def metadata(self) -> dict[str, Any]:
        """Return the selected checkpoint and resolved runtime settings."""
        return {
            "backend": "transformers",
            "ct2_model_path": None,
            "model_checkpoint": self.model_checkpoint,
            "processor_checkpoint": self.processor_checkpoint,
            "model_revision": self.model_revision or self.revision,
            "adapter_path": self.adapter_path,
            "language": self.language,
            "task": self.task,
            "device": self.device or self.requested_device,
            "compute_type": self.compute_type or self.requested_compute_type,
            "long_form_max_duration_seconds": self.max_duration_seconds,
            "loaded": self.model is not None,
        }

    def close(self) -> None:
        """Drop model references so callers can release device memory."""
        self.model = None
        self.processor = None
        self.device = None
        self.compute_type = None
        self.model_revision = None
