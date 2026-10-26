from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reimplement_cba.config import CBAConfig


def _ct2_runtime_settings(config: CBAConfig) -> tuple[str, str]:
    model_values = config.values["model"]
    device = str(model_values.get("device", "auto"))
    compute_type = str(model_values.get("compute_type", "float16"))
    if device == "auto":
        try:
            import ctranslate2
        except ImportError as error:
            raise RuntimeError(
                "CTranslate2 is required for faster-whisper and WhisperX inference"
            ) from error
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if device == "cpu" and compute_type in {"float16", "bfloat16"}:
        compute_type = "float32"
    return device, compute_type


class _CT2WhisperAdapter:
    """Shared CTranslate2 inference settings and transcript handling."""

    backend_name = "ct2"

    def __init__(self, config: CBAConfig, model_path: str | Path):
        self.config = config
        self.model_path = str(Path(model_path).expanduser().resolve())
        if not Path(self.model_path).is_dir():
            raise FileNotFoundError(f"CTranslate2 checkpoint directory does not exist: {self.model_path}")
        self.device, self.compute_type = _ct2_runtime_settings(config)
        audio_values = config.values["audio"]
        self.min_duration_seconds = float(audio_values["min_duration_seconds"])
        self.max_duration_seconds = float(
            audio_values.get(
                "long_form_max_duration_seconds", audio_values["max_duration_seconds"]
            )
        )
        self.model: Any | None = None
        metadata_path = Path(self.model_path) / "reimplement_cba.json"
        self.artifact_metadata = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.is_file()
            else {}
        )

    @staticmethod
    def _join_segments(segments) -> str:
        texts = []
        for segment in segments:
            text = (
                segment.get("text", "")
                if isinstance(segment, dict)
                else getattr(segment, "text", "")
            )
            if str(text).strip():
                texts.append(str(text).strip())
        return " ".join(texts)

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": self.backend_name,
            "ct2_model_path": self.model_path,
            "model_checkpoint": self.artifact_metadata.get("base_model"),
            "processor_checkpoint": self.artifact_metadata.get("processor_model"),
            "model_revision": self.artifact_metadata.get("model_revision"),
            "device": self.device,
            "compute_type": self.compute_type,
            "language": str(self.config.values["model"]["language"]),
            "adapter_path": self.artifact_metadata.get("adapter_path"),
        }

    def _validate_audio(self, audio) -> None:
        audio.validate()
        if audio.sample_rate != 16000:
            raise ValueError(
                f"{self.backend_name} requires 16000 Hz audio for {audio.sample_id!r}"
            )
        if not self.min_duration_seconds <= audio.duration_seconds <= self.max_duration_seconds:
            raise ValueError(
                f"{self.backend_name} inference audio duration must be between "
                f"{self.min_duration_seconds:g} and {self.max_duration_seconds:g} seconds; "
                f"sample {audio.sample_id!r} is {audio.duration_seconds:.3f} seconds"
            )

    def close(self) -> None:
        self.model = None


class FasterWhisperAdapter(_CT2WhisperAdapter):
    """Run a converted Whisper checkpoint through faster-whisper directly."""

    backend_name = "faster_whisper"

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            from faster_whisper import WhisperModel
        except ImportError as error:
            raise RuntimeError(
                "faster-whisper is required for this backend; install the inference extra"
            ) from error
        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, audio) -> str:
        if self.model is None:
            raise RuntimeError("Call FasterWhisperAdapter.load() before transcribe()")
        self._validate_audio(audio)
        model_values = self.config.values["model"]
        segments, _ = self.model.transcribe(
            audio.waveform,
            language=str(model_values["language"]),
            task=str(model_values.get("task", "transcribe")),
            beam_size=int(model_values.get("num_beams", 1)),
            condition_on_previous_text=False,
            vad_filter=False,
        )
        return self._join_segments(segments)


class WhisperXAdapter(_CT2WhisperAdapter):
    """Run WhisperX ASR with its VAD/cut-and-merge stage, without alignment."""

    backend_name = "whisperx"

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import whisperx
        except ImportError as error:
            raise RuntimeError(
                "WhisperX is required for this backend; install the whisperx extra"
            ) from error
        model_values = self.config.values["model"]
        inference_values = self.config.values.get("inference", {})
        vad_values = inference_values.get("vad", {})
        self.model = whisperx.load_model(
            self.model_path,
            self.device,
            compute_type=self.compute_type,
            language=str(model_values["language"]),
            task=str(model_values.get("task", "transcribe")),
            vad_method=str(vad_values.get("method", "pyannote")),
            vad_options={
                "chunk_size": int(vad_values.get("chunk_size_seconds", 30)),
                "vad_onset": float(vad_values.get("onset", 0.3)),
                "vad_offset": float(vad_values.get("offset", 0.3)),
            },
        )

    def transcribe(self, audio) -> str:
        if self.model is None:
            raise RuntimeError("Call WhisperXAdapter.load() before transcribe()")
        self._validate_audio(audio)
        result = self.model.transcribe(
            audio.waveform,
            batch_size=int(self.config.values.get("inference", {}).get("batch_size", 8)),
        )
        return self._join_segments(result.get("segments", []))


def create_transcriber(
    config: CBAConfig,
    *,
    adapter_path: str | Path | None = None,
    ct2_model_path: str | Path | None = None,
):
    """Build the configured inference backend for baseline or adapted runs."""
    inference_values = config.values.get("inference", {})
    backend = str(inference_values.get("backend", "transformers"))
    if backend == "transformers":
        from reimplement_cba.adapters.whisper import PhoWhisperAdapter

        return PhoWhisperAdapter(config, adapter_path=adapter_path)

    model_path = ct2_model_path or inference_values.get("ct2_model_path")
    if not model_path:
        raise ValueError(
            f"A CTranslate2 checkpoint path is required for inference.backend='{backend}'"
        )
    resolved_path = config.resolve_model_reference(model_path)
    if not Path(resolved_path).is_dir():
        raise FileNotFoundError(f"CTranslate2 checkpoint directory does not exist: {resolved_path}")
    metadata_path = Path(resolved_path) / "reimplement_cba.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_base_model = config.resolve_model_reference(
            config.values["model"]["base_model"]
        )
        artifact_base_model = metadata.get("base_model")
        if artifact_base_model is not None and artifact_base_model != expected_base_model:
            raise ValueError(
                "CTranslate2 artifact base checkpoint does not match the configured model; "
                f"artifact={artifact_base_model!r}, configured={expected_base_model!r}"
            )
        expected_revision = config.values["model"].get("revision")
        artifact_revision = metadata.get("model_revision")
        if artifact_revision != expected_revision:
            raise ValueError(
                "CTranslate2 artifact model revision does not match the configured revision; "
                f"artifact={artifact_revision!r}, configured={expected_revision!r}"
            )
        artifact_adapter = metadata.get("adapter_path")
        requested_adapter = (
            config.resolve_model_reference(adapter_path) if adapter_path is not None else None
        )
        if artifact_adapter != requested_adapter:
            raise ValueError(
                "CTranslate2 artifact adapter provenance does not match the requested run; "
                f"artifact={artifact_adapter!r}, requested={requested_adapter!r}"
            )
    if backend == "faster_whisper":
        return FasterWhisperAdapter(config, resolved_path)
    if backend == "whisperx":
        return WhisperXAdapter(config, resolved_path)
    raise ValueError(f"Unsupported inference backend: {backend}")
