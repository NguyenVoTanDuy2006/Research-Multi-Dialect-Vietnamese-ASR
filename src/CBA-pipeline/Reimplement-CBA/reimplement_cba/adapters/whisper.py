from __future__ import annotations

from reimplement_cba.config import CBAConfig


class PhoWhisperAdapter:
    """Model interface; load/generation are deliberately left for implementation."""

    def __init__(self, config: CBAConfig, *, adapter_path: str | None = None):
        self.config = config
        self.adapter_path = adapter_path

    def load(self) -> None:
        raise NotImplementedError(
            "TODO: load PhoWhisper with Transformers and optionally attach the AdaLoRA adapter"
        )

    def transcribe(self, audio) -> str:
        raise NotImplementedError("TODO: implement Vietnamese Whisper transcription")
