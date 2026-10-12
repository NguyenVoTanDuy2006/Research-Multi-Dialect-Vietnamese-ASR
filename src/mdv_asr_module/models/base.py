from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Any

from mdv_asr_module.configs.types import DecodeConfig, ModelSpec
from mdv_asr_module.data.records import AudioSample


@dataclass
class ModelBundle:
    model: Any
    processor: Any = None
    tokenizer: Any = None
    revision: str | None = None
    device: str | None = None
    dtype: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    elapsed_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ASRBackend(ABC):
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.bundle: ModelBundle | None = None

    @abstractmethod
    def load(self) -> ModelBundle:
        raise NotImplementedError

    @abstractmethod
    def transcribe(self, audio: AudioSample, decode: DecodeConfig) -> TranscriptionResult:
        raise NotImplementedError

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        self.bundle = None

    def _require_bundle(self) -> ModelBundle:
        if self.bundle is None:
            raise RuntimeError(f"{type(self).__name__}.load() must be called before transcribe()")
        return self.bundle


class TransformersBackendBase(ASRBackend):
    def move_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        import torch

        bundle = self._require_bundle()
        parameter = next(bundle.model.parameters())
        moved = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device=bundle.device,
                                      dtype=parameter.dtype if value.is_floating_point() else value.dtype)
            else:
                moved[key] = value
        return moved

    def generate(self, inputs: dict[str, Any], **kwargs):
        import torch

        bundle = self._require_bundle()
        start = time.perf_counter()
        with torch.inference_mode():
            output = bundle.model.generate(**inputs, **kwargs)
        return output, time.perf_counter() - start

    def metadata(self):
        bundle = self._require_bundle()
        return {"key": self.spec.key, "model_id": self.spec.model_id,
                "revision": bundle.revision, "backend": self.spec.backend}


def extract_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return extract_text(value[0]) if value else ""
    if isinstance(value, dict):
        for key in ("text", "transcription", "pred_text"):
            if key in value:
                return extract_text(value[key])
        return ""
    if hasattr(value, "text"):
        return extract_text(value.text)
    if hasattr(value, "pred_text"):
        return extract_text(value.pred_text)
    return str(value).strip()

