from __future__ import annotations

import importlib

from mdv_asr.config.types import ModelSpec


class ASRBackendFactory:
    _constructors = {
        "qwen3_asr": ("mdv_asr.models.qwen", "QwenASRBackend"),
        "ctc": ("mdv_asr.models.ctc", "CTCASRBackend"),
        "whisper": ("mdv_asr.models.whisper", "WhisperASRBackend"),
        "ark": ("mdv_asr.models.ark", "ArkASRBackend"),
        "cohere": ("mdv_asr.models.cohere", "CohereASRBackend"),
        "granite": ("mdv_asr.models.granite", "GraniteASRBackend"),
        "nemo_salm": ("mdv_asr.models.nemo", "CanaryASRBackend"),
        "nemo_asr": ("mdv_asr.models.nemo", "ParakeetASRBackend"),
    }

    @classmethod
    def supported_backends(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._constructors))

    @classmethod
    def register(cls, name: str, constructor) -> None:
        if not name or name in cls._constructors:
            raise ValueError(f"Backend name already registered or empty: {name!r}")
        if isinstance(constructor, tuple):
            cls._constructors[name] = constructor
        else:
            cls._constructors[name] = constructor

    @classmethod
    def validate_spec(cls, spec: ModelSpec) -> None:
        spec.validate()
        if spec.backend not in cls._constructors:
            raise ValueError(f"Unsupported backend {spec.backend!r}; registered: {cls.supported_backends()}")

    @classmethod
    def create(cls, spec: ModelSpec, **kwargs):
        cls.validate_spec(spec)
        constructor = cls._constructors[spec.backend]
        if isinstance(constructor, tuple):
            module = importlib.import_module(constructor[0])
            constructor = getattr(module, constructor[1])
        return constructor(spec, **kwargs)

