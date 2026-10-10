from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from mdv_asr_module.config.types import DecodeConfig, ModelSpec
from mdv_asr_module.data.records import AudioSample

from .base import ASRBackend, ModelBundle


class QwenModelLoader:
    def __init__(self, token: str | None = None):
        self.token = token

    def load_processor(self, spec: ModelSpec):
        from transformers import AutoProcessor

        kwargs = {"token": self.token}
        if spec.revision:
            kwargs["revision"] = spec.revision
        return AutoProcessor.from_pretrained(spec.model_id, **kwargs)

    def load_base(self, spec: ModelSpec, device: str | None = None,
                  dtype: str | None = None) -> ModelBundle:
        import torch
        from transformers import Qwen3ASRForConditionalGeneration

        processor = self.load_processor(spec)
        selected_dtype = _torch_dtype(dtype or spec.dtype)
        kwargs = {"token": self.token, "dtype": selected_dtype, "low_cpu_mem_usage": True}
        if spec.revision:
            kwargs["revision"] = spec.revision
        model = Qwen3ASRForConditionalGeneration.from_pretrained(spec.model_id, **kwargs)
        target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)
        model.eval()
        revision = getattr(model.config, "_commit_hash", None) or spec.revision
        return ModelBundle(model=model, processor=processor, revision=revision,
                           device=str(target_device), dtype=str(selected_dtype),
                           metadata={"model_id": spec.model_id, "backend": "qwen3_asr"})

    def load_adapter(self, bundle: ModelBundle, adapter_path: str | Path,
                     *, is_trainable: bool = False) -> ModelBundle:
        from peft import PeftModel

        model = PeftModel.from_pretrained(bundle.model, str(adapter_path), is_trainable=is_trainable)
        model.to(bundle.device)
        model.eval() if not is_trainable else model.train()
        bundle.model = model
        bundle.metadata["adapter_path"] = str(Path(adapter_path).resolve())
        return bundle

    @staticmethod
    def resolve_revision(spec: ModelSpec, checkpoint_metadata: dict[str, Any] | None = None) -> str | None:
        recorded = (checkpoint_metadata or {}).get("model_revision")
        if spec.revision and recorded and spec.revision != recorded:
            raise ValueError(f"Configured revision {spec.revision} differs from checkpoint base {recorded}")
        return spec.revision or recorded

    @staticmethod
    def release(bundle: ModelBundle) -> None:
        import gc
        import torch

        bundle.model = None
        bundle.processor = None
        bundle.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QwenASRBackend(ASRBackend):
    def __init__(self, spec: ModelSpec, loader: QwenModelLoader | None = None,
                 *, owns_bundle: bool = True):
        super().__init__(spec)
        if loader is None:
            from mdv_asr_module.runtime.environment import RuntimeEnvironment

            loader = QwenModelLoader(token=RuntimeEnvironment.get_hf_token())
        self.loader = loader
        self.owns_bundle = owns_bundle

    def load(self) -> ModelBundle:
        self.bundle = self.loader.load_base(self.spec)
        if self.spec.adapter_path:
            self.loader.load_adapter(self.bundle, self.spec.adapter_path)
        return self.bundle

    @classmethod
    def from_loaded(cls, spec: ModelSpec, bundle: ModelBundle) -> "QwenASRBackend":
        backend = cls(spec, owns_bundle=False)
        backend.bundle = bundle
        return backend

    def prepare_request(self, audio: AudioSample, decode: DecodeConfig):
        bundle = self._require_bundle()
        return bundle.processor.apply_transcription_request(
            audio=audio.waveform, language=decode.language
        )

    def transcribe(self, audio: AudioSample, decode: DecodeConfig):
        import torch

        bundle = self._require_bundle()
        inputs = self.prepare_request(audio, decode)
        moved = {}
        parameter_dtype = next(bundle.model.parameters()).dtype
        for key, value in inputs.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device=bundle.device,
                                      dtype=parameter_dtype if value.is_floating_point() else value.dtype)
            else:
                moved[key] = value
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = bundle.model.generate(
                **moved, max_new_tokens=decode.max_new_tokens,
                do_sample=decode.do_sample, num_beams=decode.num_beams,
            )
        generated = self.decode_generated(output_ids, moved)
        try:
            text = bundle.processor.decode(generated[0], return_format="transcription_only")
            if isinstance(text, (list, tuple)):
                text = text[0]
            if isinstance(text, dict):
                text = text.get("text", text.get("transcription", ""))
        except Exception:
            text = bundle.processor.batch_decode(generated, skip_special_tokens=True)[0]
        return __import__("mdv_asr_module.models.base", fromlist=["TranscriptionResult"]).TranscriptionResult(
            str(text).strip(), time.perf_counter() - start,
            {"model_key": self.spec.key, "revision": bundle.revision},
        )

    @staticmethod
    def decode_generated(output_ids, inputs):
        prompt_length = int(inputs["input_ids"].shape[-1]) if "input_ids" in inputs else 0
        if prompt_length and output_ids.shape[-1] > prompt_length:
            return output_ids[:, prompt_length:]
        return output_ids

    def metadata(self):
        bundle = self._require_bundle()
        return {"key": self.spec.key, "model_id": self.spec.model_id,
                "revision": bundle.revision, "adapter_path": self.spec.adapter_path,
                "backend": "qwen3_asr"}

    def close(self):
        if self.owns_bundle and self.bundle is not None:
            self.loader.release(self.bundle)
        super().close()


def _torch_dtype(name: str):
    import torch

    return {"float16": torch.float16, "bfloat16": torch.bfloat16,
            "float32": torch.float32, "auto": "auto"}[name]

