from __future__ import annotations

from mdv_asr_module.config.types import DecodeConfig
from mdv_asr_module.data.records import AudioSample

from .base import ModelBundle, TransformersBackendBase, TranscriptionResult, extract_text


class CTCASRBackend(TransformersBackendBase):
    def load(self) -> ModelBundle:
        import torch
        from transformers import AutoModelForCTC, AutoProcessor

        token = __import__("mdv_asr_module.runtime.environment", fromlist=["RuntimeEnvironment"]).RuntimeEnvironment.get_hf_token()
        kwargs = {"token": token}
        if self.spec.revision:
            kwargs["revision"] = self.spec.revision
        processor = AutoProcessor.from_pretrained(self.spec.model_id, **kwargs)
        model = AutoModelForCTC.from_pretrained(self.spec.model_id, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        self.bundle = ModelBundle(model, processor, revision=getattr(model.config, "_commit_hash", None),
                                  device=device, dtype=str(next(model.parameters()).dtype))
        return self.bundle

    def transcribe(self, audio: AudioSample, decode: DecodeConfig) -> TranscriptionResult:
        import torch
        import time

        bundle = self._require_bundle()
        inputs = bundle.processor(audio.waveform, sampling_rate=audio.sample_rate, return_tensors="pt")
        inputs = self.move_inputs(inputs)
        start = time.perf_counter()
        with torch.inference_mode():
            logits = bundle.model(**inputs).logits
        ids = logits.argmax(dim=-1)
        text = bundle.processor.batch_decode(ids)[0]
        return TranscriptionResult(extract_text(text), time.perf_counter() - start, self.metadata())

