from __future__ import annotations

import time

from mdv_asr.config.types import DecodeConfig
from mdv_asr.data.records import AudioSample

from .base import ModelBundle, TransformersBackendBase, TranscriptionResult, extract_text


class CohereASRBackend(TransformersBackendBase):
    def load(self) -> ModelBundle:
        import torch
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        token = __import__("mdv_asr.runtime.environment", fromlist=["RuntimeEnvironment"]).RuntimeEnvironment.get_hf_token()
        kwargs = {"token": token, "trust_remote_code": self.spec.trust_remote_code}
        if self.spec.revision:
            kwargs["revision"] = self.spec.revision
        processor = AutoProcessor.from_pretrained(self.spec.model_id, **kwargs)
        model = CohereAsrForConditionalGeneration.from_pretrained(self.spec.model_id, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        self.bundle = ModelBundle(model, processor, revision=getattr(model.config, "_commit_hash", None),
                                  device=device, dtype=str(next(model.parameters()).dtype))
        return self.bundle

    def transcribe(self, audio: AudioSample, decode: DecodeConfig) -> TranscriptionResult:
        bundle = self._require_bundle()
        inputs = bundle.processor(audio.waveform, sampling_rate=audio.sample_rate,
                                  return_tensors="pt", language="vi")
        inputs = self.move_inputs(inputs)
        chunk_index = inputs.get("audio_chunk_index")
        start = time.perf_counter()
        output, _ = self.generate(inputs, max_new_tokens=decode.max_new_tokens,
                                  do_sample=decode.do_sample)
        try:
            decoded = bundle.processor.decode(output, skip_special_tokens=True,
                                              audio_chunk_index=chunk_index, language="vi")
        except TypeError:
            decoded = bundle.processor.decode(output, skip_special_tokens=True)
        return TranscriptionResult(extract_text(decoded), time.perf_counter() - start, self.metadata())

