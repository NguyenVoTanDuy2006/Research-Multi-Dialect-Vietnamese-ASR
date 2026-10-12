from __future__ import annotations

import time

from mdv_asr_module.configs.types import DecodeConfig
from mdv_asr_module.data.records import AudioSample

from .base import ModelBundle, TransformersBackendBase, TranscriptionResult, extract_text


class WhisperASRBackend(TransformersBackendBase):
    def load(self) -> ModelBundle:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        token = __import__("mdv_asr_module.runtime.environment", fromlist=["RuntimeEnvironment"]).RuntimeEnvironment.get_hf_token()
        kwargs = {"token": token}
        if self.spec.revision:
            kwargs["revision"] = self.spec.revision
        processor = AutoProcessor.from_pretrained(self.spec.model_id, **kwargs)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(self.spec.model_id, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        self.bundle = ModelBundle(model, processor, revision=getattr(model.config, "_commit_hash", None),
                                  device=device, dtype=str(next(model.parameters()).dtype))
        return self.bundle

    def transcribe(self, audio: AudioSample, decode: DecodeConfig) -> TranscriptionResult:
        inputs = self._require_bundle().processor(
            audio.waveform, sampling_rate=audio.sample_rate, return_tensors="pt"
        )
        inputs = self.move_inputs(inputs)
        start = time.perf_counter()
        output, _ = self.generate(inputs, max_new_tokens=decode.max_new_tokens,
                                  do_sample=decode.do_sample, num_beams=decode.num_beams,
                                  language="vi", task="transcribe")
        text = self._require_bundle().processor.batch_decode(output, skip_special_tokens=True)[0]
        return TranscriptionResult(extract_text(text), time.perf_counter() - start, self.metadata())

