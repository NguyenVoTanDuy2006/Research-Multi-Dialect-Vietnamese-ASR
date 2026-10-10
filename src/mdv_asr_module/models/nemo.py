from __future__ import annotations

import time

from mdv_asr_module.config.types import DecodeConfig
from mdv_asr_module.data.records import AudioSample

from .base import ASRBackend, ModelBundle, TranscriptionResult, extract_text


class CanaryASRBackend(ASRBackend):
    def load(self) -> ModelBundle:
        from nemo.collections.speechlm2.models import SALM

        model = SALM.from_pretrained(self.spec.model_id).eval()
        try:
            import torch

            if torch.cuda.is_available():
                model.cuda()
                if torch.cuda.get_device_capability()[0] < 8:
                    model.half()
                device = "cuda"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
        self.bundle = ModelBundle(model, tokenizer=getattr(model, "tokenizer", None), device=device,
                                  revision=self.spec.revision, metadata={"backend": "nemo_salm"})
        return self.bundle

    def transcribe(self, audio: AudioSample, decode: DecodeConfig) -> TranscriptionResult:
        import tempfile
        import soundfile as sf

        bundle = self._require_bundle()
        with tempfile.NamedTemporaryFile(suffix=".wav") as stream:
            sf.write(stream.name, audio.waveform, audio.sample_rate)
            prompt = [[{"role": "user", "content":
                        f"Transcribe the following: {bundle.model.audio_locator_tag}",
                        "audio": [stream.name]}]]
            start = time.perf_counter()
            output = bundle.model.generate(prompts=prompt, max_new_tokens=decode.max_new_tokens)
            decoded = bundle.model.tokenizer.ids_to_text(output[0].detach().cpu())
        return TranscriptionResult(extract_text(decoded), time.perf_counter() - start, self.metadata())

    def metadata(self):
        return {"key": self.spec.key, "model_id": self.spec.model_id, "backend": "nemo_salm"}


class ParakeetASRBackend(ASRBackend):
    def load(self) -> ModelBundle:
        import nemo.collections.asr as nemo_asr

        model = nemo_asr.models.ASRModel.from_pretrained(model_name=self.spec.model_id).eval()
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        if device == "cuda":
            model.cuda()
        self.bundle = ModelBundle(model, device=device, revision=self.spec.revision,
                                  metadata={"backend": "nemo_asr"})
        return self.bundle

    def transcribe(self, audio: AudioSample, decode: DecodeConfig) -> TranscriptionResult:
        import tempfile
        import soundfile as sf

        bundle = self._require_bundle()
        with tempfile.NamedTemporaryFile(suffix=".wav") as stream:
            sf.write(stream.name, audio.waveform, audio.sample_rate)
            start = time.perf_counter()
            result = bundle.model.transcribe([stream.name])
        return TranscriptionResult(extract_text(result), time.perf_counter() - start, self.metadata())

    def metadata(self):
        return {"key": self.spec.key, "model_id": self.spec.model_id, "backend": "nemo_asr"}

