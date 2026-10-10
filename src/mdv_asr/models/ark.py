from __future__ import annotations

import time

from mdv_asr.config.types import DecodeConfig
from mdv_asr.data.records import AudioSample

from .base import ModelBundle, TransformersBackendBase, TranscriptionResult, extract_text


class ArkASRBackend(TransformersBackendBase):
    def load(self) -> ModelBundle:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        token = __import__("mdv_asr.runtime.environment", fromlist=["RuntimeEnvironment"]).RuntimeEnvironment.get_hf_token()
        model_id = "Edge0/ARK-ASR-3B" if self.spec.model_id == "AutoArk-AI/ARK-ASR-3B" else self.spec.model_id
        kwargs = {"token": token, "trust_remote_code": self.spec.trust_remote_code}
        if self.spec.revision:
            kwargs["revision"] = self.spec.revision
        processor = AutoProcessor.from_pretrained(model_id, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device).eval()
        self.bundle = ModelBundle(model, processor, tokenizer,
                                  revision=getattr(model.config, "_commit_hash", None),
                                  device=device, dtype=str(next(model.parameters()).dtype))
        return self.bundle

    def transcribe(self, audio: AudioSample, decode: DecodeConfig) -> TranscriptionResult:
        bundle = self._require_bundle()
        conversation = [{"role": "user", "content": [
            {"type": "audio", "audio": audio.waveform},
            {"type": "text", "text": "Please transcribe this audio."},
        ]}]
        inputs = bundle.processor.apply_chat_template(
            conversation, add_generation_prompt=True, return_tensors="pt",
            sampling_rate=audio.sample_rate, audio_padding="longest",
            text_kwargs={"padding": "longest"},
        )
        inputs = self.move_inputs(inputs)
        start = time.perf_counter()
        kwargs = {"do_sample": decode.do_sample, "max_new_tokens": decode.max_new_tokens,
                  "pad_token_id": bundle.tokenizer.pad_token_id,
                  "eos_token_id": bundle.tokenizer.eos_token_id}
        output, _ = self.generate(inputs, **kwargs)
        text = bundle.tokenizer.batch_decode(output, skip_special_tokens=True)
        return TranscriptionResult(extract_text(text), time.perf_counter() - start, self.metadata())

