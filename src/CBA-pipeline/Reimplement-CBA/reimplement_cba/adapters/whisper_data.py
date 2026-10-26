from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from mdv_asr_module.data.records import AudioSample, SampleRecord


class PhoWhisperTrainingDataset:
    """Lazily turn ViMD records into Whisper input features and token labels."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        *,
        audio_reader: Callable[[SampleRecord], AudioSample],
        processor: Any,
        min_duration_seconds: float = 0.5,
        max_duration_seconds: float = 30.0,
    ) -> None:
        self.records = tuple(records)
        self.audio_reader = audio_reader
        self.processor = processor
        self.min_duration_seconds = float(min_duration_seconds)
        self.max_duration_seconds = float(max_duration_seconds)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        sample_id = getattr(record, "sample_id", f"index-{index}")
        raw_text = getattr(record, "text", None)
        text = raw_text if isinstance(raw_text, str) else ""
        if not text.strip():
            raise ValueError(f"Training transcript is empty for sample {sample_id!r}")

        try:
            audio = self.audio_reader(record)
            if not isinstance(audio, AudioSample):
                raise TypeError(f"audio_reader returned {type(audio).__name__}, expected AudioSample")
            audio.validate()
            if audio.sample_rate != 16000:
                raise ValueError(f"expected 16000 Hz audio, got {audio.sample_rate} Hz")
            if not self.min_duration_seconds <= audio.duration_seconds <= self.max_duration_seconds:
                if audio.duration_seconds > self.max_duration_seconds:
                    raise ValueError(
                        f"training audio is {audio.duration_seconds:.3f}s; Whisper fine-tuning "
                        f"requires transcript-aligned segments no longer than "
                        f"{self.max_duration_seconds:g}s"
                    )
                raise ValueError(
                    f"duration {audio.duration_seconds:.3f}s is outside "
                    f"[{self.min_duration_seconds:g}, {self.max_duration_seconds:g}]s"
                )

            extracted = self.processor.feature_extractor(
                audio.waveform,
                sampling_rate=audio.sample_rate,
            )
            features = extracted["input_features"]
            # Feature extractors return a one-item batch even for one waveform.
            if len(features) == 1:
                features = features[0]
            labels = self.processor.tokenizer(text, add_special_tokens=True)["input_ids"]
            return {"input_features": features, "labels": list(labels)}
        except Exception as error:
            raise RuntimeError(f"Could not prepare PhoWhisper example for {sample_id!r}: {error}") from error


class PhoWhisperDataCollator:
    """Pad Whisper features and labels, excluding label padding from loss."""

    def __init__(
        self,
        processor: Any,
        *,
        decoder_start_token_id: int | None = None,
        input_features_dtype: Any | None = None,
    ) -> None:
        self.processor = processor
        self.input_features_dtype = input_features_dtype
        if decoder_start_token_id is None:
            prefix_tokens = getattr(processor.tokenizer, "prefix_tokens", None)
            if prefix_tokens:
                decoder_start_token_id = int(prefix_tokens[0])
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty PhoWhisper batch")

        audio_features = [
            {"input_features": feature["input_features"]}
            for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            audio_features,
            return_tensors="pt",
        )
        if self.input_features_dtype is not None:
            batch["input_features"] = batch["input_features"].to(
                dtype=self.input_features_dtype
            )

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        start_token_id = self.decoder_start_token_id
        if start_token_id is not None and labels.shape[1] and bool(
            labels[:, 0].eq(start_token_id).all().item()
        ):
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
