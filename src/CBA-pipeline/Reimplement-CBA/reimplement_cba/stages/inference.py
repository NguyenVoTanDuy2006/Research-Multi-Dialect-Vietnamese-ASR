from __future__ import annotations

from pathlib import Path
from typing import Any

from reimplement_cba.adapters.inference import create_transcriber
from reimplement_cba.config import CBAConfig
from reimplement_cba.stages.postprocess import VietnamesePostprocessor
from reimplement_cba.stages.prediction_io import (
    adapter_prediction_path,
    prediction_row,
    write_predictions,
)
from reimplement_cba.stages.preprocess import AudioPreprocessor


class AdaptedInferenceStage:
    """Run the selected fine-tuned checkpoint to obtain current predictions."""

    def __init__(
        self,
        config: CBAConfig,
        adapter_path: str | Path,
        *,
        ct2_model_path: str | Path | None = None,
    ):
        if adapter_path is None or not str(adapter_path).strip():
            raise ValueError("adapter_path is required for adapted inference")
        self.config = config
        self.adapter_path = str(adapter_path)
        self.model = create_transcriber(
            config,
            adapter_path=adapter_path,
            ct2_model_path=ct2_model_path,
        )
        self.audio_preprocessor = AudioPreprocessor(config)
        self.text_postprocessor = VietnamesePostprocessor()

    def run(
        self,
        records,
        audio_reader,
        *,
        output_path: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        rows = []
        self.model.load()
        try:
            for record in records:
                audio = self.audio_preprocessor.read(record, audio_reader, purpose="inference")
                raw_hypothesis = self.model.transcribe(audio)
                row = prediction_row(
                    record,
                    self.text_postprocessor.process(raw_hypothesis),
                    raw_hypothesis=raw_hypothesis,
                )
                row["inference"] = self.model.metadata()
                rows.append(row)
        finally:
            self.model.close()

        destination = output_path or adapter_prediction_path(
            self.config.output_dir,
            self.adapter_path,
        )
        write_predictions(rows, destination)
        return rows
