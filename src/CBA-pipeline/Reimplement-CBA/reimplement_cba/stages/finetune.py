from __future__ import annotations

from dataclasses import asdict

from reimplement_cba.config import CBAConfig
from reimplement_cba.stages.initial_filter import InitialTranscriptFilter
from reimplement_cba.stages.prediction_io import write_predictions
from reimplement_cba.stages.preprocess import AudioPreprocessor


class AdaLoRATrainingStage:
    """Trainer boundary for one AdaLoRA curriculum round."""

    def __init__(self, config: CBAConfig):
        self.config = config

    def run(
        self,
        train_records,
        validation_records,
        *,
        audio_reader,
        round_index: int,
        resume_adapter: str | None = None,
    ) -> str:
        if round_index < 1:
            raise ValueError("round_index must be at least 1")
        initial_rule = self.config.values["initial_rule"]
        initial_filter = InitialTranscriptFilter(
            rejected_markers=initial_rule.get("reject_transcript_markers", []),
            rejected_sample_ids=self.config.english_exclusion_sample_ids,
        )
        filter_result = initial_filter.apply(train_records)
        preprocessor = AudioPreprocessor(self.config)
        train_selection = preprocessor.select_training_records(filter_result.kept)
        validation_selection = preprocessor.select_training_records(validation_records)

        rejected = [
            asdict(preprocessor.rejected_record(record, initial_filter.rejection_reason(record)))
            | {"purpose": "train"}
            for record in filter_result.rejected
        ]
        rejected.extend(
            asdict(item) | {"purpose": "train"} for item in train_selection.rejected
        )
        rejected.extend(
            asdict(item) | {"purpose": "trainer_validation"}
            for item in validation_selection.rejected
        )
        write_predictions(
            rejected,
            self.config.output_dir / "preprocess" / f"round-{round_index}-rejected.jsonl",
        )

        if not train_selection.kept:
            raise ValueError(f"No training records remain after preprocessing for round {round_index}")
        if not validation_selection.kept:
            raise ValueError(
                f"No Trainer validation records remain after preprocessing for round {round_index}"
            )

        def prepared_audio_reader(record):
            return preprocessor.process(audio_reader(record), purpose="training")

        # Keep optional model-training imports out of config and CLI inspection.
        from reimplement_cba.stages.adalora_trainer import AdaLoRATrainer
        trainer_stage = AdaLoRATrainer(self.config)
        return trainer_stage.run(
            train_selection.kept,
            validation_selection.kept,
            audio_reader=prepared_audio_reader,
            round_index=round_index,
            resume_adapter=resume_adapter,
        )
