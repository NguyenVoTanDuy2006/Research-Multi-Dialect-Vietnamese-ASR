from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
CBA_ROOT = REPO_ROOT / "src" / "CBA-pipeline" / "Reimplement-CBA"
if str(CBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CBA_ROOT))

# Keep the stage tests independent of optional model packages.
adapter_package = types.ModuleType("reimplement_cba.adapters")
adapter_package.__path__ = [str(CBA_ROOT / "reimplement_cba" / "adapters")]
sys.modules.setdefault("reimplement_cba.adapters", adapter_package)

from reimplement_cba.config import CBAConfig  # noqa: E402
from reimplement_cba.stages.finetune import AdaLoRATrainingStage  # noqa: E402
from reimplement_cba.stages.postprocess import VietnamesePostprocessor  # noqa: E402
from reimplement_cba.stages.preprocess import AudioPreprocessor  # noqa: E402


CONFIG_PATH = CBA_ROOT / "configs" / "vimd_phowhisper_medium.json"


class DummyAudio:
    def __init__(self, duration_seconds: float, sample_id: str, sample_rate: int = 16000):
        self.duration_seconds = duration_seconds
        self.sample_id = sample_id
        self.sample_rate = sample_rate

    def validate(self):
        return None


def make_config(output_dir: str | None = None) -> CBAConfig:
    values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if output_dir is not None:
        values["output_dir"] = output_dir
    return CBAConfig(CONFIG_PATH, values)


def record(sample_id: str, duration: float, split: str = "train", text: str = "Xin chào"):
    return types.SimpleNamespace(
        sample_id=sample_id, duration_seconds=duration, split=split, text=text
    )


class AudioPreprocessorTests(unittest.TestCase):
    def test_select_training_records_excludes_out_of_range_with_reasons(self):
        selection = AudioPreprocessor(make_config()).select_training_records([
            record("short", 0.49),
            record("edge", 30.0),
            record("long", 30.01),
            record("bad", float("nan")),
        ])
        self.assertEqual([item.sample_id for item in selection.kept], ["edge"])
        self.assertEqual(
            [(item.sample_id, item.reason) for item in selection.rejected],
            [
                ("short", "below_min_duration"),
                ("long", "above_max_duration"),
                ("bad", "invalid_duration"),
            ],
        )

    def test_audio_boundary_accepts_long_inference_but_rejects_long_training(self):
        audio = DummyAudio(31, "long")
        preprocessor = AudioPreprocessor(make_config())
        self.assertIs(preprocessor.process(audio, purpose="inference"), audio)
        with self.assertRaisesRegex(ValueError, "long.*31.*30"):
            preprocessor.process(audio, purpose="training")

    def test_invalid_decoded_audio_error_identifies_sample(self):
        class InvalidAudio(DummyAudio):
            def validate(self):
                raise ValueError("Audio contains NaN")

        with self.assertRaisesRegex(ValueError, "broken.*NaN"):
            AudioPreprocessor(make_config()).process(
                InvalidAudio(1, "broken"), purpose="training"
            )


class VietnamesePostprocessorTests(unittest.TestCase):
    def test_normalizes_unicode_and_spacing_without_erasing_spoken_content(self):
        raw = "  Tôi   tôi  tôi,\tOpenAI!  "
        expected = "Tôi tôi tôi, OpenAI!"
        processor = VietnamesePostprocessor()
        self.assertEqual(processor.process(raw), expected)
        self.assertEqual(processor.process(expected), expected)


class StageIntegrationTests(unittest.TestCase):
    def test_finetune_filters_duration_and_writes_auditable_report(self):
        captured = {}

        class FakeTrainer:
            def __init__(self, config):
                self.config = config

            def run(self, train_records, validation_records, *, audio_reader,
                    round_index, resume_adapter):
                captured["train"] = [item.sample_id for item in train_records]
                captured["validation"] = [item.sample_id for item in validation_records]
                captured["audio_reader"] = audio_reader
                return "adapter"

        fake_module = types.SimpleNamespace(AdaLoRATrainer=FakeTrainer)
        with TemporaryDirectory() as temporary:
            config = make_config(temporary)
            manifest = Path(temporary) / "english.json"
            manifest.write_text(json.dumps({"sample_ids": ["english"]}), encoding="utf-8")
            config.values["initial_rule"]["english_sample_ids_manifest"] = str(manifest)
            with patch.dict(sys.modules, {
                "reimplement_cba.stages.adalora_trainer": fake_module
            }):
                result = AdaLoRATrainingStage(config).run(
                    [record("keep", 30), record("long", 31),
                     record("marker", 1, text="[noise]"),
                     record("english", 1, text="hello")],
                    [record("valid", 1, "validation"),
                     record("valid-long", 31, "validation")],
                    audio_reader=lambda item: DummyAudio(1, item.sample_id),
                    round_index=2,
                )
            self.assertEqual(result, "adapter")
            self.assertEqual(captured["train"], ["keep"])
            self.assertEqual(captured["validation"], ["valid"])
            report = Path(temporary) / "preprocess" / "round-2-rejected.jsonl"
            rows = [json.loads(line) for line in report.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(
                [(row["sample_id"], row["reason"]) for row in rows],
                [("marker", "transcript_marker"),
                 ("english", "approved_english_sample_id"),
                 ("long", "above_max_duration"),
                 ("valid-long", "above_max_duration")],
            )
            self.assertEqual(rows[2]["duration_seconds"], 31.0)
            self.assertEqual(rows[2]["purpose"], "train")
            self.assertEqual(rows[3]["split"], "validation")
            self.assertEqual(rows[3]["purpose"], "trainer_validation")

    def test_finetune_rejects_actual_audio_longer_than_its_indexed_duration(self):
        class ReadingTrainer:
            def __init__(self, config):
                pass

            def run(self, train_records, validation_records, *, audio_reader,
                    round_index, resume_adapter):
                audio_reader(train_records[0])
                return "adapter"

        with TemporaryDirectory() as temporary:
            with patch.dict(sys.modules, {
                "reimplement_cba.stages.adalora_trainer":
                    types.SimpleNamespace(AdaLoRATrainer=ReadingTrainer)
            }):
                with self.assertRaisesRegex(ValueError, "listed-short.*31.*30"):
                    AdaLoRATrainingStage(make_config(temporary)).run(
                        [record("listed-short", 1)],
                        [record("valid", 1, "validation")],
                        audio_reader=lambda item: DummyAudio(31, item.sample_id),
                        round_index=1,
                    )

    def test_inference_preserves_raw_and_writes_postprocessed_hypothesis(self):
        from reimplement_cba.stages.initial_inference import InitialInferenceStage

        class FakeTranscriber:
            def load(self):
                pass

            def transcribe(self, audio):
                return " Tôi   tôi  tôi. "

            def metadata(self):
                return {"backend": "fake"}

            def close(self):
                pass

        with TemporaryDirectory() as temporary:
            config = make_config(temporary)
            with patch("reimplement_cba.stages.initial_inference.create_transcriber",
                       return_value=FakeTranscriber()):
                rows = InitialInferenceStage(config).run(
                    [record("a", 1)],
                    lambda item: DummyAudio(1, item.sample_id),
                )
            self.assertEqual(rows[0]["hypothesis_raw"], " Tôi   tôi  tôi. ")
            self.assertEqual(rows[0]["hypothesis"], "Tôi tôi tôi.")
            saved = Path(temporary) / "predictions" / "baseline.jsonl"
            saved_row = json.loads(saved.read_text(encoding="utf-8"))
            self.assertEqual(saved_row["hypothesis_raw"], " Tôi   tôi  tôi. ")
            self.assertEqual(saved_row["hypothesis"], "Tôi tôi tôi.")

    def test_adapted_inference_accepts_long_audio_and_postprocesses_output(self):
        from reimplement_cba.stages.inference import AdaptedInferenceStage

        class FakeTranscriber:
            def load(self):
                pass

            def transcribe(self, audio):
                return "  Xin   chào  "

            def metadata(self):
                return {"backend": "fake"}

            def close(self):
                pass

        with TemporaryDirectory() as temporary:
            with patch("reimplement_cba.stages.inference.create_transcriber",
                       return_value=FakeTranscriber()):
                rows = AdaptedInferenceStage(make_config(temporary), "adapter").run(
                    [record("long", 31, split="validation")],
                    lambda item: DummyAudio(31, item.sample_id),
                )
            self.assertEqual(rows[0]["hypothesis"], "Xin chào")
            self.assertEqual(rows[0]["hypothesis_raw"], "  Xin   chào  ")

    def test_inference_reader_failure_identifies_sample_in_both_stages(self):
        from reimplement_cba.stages.initial_inference import InitialInferenceStage
        from reimplement_cba.stages.inference import AdaptedInferenceStage

        def bad_reader(record):
            raise ValueError("Audio contains NaN or Inf")

        class FakeTranscriber:
            def load(self):
                pass

            def close(self):
                pass

        with TemporaryDirectory() as temporary:
            config = make_config(temporary)
            cases = (
                ("reimplement_cba.stages.initial_inference.create_transcriber",
                 lambda: InitialInferenceStage(config)),
                ("reimplement_cba.stages.inference.create_transcriber",
                 lambda: AdaptedInferenceStage(config, "adapter")),
            )
            for target, build_stage in cases:
                with self.subTest(target=target):
                    with patch(target, return_value=FakeTranscriber()):
                        with self.assertRaisesRegex(RuntimeError, "broken.*NaN"):
                            build_stage().run([record("broken", 1)], bad_reader)


if __name__ == "__main__":
    unittest.main()
