from __future__ import annotations

import json
import io
import sys
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
CBA_ROOT = REPO_ROOT / "src" / "CBA-pipeline" / "Reimplement-CBA"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(CBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CBA_ROOT))

from reimplement_cba.config import CBAConfig  # noqa: E402
from reimplement_cba.workflow import run  # noqa: E402


CONFIG_PATH = CBA_ROOT / "configs" / "vimd_phowhisper_medium.json"


def make_config(output_dir: str, *, rounds: int = 2, domain: str = "in_domain",
                backend: str = "transformers") -> CBAConfig:
    values = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    values["output_dir"] = output_dir
    values["curriculum"]["max_rounds"] = rounds
    values["curriculum"]["domain"] = domain
    values["inference"]["backend"] = backend
    return CBAConfig(CONFIG_PATH, values)


def record(sample_id: str, split: str, duration: float = 1.0,
           text: str = "Xin chào"):
    return types.SimpleNamespace(
        sample_id=sample_id, split=split, duration_seconds=duration,
        text=text, region="miền Nam", province_name=None,
    )


class FakeViMD:
    def __init__(self):
        self.index = types.SimpleNamespace(records=[
            record("a", "train"), record("b", "train"),
            record("marked", "train", text="[noise]"),
            record("long", "train", duration=31),
            record("v", "validation"), record("t", "test"),
        ])
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True

    def validate_config(self):
        return {"dataset_id": "fake_vimd"}

    def validate_splits(self):
        return {"record_count": len(self.index.records)}

    def open(self):
        return self

    def read_audio(self, item):
        return item.sample_id


class FakeMetrics:
    def score_batch(self, references, hypotheses, *, regions=None):
        return [types.SimpleNamespace(semscore=float(hypothesis[1:]))
                for hypothesis in hypotheses]

    def evaluate_corpus(self, references, hypotheses, *, regions=None):
        scores = [float(hypothesis[1:]) for hypothesis in hypotheses]
        mean = sum(scores) / len(scores)
        return {"n_samples": len(scores), "semscore": mean,
                "WER": (100 - mean) / 100}


class FakeRuntime:
    def __init__(self):
        self.dataset = FakeViMD()
        self.events = []
        self.train_calls = []
        self.convert_calls = []
        self.ct2_paths = []

    def open_dataset(self, config):
        return self.dataset

    def create_metrics(self, config):
        return FakeMetrics()

    def baseline_stage(self, config, *, ct2_model_path=None):
        runtime = self

        class Stage:
            def run(self, records, audio_reader, *, output_path):
                runtime.events.append(("baseline", records[0].split))
                return [
                    {"sample_id": item.sample_id, "split": item.split,
                     "reference": item.text,
                     "hypothesis": "q20" if item.sample_id == "a" else
                                   "q10" if item.sample_id == "b" else "q50",
                     "region": item.region}
                    for item in records
                ]

        return Stage()

    def training_stage(self, config):
        runtime = self

        class Stage:
            def run(self, train_records, validation_records, *, audio_reader,
                    round_index, resume_adapter=None):
                runtime.events.append(("train", round_index))
                runtime.train_calls.append({
                    "round": round_index,
                    "train_ids": [item.sample_id for item in train_records],
                    "resume_adapter": resume_adapter,
                })
                root = config.output_dir / "adapters" / f"round-{round_index}"
                candidates = [root / "trainer" / "checkpoint-1",
                              root / "checkpoint-final"]
                for candidate in candidates:
                    candidate.mkdir(parents=True, exist_ok=True)
                    (candidate / "adapter_config.json").write_text("{}", encoding="utf-8")
                    (candidate / "adapter_model.safetensors").write_bytes(b"fake")
                return str(candidates[-1])

        return Stage()

    def adapted_stage(self, config, adapter_path, *, ct2_model_path=None):
        runtime = self
        checkpoint = Path(adapter_path)
        runtime.ct2_paths.append(ct2_model_path)

        class Stage:
            def run(self, records, audio_reader, *, output_path):
                split = records[0].split
                runtime.events.append(("adapted", split, str(checkpoint)))
                round_index = 1 if "round-1" in checkpoint.parts else 2
                if split == "validation":
                    score = (80 if checkpoint.name == "checkpoint-1" else 70) \
                        if round_index == 1 else \
                        (90 if checkpoint.name == "checkpoint-1" else 85)
                    scores = {item.sample_id: score for item in records}
                elif split == "train":
                    scores = {"a": 80, "b": 20}
                else:
                    scores = {item.sample_id: 90 for item in records}
                return [
                    {"sample_id": item.sample_id, "split": item.split,
                     "reference": item.text, "hypothesis": f"q{scores[item.sample_id]}",
                     "region": item.region}
                    for item in records
                ]

        return Stage()

    def prepare_ct2(self, config, destination, *, adapter_path=None):
        destination = Path(destination)
        destination.mkdir(parents=True)
        self.convert_calls.append((str(destination), adapter_path))
        return destination


class WorkflowTests(unittest.TestCase):
    def test_smoke_limit_uses_first_eligible_train_and_first_validation_test_records(self):
        with TemporaryDirectory() as temporary:
            runtime = FakeRuntime()
            runtime.dataset.index.records.extend([
                record("v2", "validation"), record("t2", "test"),
            ])

            summary = run(
                make_config(temporary, rounds=1),
                runtime=runtime,
                max_samples_per_split=1,
            )

            self.assertEqual(summary["initial_train_count"], 1)
            self.assertEqual(runtime.train_calls[0]["train_ids"], ["a"])
            self.assertEqual(summary["baseline_validation"]["n_samples"], 1)
            self.assertEqual(summary["final"]["test"]["n_samples"], 1)
            self.assertEqual(summary["sample_limit_per_split"], 1)

    def test_run_cli_prints_summary_location_and_final_metrics(self):
        from reimplement_cba.cli import main

        result = {"status": "complete", "rounds": [{"round_index": 1}],
                  "final": {"checkpoint": "adapter", "test": {"WER": 0.1}}}
        output = io.StringIO()
        with patch("reimplement_cba.workflow.run", return_value=result):
            with redirect_stdout(output):
                exit_code = main(["run", "--config", str(CONFIG_PATH)])
        payload = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["final"]["test"]["WER"], 0.1)
        self.assertEqual(payload["rounds_completed"], 1)
        self.assertTrue(payload["summary_path"].endswith("run-summary.json"))

    def test_two_rounds_select_validation_checkpoint_and_filter_noisy_train_record(self):
        with TemporaryDirectory() as temporary:
            config = make_config(temporary)
            runtime = FakeRuntime()
            summary = run(config, runtime=runtime)

            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(summary["rounds"]), 2)
            self.assertTrue(summary["rounds"][0]["selected_checkpoint"].endswith(
                "round-1/trainer/checkpoint-1"
            ))
            self.assertEqual(summary["rounds"][0]["curriculum"]["rejected_count"], 1)
            self.assertEqual(runtime.train_calls[0]["train_ids"], ["a", "b"])
            self.assertEqual(runtime.train_calls[1]["train_ids"], ["a"])
            self.assertEqual(runtime.train_calls[1]["resume_adapter"],
                             summary["rounds"][0]["selected_checkpoint"])
            self.assertTrue(summary["final"]["checkpoint"].endswith(
                "round-2/trainer/checkpoint-1"
            ))
            self.assertEqual(summary["final"]["test"]["semscore"], 90)
            self.assertTrue(runtime.dataset.closed)
            self.assertTrue((Path(temporary) / "run-summary.json").is_file())
            decisions = Path(temporary) / "curriculum" / "round-1-decisions.jsonl"
            rows = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(row["sample_id"], row["keep"]) for row in rows],
                             [("a", True), ("b", False)])
            self.assertFalse(any(event[1] == "test" for event in runtime.events
                                 if event[0] in {"baseline", "adapted"} and
                                 runtime.events.index(event) < runtime.events.index(("train", 2))))

    def test_out_of_domain_empty_selection_stops_before_next_training_round(self):
        with TemporaryDirectory() as temporary:
            summary = run(make_config(temporary, domain="out_of_domain"),
                          runtime=FakeRuntime())
            self.assertEqual(len(summary["rounds"]), 1)
            self.assertEqual(summary["stop_reason"], "no_samples_after_round_1")
            self.assertEqual(summary["final"]["test"]["semscore"], 90)

    def test_ct2_backend_prepares_baseline_and_candidate_artifacts(self):
        with TemporaryDirectory() as temporary:
            runtime = FakeRuntime()
            summary = run(make_config(temporary, rounds=1, backend="faster_whisper"),
                          runtime=runtime)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(runtime.convert_calls), 3)
            self.assertEqual(runtime.convert_calls[0][1], None)
            self.assertTrue(all(path is not None for path in runtime.ct2_paths))


if __name__ == "__main__":
    unittest.main()
