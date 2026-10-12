from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mdv_asr_module.configs.types import (
    BenchmarkConfig,
    ExperimentConfig,
    ModelSpec,
)
from mdv_asr_module.workflows.benchmark import ASRBenchmarkWorkflow


class _Index:
    source_fingerprint = "dataset-fingerprint"

    def for_split(self, split):
        return [object()]


class _Runner:
    store = None

    def __init__(self, config, source, evaluator, store, decode):
        type(self).store = store

    def run(self, records):
        return []

    def validate_comparison(self):
        return None


class _Reporter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def write_all(self, results, baseline_key=None):
        return {}


class BenchmarkWorkflowTests(unittest.TestCase):
    def test_run_test_passes_flush_interval_to_prediction_store(self):
        config = ExperimentConfig(
            benchmark=BenchmarkConfig(
                models=(ModelSpec(key="base", backend="test", model_id="test-model"),),
                split="test",
                resume=False,
                flush_every=7,
            )
        )
        dataset_workflow = SimpleNamespace(index=_Index(), source=object())

        with tempfile.TemporaryDirectory() as tmp:
            workflow = ASRBenchmarkWorkflow(config, dataset_workflow, output_dir=tmp)
            with patch("mdv_asr_module.workflows.benchmark.BenchmarkRunner", _Runner), \
                    patch("mdv_asr_module.workflows.benchmark.BenchmarkReporter", _Reporter):
                results = workflow.run_test()

        self.assertEqual(results, [])
        self.assertEqual(_Runner.store.flush_every, 7)


if __name__ == "__main__":
    unittest.main()
