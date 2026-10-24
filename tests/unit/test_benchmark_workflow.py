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
    SemanticMetricsConfig,
)
from mdv_asr_module.workflows.benchmark import ASRBenchmarkWorkflow


class _Index:
    source_fingerprint = "dataset-fingerprint"

    def for_split(self, split):
        return [object()]


class _Runner:
    store = None
    events = []
    results = []

    def __init__(self, config, source, evaluator, store, decode):
        type(self).store = store

    def run(self, records):
        type(self).events.append("inference")
        return type(self).results

    def validate_comparison(self):
        type(self).events.append("comparison")
        return None


class _Reporter:
    events = []

    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def write_all(self, results, baseline_key=None):
        type(self).events.append("ordinary_report")
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
        _Runner.results = []

        with tempfile.TemporaryDirectory() as tmp:
            workflow = ASRBenchmarkWorkflow(config, dataset_workflow, output_dir=tmp)
            with patch("mdv_asr_module.workflows.benchmark.BenchmarkRunner", _Runner), \
                    patch("mdv_asr_module.workflows.benchmark.BenchmarkReporter", _Reporter):
                results = workflow.run_test()

        self.assertEqual(results, [])
        self.assertEqual(_Runner.store.flush_every, 7)

    def test_enabled_semantic_report_runs_after_ordinary_artifacts_and_before_comparison(self):
        config = ExperimentConfig(
            benchmark=BenchmarkConfig(
                models=(ModelSpec(key="base", backend="test", model_id="test-model"),),
                split="test", resume=False,
            ),
            semantic_metrics=SemanticMetricsConfig(enabled=True),
        )
        dataset_workflow = SimpleNamespace(index=_Index(), source=object())
        results = [SimpleNamespace(predictions=["prediction-row"])]
        events = []
        _Runner.events = events
        _Runner.results = results
        _Reporter.events = events
        semantic_calls = []

        class _SemanticReporter:
            def __init__(self, output_dir, semantic_config, *, resume):
                self.resume = resume
                semantic_calls.append((output_dir, semantic_config, resume))
                events.append("semantic_init")

            def write_all(self, predictions):
                self.predictions = predictions
                semantic_calls.append(predictions)
                events.append("semantic_report")
                return {}

        with tempfile.TemporaryDirectory() as tmp:
            workflow = ASRBenchmarkWorkflow(config, dataset_workflow, output_dir=tmp)
            with patch("mdv_asr_module.workflows.benchmark.BenchmarkRunner", _Runner), \
                    patch("mdv_asr_module.workflows.benchmark.BenchmarkReporter", _Reporter), \
                    patch("mdv_asr_module.workflows.benchmark.SemanticBenchmarkReporter",
                          _SemanticReporter, create=True):
                returned = workflow.run_test()

        self.assertIs(returned, results)
        self.assertEqual(events, [
            "inference", "ordinary_report", "semantic_init", "semantic_report", "comparison",
        ])
        self.assertEqual(
            semantic_calls,
            [(Path(tmp), config.semantic_metrics, False), ["prediction-row"]],
        )

    def test_disabled_semantic_reporter_is_not_constructed(self):
        config = ExperimentConfig(
            benchmark=BenchmarkConfig(
                models=(ModelSpec(key="base", backend="test", model_id="test-model"),),
                split="test", resume=False,
            )
        )
        dataset_workflow = SimpleNamespace(index=_Index(), source=object())
        _Runner.events = []
        _Runner.results = []
        _Reporter.events = []

        with tempfile.TemporaryDirectory() as tmp:
            workflow = ASRBenchmarkWorkflow(config, dataset_workflow, output_dir=tmp)
            with patch("mdv_asr_module.workflows.benchmark.BenchmarkRunner", _Runner), \
                    patch("mdv_asr_module.workflows.benchmark.BenchmarkReporter", _Reporter), \
                    patch("mdv_asr_module.workflows.benchmark.SemanticBenchmarkReporter",
                          create=True) as semantic_reporter:
                workflow.run_test()

        semantic_reporter.assert_not_called()


if __name__ == "__main__":
    unittest.main()
