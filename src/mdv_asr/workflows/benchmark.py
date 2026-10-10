from __future__ import annotations

from pathlib import Path

from mdv_asr.config.types import BenchmarkConfig, DecodeConfig
from mdv_asr.data.audio import AudioReader
from mdv_asr.data.records import SampleRecord
from mdv_asr.data.sources import DatasetSource, DatasetSourceFactory
from mdv_asr.evaluation.benchmark import BenchmarkRunner
from mdv_asr.evaluation.evaluator import ASREvaluator
from mdv_asr.evaluation.records import PredictionRecord
from mdv_asr.evaluation.reports import BenchmarkReporter
from mdv_asr.evaluation.store import PredictionStore
from mdv_asr.models.audit import AdapterAuditor
from mdv_asr.models.factory import ASRBackendFactory


class ASRBenchmarkWorkflow:
    def __init__(self, config, dataset_workflow, *, output_dir: str | Path | None = None):
        self.config = config
        self.dataset_workflow = dataset_workflow
        self.output_dir = Path(output_dir or config.output_dir)
        self.reader = AudioReader()
        self._single_backends = {}

    def prepare(self):
        self.config.validate()
        if self.dataset_workflow.index is None:
            self.dataset_workflow.scan()
        if self.config.benchmark is None:
            raise ValueError("Experiment config has no benchmark section")
        return self.dataset_workflow.index

    def audit_adapters(self):
        reports = {}
        for spec in self.config.benchmark.models:
            if spec.adapter_path:
                reports[spec.key] = AdapterAuditor().audit(spec.adapter_path)
        return reports

    def predict_by_filename(self, model_key: str, filename: str,
                            split: str | None = None) -> PredictionRecord:
        index = self.prepare()
        record = index.find_by_filename(filename, split=split)
        spec = next((model for model in self.config.benchmark.models if model.key == model_key), None)
        if spec is None:
            raise KeyError(f"Unknown model key: {model_key}")
        backend = self._single_backends.get(model_key)
        if backend is None:
            backend = ASRBackendFactory.create(spec)
            backend.decode_config = self.config.decode
            backend.load()
            self._single_backends[model_key] = backend
        audio = self.reader.read(record, self.dataset_workflow.source)
        result = backend.transcribe(audio, self.config.decode)
        return PredictionRecord(
            dataset_id=record.dataset_id, sample_id=record.sample_id, model_key=model_key,
            reference=record.text, prediction=result.text, split=record.split,
            filename=record.filename, region=record.region,
            province_name=record.province_name, speaker_id=record.speaker_id,
            inference_seconds=result.elapsed_seconds, metadata=result.metadata,
        )

    def run_test(self):
        index = self.prepare()
        records = list(index.for_split(self.config.benchmark.split))
        prediction_path = self.output_dir / "predictions.jsonl"
        run_config = {"dataset_fingerprint": index.source_fingerprint,
                      "models": [model.to_dict() for model in self.config.benchmark.models],
                      "decode": self.config.decode.to_dict(), "split": self.config.benchmark.split}
        store = PredictionStore(prediction_path, self.config.benchmark.flush_every)
        store.open(run_config, resume=self.config.benchmark.resume)
        runner = BenchmarkRunner(
            self.config.benchmark, self.dataset_workflow.source,
            ASREvaluator(), store, self.config.decode,
        )
        results = runner.run(records)
        BenchmarkReporter(self.output_dir).write_all(
            results, baseline_key=self.config.benchmark.models[0].key
        )
        # Keep predictions/coverage artifacts even when an inference error leaves
        # models with different paired sample sets.
        runner.validate_comparison()
        return results

    def run_cases(self, case_ids):
        index = self.prepare()
        records = [index.get(sample_id) for sample_id in case_ids]
        runner = BenchmarkRunner(self.config.benchmark, self.dataset_workflow.source,
                                 ASREvaluator(), decode=self.config.decode)
        return runner.run(records)

    def report(self):
        rows = PredictionStore(self.output_dir / "predictions.jsonl").load_all()
        from mdv_asr.evaluation.metrics import MetricAggregator

        aggregator = MetricAggregator()
        return {model: aggregator.overall(
            row for row in rows if row.model_key == model and not row.error
        ) for model in sorted({row.model_key for row in rows})}

    def close(self):
        for backend in self._single_backends.values():
            backend.close()
        self._single_backends.clear()

