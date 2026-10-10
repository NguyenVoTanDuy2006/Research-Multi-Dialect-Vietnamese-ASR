from __future__ import annotations

from mdv_asr.config.types import BenchmarkConfig, DecodeConfig
from mdv_asr.core.errors import IncompleteEvaluationError
from mdv_asr.models.factory import ASRBackendFactory
from mdv_asr.data.records import SampleRecord
from mdv_asr.data.sources import DatasetSource

from .evaluator import ASREvaluator
from .records import EvaluationResult
from .store import PredictionStore


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig, source: DatasetSource,
                 evaluator: ASREvaluator | None = None,
                 store: PredictionStore | None = None,
                 decode: DecodeConfig | None = None):
        config.validate()
        self.config = config
        self.source = source
        self.evaluator = evaluator or ASREvaluator()
        self.store = store
        self.decode = decode or DecodeConfig()
        self.results: list[EvaluationResult] = []

    def prepare(self, records: list[SampleRecord]) -> list[SampleRecord]:
        keys = [(row.dataset_id, row.sample_id) for row in records]
        if len(keys) != len(set(keys)):
            raise ValueError("Benchmark input contains duplicate dataset/sample IDs")
        if not records:
            raise ValueError("Benchmark requires at least one sample")
        return records

    def run_model(self, model_spec, records: list[SampleRecord]) -> EvaluationResult:
        backend = ASRBackendFactory.create(model_spec)
        backend.decode_config = self.decode
        try:
            backend.load()
            if self.store is not None:
                completed = self.store.completed_keys(model_spec.key)
                pending = [record for record in records
                           if (record.dataset_id, record.sample_id, model_spec.key) not in completed]
            else:
                pending = records
            result = self.evaluator.evaluate_model(backend, pending, self.source)
            expected_keys = {(row.dataset_id, row.sample_id, model_spec.key) for row in records}
            if self.store is not None:
                for row in result.predictions:
                    self.store.upsert(row)
                self.store.flush()
                cached = [row for row in self.store.load_all() if row.model_key == model_spec.key]
                expected_keys = {(row.dataset_id, row.sample_id, model_spec.key) for row in records}
                observed = {row.key for row in cached}
                if observed != expected_keys:
                    missing = expected_keys - observed
                    raise IncompleteEvaluationError(f"Prediction cache missing {len(missing)} rows")
                result.predictions = cached
            result.expected_keys = expected_keys
            self.results.append(result)
            return result
        finally:
            backend.close()

    def run(self, records: list[SampleRecord]) -> list[EvaluationResult]:
        records = self.prepare(records)
        return [self.run_model(spec, records) for spec in self.config.models]

    def resume(self, records: list[SampleRecord]) -> list[EvaluationResult]:
        return self.run(records)

    def validate_comparison(self) -> None:
        key_sets = [{row.key[:2] for row in result.predictions if not row.error}
                    for result in self.results]
        if key_sets and not key_sets[0]:
            raise IncompleteEvaluationError("No successful predictions are available for comparison")
        if key_sets and any(keys != key_sets[0] for keys in key_sets[1:]):
            raise IncompleteEvaluationError("Models do not have paired successful predictions")

    def close(self) -> None:
        self.source.close()

