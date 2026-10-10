from __future__ import annotations

import time

from mdv_asr_module.core.errors import IncompleteEvaluationError
from mdv_asr_module.data.audio import AudioReader
from mdv_asr_module.data.records import SampleRecord
from mdv_asr_module.data.sources import DatasetSource
from mdv_asr_module.runtime.distributed import DistributedContext

from .metrics import ErrorCounter
from .records import EvaluationResult, PredictionRecord


class ASREvaluator:
    def __init__(self, reader: AudioReader | None = None,
                 counter: ErrorCounter | None = None,
                 context: DistributedContext | None = None):
        self.reader = reader or AudioReader()
        self.counter = counter or ErrorCounter()
        self.context = context or DistributedContext()

    def evaluate_one(self, backend, record: SampleRecord, source: DatasetSource) -> PredictionRecord:
        start = time.perf_counter()
        try:
            audio = self.reader.read(record, source)
            result = backend.transcribe(audio, backend.decode_config)
            
            return PredictionRecord(
                dataset_id=record.dataset_id, sample_id=record.sample_id,
                model_key=backend.spec.key, reference=record.text,
                prediction=result.text, split=record.split, filename=record.filename,
                region=record.region, province_name=record.province_name,
                speaker_id=record.speaker_id, inference_seconds=result.elapsed_seconds,
                metadata={"backend": backend.spec.backend, **result.metadata},
            )
        except Exception as exc:
            
            return PredictionRecord(
                dataset_id=record.dataset_id, sample_id=record.sample_id,
                model_key=backend.spec.key, reference=record.text, prediction="",
                split=record.split, filename=record.filename, region=record.region,
                province_name=record.province_name, speaker_id=record.speaker_id,
                inference_seconds=time.perf_counter() - start, error=repr(exc),
            )

    def evaluate_model(self, backend, records, source: DatasetSource) -> EvaluationResult:
        records = list(records)
        
        if self.context.is_distributed:
            local_records = records[self.context.rank::self.context.world_size]
        else:
            local_records = records
        
        local = [self.evaluate_one(backend, record, source) for record in local_records]
        chunks = self.context.gather_objects(local)
        merged = {}
        
        for chunk in chunks:
            for row in chunk:
                merged[row.key] = row
        predictions = sorted(merged.values(), key=lambda row: row.key)
        expected = {(record.dataset_id, record.sample_id, backend.spec.key) for record in records}
        result = EvaluationResult(predictions, self.counter.sum_counts(
            self.counter.count(row.reference, row.prediction)
            for row in predictions if not row.error
        ), expected)
        return result

    def validate_coverage(self, result: EvaluationResult, *, allow_errors: bool = False) -> None:
        observed = {row.key for row in result.predictions}
        missing = result.expected_keys - observed
        unexpected = observed - result.expected_keys
        errors = [row.key for row in result.predictions if row.error]
        if missing or unexpected or (errors and not allow_errors):
            raise IncompleteEvaluationError(
                f"Evaluation incomplete: missing={len(missing)}, unexpected={len(unexpected)}, errors={len(errors)}"
            )

    def merge_distributed(self, local_predictions, expected_keys):
        chunks = self.context.gather_objects(local_predictions)
        merged = {row.key: row for chunk in chunks for row in chunk}
        expected = set(expected_keys)
        if set(merged) != expected:
            raise IncompleteEvaluationError("Distributed prediction IDs do not match expected IDs")
        return sorted(merged.values(), key=lambda row: row.key)

