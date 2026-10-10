from __future__ import annotations

from pathlib import Path

from mdv_asr.core.artifacts import ArtifactStore
from mdv_asr.core.errors import DatasetIntegrityError

from .metrics import MetricAggregator
from .records import EvaluationResult


class BenchmarkReporter:
    def __init__(self, output_dir: str | Path, aggregator: MetricAggregator | None = None):
        self.output_dir = Path(output_dir)
        self.aggregator = aggregator or MetricAggregator()
        self.store = ArtifactStore(self.output_dir)

    def write_predictions(self, result: EvaluationResult) -> Path:
        return self.store.write_csv("predictions.csv", [row.to_dict() for row in result.predictions])

    def write_all(self, results: list[EvaluationResult], baseline_key: str | None = None) -> dict[str, Path]:
        predictions = [row for result in results for row in result.predictions]
        expected = set().union(*(result.expected_keys for result in results)) if results else set()
        counts = self.aggregator.counter.sum_counts(
            self.aggregator.counter.count(row.reference, row.prediction)
            for row in predictions if not row.error
        )
        combined = EvaluationResult(predictions, counts, expected)
        outputs = {"predictions": self.write_predictions(combined),
                   "overall": self.write_overall_metrics(combined),
                   "region": self.write_region_metrics(combined),
                   "province": self.write_province_metrics(combined),
                   "coverage": self.write_coverage_report(combined)}
        if baseline_key:
            try:
                outputs["comparison"] = self.write_comparison(combined, baseline_key)
            except (KeyError, DatasetIntegrityError, ValueError) as exc:
                outputs["comparison"] = self.store.write_json("comparison.json", {
                    "status": "incomplete", "reason": str(exc),
                    "note": "Predictions and per-model metrics were saved; compare only paired successful samples.",
                })
        return outputs

    def write_overall_metrics(self, result: EvaluationResult) -> Path:
        models = sorted({row.model_key for row in result.predictions})
        rows = [{"model": key, **self.aggregator.overall(
            row for row in result.predictions if row.model_key == key and not row.error
        )} for key in models]
        return self.store.write_csv("metrics_overall.csv", rows)

    def write_region_metrics(self, result: EvaluationResult) -> Path:
        rows = []
        for model in sorted({row.model_key for row in result.predictions}):
            groups = self.aggregator.by_region(row for row in result.predictions
                                               if row.model_key == model and not row.error)
            rows.extend({"model": model, "region": name, **metrics} for name, metrics in groups.items())
        fields = ["model", "region", "n_samples", "wer_errors", "reference_words",
                  "WER", "cer_errors", "reference_chars", "CER"]
        return self.store.write_csv("metrics_by_region.csv", rows, fieldnames=fields)

    def write_province_metrics(self, result: EvaluationResult) -> Path:
        rows = []
        for model in sorted({row.model_key for row in result.predictions}):
            groups = self.aggregator.by_province(row for row in result.predictions
                                                 if row.model_key == model and not row.error)
            rows.extend({"model": model, "province": name, **metrics} for name, metrics in groups.items())
        fields = ["model", "province", "n_samples", "wer_errors", "reference_words",
                  "WER", "cer_errors", "reference_chars", "CER"]
        return self.store.write_csv("metrics_by_province.csv", rows, fieldnames=fields)

    def write_comparison(self, result: EvaluationResult, baseline_key: str) -> Path:
        rows_by_model = {}
        for row in result.predictions:
            rows_by_model.setdefault(row.model_key, []).append(row)
        if baseline_key not in rows_by_model:
            raise KeyError(f"Baseline model {baseline_key!r} has no prediction rows")
        baseline_ids = {row.key[:2] for row in rows_by_model[baseline_key] if not row.error}
        for model_key, rows in rows_by_model.items():
            successful_ids = {row.key[:2] for row in rows if not row.error}
            if successful_ids != baseline_ids:
                raise DatasetIntegrityError(
                    f"Model {model_key} does not cover the same successful sample IDs as {baseline_key}"
                )
        metrics = self.aggregator.compare_models(
            [row for row in result.predictions if not row.error], baseline_key
        )
        return self.store.write_json("comparison.json", metrics)

    def write_coverage_report(self, result: EvaluationResult) -> Path:
        errors = [row.to_dict() for row in result.predictions if row.error]
        return self.store.write_json("coverage.json", {
            "coverage": result.coverage, "expected": len(result.expected_keys),
            "predicted": len(result.predictions), "errors": errors,
        })

