"""WER/CER sentence counters and corpus-level error aggregation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from mdv_asr_module.core.errors import DatasetIntegrityError

from .helper import TextNormalizer, distribution_summary, edit_distance, validate_text_pair
from .metric import Metric
from .records import ErrorCounts, PredictionRecord


@dataclass(frozen=True)
class MetricCount:
    """Error numerator and reference denominator for one metric."""

    errors: int
    reference_units: int

    @property
    def rate(self) -> float:
        return self.errors / self.reference_units if self.reference_units else 0.0


class WERCounter:
    """Compute word-level edit errors and reference-word count."""

    def __init__(self, normalizer: TextNormalizer | None = None):
        self.normalizer = normalizer or TextNormalizer()

    def count(self, reference: str, prediction: str) -> MetricCount:
        reference_words = self.normalizer.tokens(reference)
        prediction_words = self.normalizer.tokens(prediction)
        if not reference_words:
            raise DatasetIntegrityError("Reference has no words after evaluation normalization")
        return MetricCount(edit_distance(reference_words, prediction_words), len(reference_words))


class CERCounter:
    """Compute character-level edit errors, excluding spaces."""

    def __init__(self, normalizer: TextNormalizer | None = None):
        self.normalizer = normalizer or TextNormalizer()

    def count(self, reference: str, prediction: str) -> MetricCount:
        reference_chars = self.normalizer.characters(reference)
        prediction_chars = self.normalizer.characters(prediction)
        if not reference_chars:
            raise DatasetIntegrityError("Reference has no characters after evaluation normalization")
        return MetricCount(edit_distance(reference_chars, prediction_chars), len(reference_chars))


class WERMetric(Metric[MetricCount]):
    """Sentence-level WER score plus macro distribution and corpus aggregation."""

    name = "WER"

    def __init__(self, counter: WERCounter | None = None):
        self.counter = counter or WERCounter()

    def score(self, reference: str, prediction: str) -> MetricCount:
        reference, prediction = validate_text_pair(reference, prediction)
        return self.counter.count(reference, prediction)

    def aggregate(self, scores: Sequence[MetricCount]) -> dict[str, Any]:
        scores = list(scores)
        distribution = distribution_summary(score.rate for score in scores)
        errors = sum(score.errors for score in scores)
        units = sum(score.reference_units for score in scores)
        return {
            "n_samples": len(scores), "wer_errors": errors, "reference_words": units,
            "WER": errors / units if units else None,
            "WER_sentence_mean": distribution["mean"],
            "WER_sentence_median": distribution["median"],
            "WER_sentence_p05": distribution["p05"],
        }

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "normalization": self.counter.normalizer.version}


class CERMetric(Metric[MetricCount]):
    """Sentence-level CER score plus macro distribution and corpus aggregation."""

    name = "CER"

    def __init__(self, counter: CERCounter | None = None):
        self.counter = counter or CERCounter()

    def score(self, reference: str, prediction: str) -> MetricCount:
        reference, prediction = validate_text_pair(reference, prediction)
        return self.counter.count(reference, prediction)

    def aggregate(self, scores: Sequence[MetricCount]) -> dict[str, Any]:
        scores = list(scores)
        distribution = distribution_summary(score.rate for score in scores)
        errors = sum(score.errors for score in scores)
        units = sum(score.reference_units for score in scores)
        return {
            "n_samples": len(scores), "cer_errors": errors, "reference_chars": units,
            "CER": errors / units if units else None,
            "CER_sentence_mean": distribution["mean"],
            "CER_sentence_median": distribution["median"],
            "CER_sentence_p05": distribution["p05"],
        }

    def metadata(self) -> dict[str, Any]:
        return {"name": self.name, "normalization": self.counter.normalizer.version}


class ErrorCounter:
    """Compatibility facade that combines separate WER and CER counters."""

    def __init__(self, normalizer: TextNormalizer | None = None,
                 wer_counter: WERCounter | None = None,
                 cer_counter: CERCounter | None = None):
        normalizer = normalizer or TextNormalizer()
        self.wer_counter = wer_counter or WERCounter(normalizer)
        self.cer_counter = cer_counter or CERCounter(normalizer)

    def count(self, reference: str, prediction: str) -> ErrorCounts:
        wer = self.wer_counter.count(reference, prediction)
        cer = self.cer_counter.count(reference, prediction)
        return ErrorCounts(word_errors=wer.errors, reference_words=wer.reference_units,
                           char_errors=cer.errors, reference_chars=cer.reference_units)

    @staticmethod
    def sum_counts(counts: Iterable[ErrorCounts]) -> ErrorCounts:
        total = ErrorCounts()
        for item in counts:
            total += item
        return total


class MetricAggregator:
    """Compatibility corpus aggregator used by existing reports/workflows."""

    def __init__(self, counter: ErrorCounter | None = None):
        self.counter = counter or ErrorCounter()

    def overall(self, predictions: Iterable[PredictionRecord]) -> dict:
        rows = list(predictions)
        wer = self.wer(rows)
        cer = self.cer(rows)
        return {"n_samples": len(rows),
                "wer_errors": wer["wer_errors"], "reference_words": wer["reference_words"],
                "WER": wer["WER"],
                "cer_errors": cer["cer_errors"], "reference_chars": cer["reference_chars"],
                "CER": cer["CER"]}

    def wer(self, predictions: Iterable[PredictionRecord]) -> dict:
        """Aggregate only word-level errors and reference-word denominator."""
        rows = list(predictions)
        counts = [self.counter.wer_counter.count(row.reference, row.prediction) for row in rows]
        errors = sum(value.errors for value in counts)
        units = sum(value.reference_units for value in counts)
        return {"n_samples": len(rows), "wer_errors": errors,
                "reference_words": units, "WER": errors / units if units else 0.0}

    def cer(self, predictions: Iterable[PredictionRecord]) -> dict:
        """Aggregate only character-level errors and reference-character denominator."""
        rows = list(predictions)
        counts = [self.counter.cer_counter.count(row.reference, row.prediction) for row in rows]
        errors = sum(value.errors for value in counts)
        units = sum(value.reference_units for value in counts)
        return {"n_samples": len(rows), "cer_errors": errors,
                "reference_chars": units, "CER": errors / units if units else 0.0}

    def by_region(self, predictions: Iterable[PredictionRecord]) -> dict[str, dict]:
        return self._group(predictions, lambda row: row.region or "Unknown")

    def by_province(self, predictions: Iterable[PredictionRecord]) -> dict[str, dict]:
        return self._group(predictions, lambda row: row.province_name or "Unknown")

    def _group(self, predictions: Iterable[PredictionRecord], key_fn) -> dict[str, dict]:
        grouped = defaultdict(list)
        for row in predictions:
            grouped[key_fn(row)].append(row)
        return {key: self.overall(rows) for key, rows in sorted(grouped.items())}

    def compare_models(self, predictions: Iterable[PredictionRecord], baseline_key: str) -> dict:
        groups = defaultdict(list)
        for row in predictions:
            groups[row.model_key].append(row)
        if baseline_key not in groups:
            raise KeyError(f"Baseline model not found: {baseline_key}")
        expected = {row.key[:2] for row in groups[baseline_key] if not row.error}
        if not expected:
            raise DatasetIntegrityError(f"Baseline model {baseline_key} has no successful predictions")
        report = {}
        for model_key, rows in groups.items():
            successful = {row.key[:2] for row in rows if not row.error}
            if successful != expected:
                raise DatasetIntegrityError(
                    f"Model {model_key} does not cover the same successful sample IDs as {baseline_key}"
                )
            report[model_key] = self.overall([row for row in rows if not row.error])
        return report

    def sentence_macro(self, predictions: Iterable[PredictionRecord]) -> dict:
        rows = list(predictions)
        wer_counts = [self.counter.wer_counter.count(row.reference, row.prediction) for row in rows]
        cer_counts = [self.counter.cer_counter.count(row.reference, row.prediction) for row in rows]
        return {"n_samples": len(rows),
                "macro_wer": sum(value.rate for value in wer_counts) / len(wer_counts) if wer_counts else 0.0,
                "macro_cer": sum(value.rate for value in cer_counts) / len(cer_counts) if cer_counts else 0.0}
