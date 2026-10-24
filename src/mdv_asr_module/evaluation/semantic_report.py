"""Batch semantic scoring and reports for ASR benchmark predictions."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from mdv_asr_module.configs.types import SemanticMetricsConfig
from mdv_asr_module.core.artifacts import ArtifactStore
from mdv_asr_module.evaluation.helper import distribution_summary
from mdv_asr_module.evaluation.records import PredictionRecord

from .semantic_store import SemanticScoreStore


_METRIC_COLUMNS = {
    "semsim_e5": {
        "similarity": "SemSim_E5",
        "distance": "SemDist_E5",
        "chunked": "SemSim_E5_chunked",
    },
    "bertscore": {
        "precision": "BERTScore_precision",
        "recall": "BERTScore_recall",
        "f1": "BERTScore_f1",
        "chunked": "BERTScore_chunked",
    },
}
_REQUIRED_SCORE_FIELDS = {
    "semsim_e5": {"similarity", "distance"},
    "bertscore": {"precision", "recall", "f1"},
}


class SemanticBenchmarkReporter:
    """Score unique successful transcript pairs and write sample/group reports."""

    def __init__(
        self,
        output_dir: str | Path,
        config: SemanticMetricsConfig,
        *,
        resume: bool,
        metric_factories: Mapping[str, Callable[[], Any]] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.config = config
        self.resume = bool(resume)
        self.metric_factories = metric_factories
        self.artifacts = ArtifactStore(self.output_dir)
        cache_path = Path(config.cache_path)
        if not cache_path.is_absolute():
            cache_path = self.output_dir / cache_path
        self.score_store = SemanticScoreStore(cache_path)

    @staticmethod
    def _pair_hash(reference: str, prediction: str) -> str:
        encoded = json.dumps(
            [reference, prediction], ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _default_factories(config: SemanticMetricsConfig) -> dict[str, Callable[[], Any]]:
        def make_e5():
            from .similarity_metrics import SemSimE5

            return SemSimE5(
                device=config.device, batch_size=config.batch_size,
                max_length=config.max_length, long_text_policy=config.long_text_policy,
            )

        def make_bertscore():
            from .similarity_metrics import BERTScoreMetric

            return BERTScoreMetric(
                device=config.device, batch_size=config.batch_size,
                max_length=config.max_length, long_text_policy=config.long_text_policy,
            )

        return {"semsim_e5": make_e5, "bertscore": make_bertscore}

    @staticmethod
    def _empty_score(metric_key: str) -> dict[str, object]:
        if metric_key == "semsim_e5":
            return {"similarity": 0.0, "distance": 1.0, "chunked": False}
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "chunked": False}

    @staticmethod
    def _metric_metadata_without_load(metric_key: str, config: SemanticMetricsConfig) -> dict[str, Any]:
        if metric_key == "semsim_e5":
            return {
                "name": "SemSim-E5", "model": "intfloat/multilingual-e5-base",
                "revision": None, "device": config.device,
                "batch_size": config.batch_size, "max_tokens": config.max_length,
                "long_text_policy": config.long_text_policy,
                "status": "not_loaded_no_nonempty_pairs",
            }
        return {
            "name": "BERTScore", "model": "FacebookAI/xlm-roberta-large",
            "revision": None, "device": config.device,
            "batch_size": config.batch_size, "max_tokens": config.max_length,
            "long_text_policy": config.long_text_policy,
            "status": "not_loaded_no_nonempty_pairs",
        }

    @staticmethod
    def _score_dict(score: Any, metric_key: str) -> dict[str, object]:
        if is_dataclass(score):
            values = asdict(score)
        elif isinstance(score, Mapping):
            values = dict(score)
        elif hasattr(score, "__dict__"):
            values = vars(score).copy()
        else:
            raise TypeError(f"{metric_key} returned unsupported score type {type(score).__name__}")

        missing = _REQUIRED_SCORE_FIELDS[metric_key] - values.keys()
        if missing:
            raise ValueError(f"{metric_key} score is missing fields: {sorted(missing)}")
        values.setdefault("chunked", False)
        normalized: dict[str, object] = {}
        for name, value in values.items():
            if hasattr(value, "item"):
                value = value.item()
            if name == "chunked":
                normalized[name] = bool(value)
                continue
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{metric_key} returned non-numeric {name}: {value!r}") from exc
            if not math.isfinite(number):
                raise ValueError(f"{metric_key} returned non-finite {name}: {number}")
            normalized[name] = number
        return normalized

    @staticmethod
    def _fingerprint(metric_key: str, config: SemanticMetricsConfig,
                     metadata: Mapping[str, Any]) -> str:
        settings = {
            "metric": metric_key,
            "device": config.device,
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "long_text_policy": config.long_text_policy,
            "metadata": dict(metadata),
        }
        encoded = json.dumps(
            settings, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _release_metric(metric: Any) -> None:
        close = getattr(metric, "close", None)
        if callable(close):
            close()
        for attribute in ("_encoder", "_tokenizer", "_scorer"):
            if hasattr(metric, attribute):
                setattr(metric, attribute, None)
        del metric
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _score_metric(
        self,
        metric_key: str,
        pairs: Mapping[str, tuple[str, str]],
        *,
        load_models: bool,
    ) -> tuple[dict[str, dict[str, object]], dict[str, Any]]:
        if not load_models:
            return {}, self._metric_metadata_without_load(metric_key, self.config)

        factories = self.metric_factories or self._default_factories(self.config)
        if metric_key not in factories:
            raise KeyError(f"No semantic metric factory configured for {metric_key!r}")
        metric = factories[metric_key]()
        started = time.perf_counter()
        try:
            # The existing semantic metrics load their encoder lazily. Resolve its
            # revision before fingerprint lookup so caches cannot cross model revisions.
            load_method = getattr(metric, "_load", None)
            if callable(load_method):
                load_method()
            metadata = dict(metric.metadata())
            fingerprint = self._fingerprint(metric_key, self.config, metadata)
            cached = self.score_store.load(fingerprint) if self.resume else {}
            valid_cached = {
                pair_hash: self._score_dict(score, metric_key)
                for pair_hash, score in cached.items() if pair_hash in pairs
            }
            results = dict(valid_cached)
            missing = [(pair_hash, pair) for pair_hash, pair in pairs.items()
                       if pair_hash not in results]

            if missing:
                # The first missing transcript pair is the scoring smoke check. Its
                # result is persisted before the rest of the dataset is processed.
                pair_hash, (reference, prediction) = missing.pop(0)
                smoke = metric.score_batch([reference], [prediction])
                if len(smoke) != 1:
                    raise ValueError(f"{metric_key} smoke check returned {len(smoke)} scores for 1 pair")
                results[pair_hash] = self._score_dict(smoke[0], metric_key)
                self.score_store.append_batch(fingerprint, {pair_hash: results[pair_hash]})
                completed = len(valid_cached) + 1
                print(f"{metadata.get('name', metric_key)}: {completed:,}/{len(pairs):,} unique pairs", flush=True)

                for start in range(0, len(missing), self.config.batch_size):
                    batch = missing[start:start + self.config.batch_size]
                    scores = metric.score_batch(
                        [pair[0] for _, pair in batch], [pair[1] for _, pair in batch]
                    )
                    if len(scores) != len(batch):
                        raise ValueError(
                            f"{metric_key} returned {len(scores)} scores for {len(batch)} pairs"
                        )
                    batch_scores = {
                        pair_hash: self._score_dict(score, metric_key)
                        for (pair_hash, _), score in zip(batch, scores)
                    }
                    self.score_store.append_batch(fingerprint, batch_scores)
                    results.update(batch_scores)
                    completed += len(batch)
                    print(
                        f"{metadata.get('name', metric_key)}: {completed:,}/{len(pairs):,} unique pairs",
                        flush=True,
                    )

            metadata.update({
                "metric_fingerprint": fingerprint,
                "unique_nonempty_pairs": len(pairs),
                "cached_pairs_reused": len(valid_cached),
                "new_pairs_scored": len(pairs) - len(valid_cached),
                "elapsed_seconds": time.perf_counter() - started,
            })
            return results, metadata
        finally:
            self._release_metric(metric)

    def _prediction_rows(
        self,
        predictions: Sequence[PredictionRecord],
        pair_scores: Mapping[str, Mapping[str, Mapping[str, object]]],
    ) -> list[dict[str, Any]]:
        rows = []
        for prediction in predictions:
            reference = str(prediction.reference or "")
            hypothesis = str(prediction.prediction or "")
            row = {
                "dataset_id": prediction.dataset_id,
                "sample_id": prediction.sample_id,
                "model_key": prediction.model_key,
                "split": prediction.split,
                "filename": prediction.filename,
                "region": prediction.region,
                "province_name": prediction.province_name,
                "speaker_id": prediction.speaker_id,
                "reference": reference,
                "prediction": hypothesis,
                "error": prediction.error,
            }
            pair_hash = self._pair_hash(reference, hypothesis)
            for metric_key in self.config.metrics:
                score = None
                if not prediction.error:
                    score = pair_scores.get(metric_key, {}).get(pair_hash)
                    if score is None and not hypothesis:
                        score = self._empty_score(metric_key)
                for field, column in _METRIC_COLUMNS[metric_key].items():
                    row[column] = None if score is None else score.get(field)
            rows.append(row)
        return rows

    def _summary_rows(self, rows: Sequence[Mapping[str, Any]], group_field: str | None):
        grouped: dict[tuple[str, str | None], list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["error"]:
                continue
            label = str(row[group_field] or "Unknown") if group_field else None
            grouped[(str(row["model_key"]), label)].append(row)

        output = []
        for (model_key, label), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1] or "")):
            result: dict[str, Any] = {"model_key": model_key}
            if group_field:
                result[group_field] = label
            result["n_samples"] = len(group)
            for metric_key in self.config.metrics:
                columns = _METRIC_COLUMNS[metric_key]
                for field, column in columns.items():
                    values = [row[column] for row in group if row[column] is not None]
                    if field == "chunked":
                        result[f"{column}_count"] = sum(bool(value) for value in values)
                        continue
                    summary = distribution_summary(values)
                    for stat, value in summary.items():
                        result[f"{column}_{stat}"] = value
            output.append(result)
        return output

    def write_all(self, predictions: Sequence[PredictionRecord]) -> dict[str, Path]:
        self.config.validate()
        if not self.config.enabled:
            raise ValueError("Semantic metrics are disabled in the configuration")

        unique_pairs: dict[str, tuple[str, str]] = {}
        for prediction in predictions:
            if prediction.error:
                continue
            reference = str(prediction.reference or "")
            hypothesis = str(prediction.prediction or "")
            if hypothesis:
                pair_hash = self._pair_hash(reference, hypothesis)
                unique_pairs.setdefault(pair_hash, (reference, hypothesis))

        pair_scores: dict[str, dict[str, dict[str, object]]] = {}
        metadata: dict[str, Any] = {}
        for metric_key in self.config.metrics:
            scores, metric_metadata = self._score_metric(
                metric_key, unique_pairs, load_models=bool(unique_pairs)
            )
            pair_scores[metric_key] = scores
            metadata[metric_key] = metric_metadata

        rows = self._prediction_rows(predictions, pair_scores)
        sample_fields = [
            "dataset_id", "sample_id", "model_key", "split", "filename", "region",
            "province_name", "speaker_id", "reference", "prediction", "error",
        ]
        for metric_key in self.config.metrics:
            sample_fields.extend(_METRIC_COLUMNS[metric_key].values())
        summary_fields = ["model_key", "n_samples"]
        for metric_key in self.config.metrics:
            for field, column in _METRIC_COLUMNS[metric_key].items():
                if field == "chunked":
                    summary_fields.append(f"{column}_count")
                else:
                    summary_fields.extend(f"{column}_{stat}" for stat in ("mean", "median", "p05"))

        outputs = {
            "predictions": self.artifacts.write_csv(
                "semantic_predictions.csv", rows, fieldnames=sample_fields
            ),
        }
        overall = self._summary_rows(rows, None)
        outputs["overall"] = self.artifacts.write_csv(
            "semantic_metrics_overall.csv", overall,
            fieldnames=summary_fields,
        )
        for key, field, filename in (
            ("region", "region", "semantic_metrics_by_region.csv"),
            ("province", "province_name", "semantic_metrics_by_province.csv"),
        ):
            subgroup_fields = ["model_key", field, "n_samples"] + [
                column for column in summary_fields if column not in {"model_key", "n_samples"}
            ]
            outputs[key] = self.artifacts.write_csv(
                filename, self._summary_rows(rows, field), fieldnames=subgroup_fields
            )

        config_path = self.artifacts.write_json("semantic_metrics_config.json", {
            "enabled": self.config.enabled,
            "metrics": list(self.config.metrics),
            "device": self.config.device,
            "batch_size": self.config.batch_size,
            "max_length": self.config.max_length,
            "long_text_policy": self.config.long_text_policy,
            "cache_path": str(self.score_store.path),
            "resume": self.resume,
            "successful_unique_nonempty_pairs": len(unique_pairs),
            "metadata": metadata,
            "artifacts": {key: path.name for key, path in outputs.items()},
        })
        outputs["config"] = config_path
        return outputs
