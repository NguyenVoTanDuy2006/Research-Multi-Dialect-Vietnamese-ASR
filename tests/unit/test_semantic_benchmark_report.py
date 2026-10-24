from __future__ import annotations

import csv
from dataclasses import dataclass

from mdv_asr_module.configs.types import SemanticMetricsConfig
from mdv_asr_module.evaluation import SemanticBenchmarkReporter
from mdv_asr_module.evaluation.records import PredictionRecord


@dataclass
class _E5Score:
    similarity: float
    distance: float
    chunked: bool = False


@dataclass
class _BertScore:
    precision: float
    recall: float
    f1: float
    chunked: bool = False


class _FakeMetric:
    def __init__(self, metric_name, calls, revision):
        self.metric_name = metric_name
        self.calls = calls
        self.revision = revision

    def metadata(self):
        return {"name": self.metric_name, "revision": self.revision}

    def score_batch(self, references, predictions):
        self.calls.append(list(zip(references, predictions)))
        output = []
        for reference, prediction in zip(references, predictions):
            value = len(prediction) / len(reference)
            if self.metric_name == "e5":
                output.append(_E5Score(value, 1.0 - value))
            else:
                output.append(_BertScore(value, value, value))
        return output


def _row(sample_id, model, reference, prediction, *, region="North", province="A",
         error=""):
    return PredictionRecord(
        dataset_id="vimd", sample_id=sample_id, model_key=model,
        reference=reference, prediction=prediction, split="test",
        filename=f"{sample_id}.wav", region=region, province_name=province,
        error=error,
    )


def _factories(revision="fake-rev-1"):
    calls = {"e5": [], "bert": []}
    return calls, {
        "semsim_e5": lambda: _FakeMetric("e5", calls["e5"], revision),
        "bertscore": lambda: _FakeMetric("bert", calls["bert"], revision),
    }


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_reporter_scores_unique_pairs_and_maps_them_to_every_model_row(tmp_path):
    calls, factories = _factories()
    rows = [
        _row("1", "base", "xin chao", "xin"),
        _row("1", "adapted", "xin chao", "xin"),
        _row("2", "base", "xin", "xin", region="South", province="B"),
        _row("3", "base", "xin chao", ""),
        _row("4", "base", "xin", "", error="decode failed"),
    ]
    config = SemanticMetricsConfig(enabled=True, batch_size=8)

    outputs = SemanticBenchmarkReporter(
        tmp_path, config, resume=False, metric_factories=factories
    ).write_all(rows)

    assert set(outputs) == {"predictions", "overall", "region", "province", "config"}
    assert outputs["predictions"].name == "semantic_predictions.csv"
    assert outputs["overall"].name == "semantic_metrics_overall.csv"
    assert outputs["region"].name == "semantic_metrics_by_region.csv"
    assert outputs["province"].name == "semantic_metrics_by_province.csv"
    assert len(calls["e5"]) == 2
    assert sum(len(batch) for batch in calls["e5"]) == 2
    assert sum(len(batch) for batch in calls["bert"]) == 2
    predictions = _read_csv(outputs["predictions"])
    assert len(predictions) == 5
    assert predictions[0]["model_key"] == "base"
    assert predictions[1]["model_key"] == "adapted"
    assert predictions[0]["SemSim_E5"] == predictions[1]["SemSim_E5"]
    assert predictions[2]["SemSim_E5"] == "1.0"
    assert predictions[3]["SemSim_E5"] == "0.0"
    assert predictions[3]["SemDist_E5"] == "1.0"
    assert predictions[4]["SemSim_E5"] == ""
    assert "BERTScore_f1" in predictions[0]
    summary = _read_csv(outputs["overall"])
    base_summary = next(row for row in summary if row["model_key"] == "base")
    assert base_summary["n_samples"] == "3"
    assert base_summary["SemSim_E5_chunked_count"] == "0"


def test_reporter_resumes_cached_pairs_without_scoring_them_again(tmp_path):
    rows = [_row("1", "base", "xin chao", "xin")]
    first_calls, first_factories = _factories()
    SemanticBenchmarkReporter(
        tmp_path, SemanticMetricsConfig(enabled=True), resume=False,
        metric_factories=first_factories,
    ).write_all(rows)
    assert sum(len(batch) for batch in first_calls["e5"]) == 1

    resumed_calls, resumed_factories = _factories()
    SemanticBenchmarkReporter(
        tmp_path, SemanticMetricsConfig(enabled=True), resume=True,
        metric_factories=resumed_factories,
    ).write_all(rows)

    assert resumed_calls == {"e5": [], "bert": []}

    updated_calls, updated_factories = _factories(revision="fake-rev-2")
    SemanticBenchmarkReporter(
        tmp_path, SemanticMetricsConfig(enabled=True), resume=True,
        metric_factories=updated_factories,
    ).write_all(rows)
    assert sum(len(batch) for batch in updated_calls["e5"]) == 1


def test_empty_successes_are_scored_without_loading_metrics_and_errors_are_excluded(tmp_path):
    def forbidden_factory():
        raise AssertionError("A semantic model should not load for empty hypotheses only")

    config = SemanticMetricsConfig(enabled=True, metrics=("semsim_e5",))
    outputs = SemanticBenchmarkReporter(
        tmp_path, config, resume=False,
        metric_factories={"semsim_e5": forbidden_factory},
    ).write_all([
        _row("1", "base", "xin chao", ""),
        _row("2", "base", "xin chao", "", error="decode failed"),
    ])

    rows = _read_csv(outputs["predictions"])
    assert rows[0]["SemSim_E5"] == "0.0"
    assert rows[0]["SemDist_E5"] == "1.0"
    assert rows[1]["SemSim_E5"] == ""
    summary = _read_csv(outputs["overall"])
    assert summary[0]["n_samples"] == "1"
    assert summary[0]["SemSim_E5_mean"] == "0.0"
