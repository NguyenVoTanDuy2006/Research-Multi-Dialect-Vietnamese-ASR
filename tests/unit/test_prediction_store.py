from __future__ import annotations

from mdv_asr_module.evaluation.records import PredictionRecord
from mdv_asr_module.evaluation.records import ErrorCounts, EvaluationResult
from mdv_asr_module.evaluation.reports import BenchmarkReporter
from mdv_asr_module.evaluation.store import PredictionStore


def test_prediction_store_flush_reload_and_retry_errors(tmp_path):
    store = PredictionStore(tmp_path / "predictions.jsonl", flush_every=2)
    store.open({"fingerprint": "one"}, resume=False)
    good = PredictionRecord("vimd", "s1", "base", "xin", "xin", "test", "s1.wav")
    failed = PredictionRecord("vimd", "s2", "base", "xin", "", "test", "s2.wav", error="temporary")
    store.upsert(good)
    store.upsert(failed)
    assert store.completed_keys("base") == {good.key}
    restarted = PredictionStore(tmp_path / "predictions.jsonl")
    restarted.open({"fingerprint": "one"}, resume=True)
    assert restarted.load_all() == [good, failed]
    records = [type("Record", (), {"dataset_id": "vimd", "sample_id": sample_id})()
               for sample_id in ("s1", "s2")]
    assert restarted.pending(records, "base")[0].sample_id == "s2"


def test_incomplete_benchmark_still_writes_coverage_and_metrics(tmp_path):
    base = PredictionRecord("vimd", "s1", "base", "xin chào", "xin chào",
                            "test", "s1.wav")
    failed = PredictionRecord("vimd", "s1", "adapted", "xin chào", "",
                              "test", "s1.wav", error="decode failed")
    expected = {base.key, failed.key}
    result = EvaluationResult([base, failed], ErrorCounts(), expected)
    assert result.coverage == 0.5
    paths = BenchmarkReporter(tmp_path).write_all(
        [EvaluationResult([base], ErrorCounts(), {base.key}),
         EvaluationResult([failed], ErrorCounts(), {failed.key})],
        baseline_key="base",
    )
    assert all(path.is_file() for path in paths.values())
    import json

    comparison = json.loads(paths["comparison"].read_text(encoding="utf-8"))
    assert comparison["status"] == "incomplete"

    all_failed = [
        PredictionRecord("vimd", "s1", model, "xin chào", "", "test", "s1.wav",
                         error="decode failed")
        for model in ("base", "adapted")
    ]
    failed_paths = BenchmarkReporter(tmp_path / "all_failed").write_all(
        [EvaluationResult([row], ErrorCounts(), {row.key}) for row in all_failed],
        baseline_key="base",
    )
    assert all(path.is_file() for path in failed_paths.values())
    all_failed_comparison = json.loads(
        failed_paths["comparison"].read_text(encoding="utf-8")
    )
    assert all_failed_comparison["status"] == "incomplete"
