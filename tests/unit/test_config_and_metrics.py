from __future__ import annotations

import json

from mdv_asr_module.config.loader import ConfigLoader
from mdv_asr_module.config.types import BenchmarkConfig, ModelSpec
from mdv_asr_module.evaluation.alignment import WordAligner
from mdv_asr_module.evaluation.metrics import ErrorCounter, MetricAggregator
from mdv_asr_module.evaluation.records import PredictionRecord
from mdv_asr_module.evaluation.normalization import TextNormalizer


def test_config_loader_round_trip_and_unknown_field(tmp_path):
    config_data = {
        "dataset": {"kind": "package", "root": str(tmp_path), "dataset_id": "unit"},
        "model": {"key": "qwen", "backend": "qwen3_asr", "model_id": "Qwen/test"},
        "train": {"epochs": 2, "adalora": {"target_modules": ["q_proj", "v_proj"]}},
        "output_dir": str(tmp_path / "run"),
    }
    loader = ConfigLoader()
    config = loader.from_dict(config_data)
    config.validate()
    output = loader.save_resolved(config, tmp_path / "resolved.json")
    restored = loader.load(output)
    assert restored.train.adalora.target_modules == ("q_proj", "v_proj")
    selected = loader.load_sections(output, ["dataset", "model"])
    assert selected.dataset.dataset_id == "unit"
    assert selected.model.key == "qwen"
    try:
        loader.from_dict({**config_data, "typo": True})
    except ValueError as exc:
        assert "Unknown" in str(exc)
    else:
        raise AssertionError("Unknown config key was accepted")


def test_benchmark_config_rejects_duplicate_models():
    model = ModelSpec("same", "qwen3_asr", "Qwen/test")
    try:
        BenchmarkConfig(models=(model, model)).validate()
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("Duplicate model keys were accepted")


def test_normalization_and_corpus_metrics_vietnamese():
    normalizer = TextNormalizer()
    assert normalizer.normalize("  Đường,   đi! ") == "đường đi"
    counts = ErrorCounter().count("xin chào bạn", "xin chào")
    assert counts.word_errors == 1
    assert counts.reference_words == 3
    assert counts.char_errors == 3


def test_display_alignment_maps_punctuation_to_whitespace_tokens():
    status = WordAligner().displayed_prediction_status("xin chào", "xin, chào nhé")
    assert status == [True, True, False]
    spans = WordAligner().prediction_spans("xin chào", "xin, chào nhé")
    assert spans == [("xin", True), ("chào", True), ("nhé", False)]


def test_model_comparison_requires_paired_successes():
    rows = [
        PredictionRecord("vimd", "a", "base", "xin chào", "xin chào", "test", "a.wav"),
        PredictionRecord("vimd", "a", "adapted", "xin chào", "xin", "test", "a.wav"),
    ]
    compared = MetricAggregator().compare_models(rows, "base")
    assert compared["base"]["CER"] == 0
    assert compared["adapted"]["WER"] == 0.5
