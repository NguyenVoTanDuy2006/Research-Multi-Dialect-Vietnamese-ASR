from __future__ import annotations

import json
from unittest.mock import patch

from mdv_asr_module.evaluation.semantic_store import SemanticScoreStore


def test_semantic_score_store_round_trips_batches_and_filters_fingerprint(tmp_path):
    store = SemanticScoreStore(tmp_path / "cache" / "scores.jsonl")
    store.append_batch("e5-r1", {"pair-a": {"similarity": 0.8}})
    store.append_batch("bert-r1", {"pair-a": {"f1": 0.7}})
    store.append_batch("e5-r1", {"pair-b": {"similarity": 0.9}})

    assert store.load("e5-r1") == {
        "pair-a": {"similarity": 0.8}, "pair-b": {"similarity": 0.9},
    }
    assert store.load("bert-r1") == {"pair-a": {"f1": 0.7}}
    assert store.load("new-model-revision") == {}


def test_later_batch_overwrites_a_score_for_the_same_pair(tmp_path):
    store = SemanticScoreStore(tmp_path / "scores.jsonl")
    store.append_batch("e5", {"pair-a": {"similarity": 0.2}})
    store.append_batch("e5", {"pair-a": {"similarity": 0.95}})

    assert store.load("e5")["pair-a"] == {"similarity": 0.95}


def test_truncated_final_jsonl_line_keeps_completed_batches(tmp_path):
    path = tmp_path / "scores.jsonl"
    store = SemanticScoreStore(path)
    store.append_batch("e5", {"pair-a": {"similarity": 0.8}})
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"metric_fingerprint":"e5","scores":')

    assert store.load("e5") == {"pair-a": {"similarity": 0.8}}


def test_append_repairs_a_truncated_tail_before_writing_next_batch(tmp_path):
    path = tmp_path / "scores.jsonl"
    store = SemanticScoreStore(path)
    store.append_batch("e5", {"pair-a": {"similarity": 0.8}})
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"metric_fingerprint":"e5","scores":')

    store.append_batch("e5", {"pair-b": {"similarity": 0.9}})

    assert store.load("e5") == {
        "pair-a": {"similarity": 0.8}, "pair-b": {"similarity": 0.9},
    }


def test_repeated_appends_do_not_rescan_the_entire_cache(tmp_path):
    store = SemanticScoreStore(tmp_path / "scores.jsonl")
    store.append_batch("e5", {"pair-a": {"similarity": 0.8}})

    with patch.object(store, "_read_batches", side_effect=AssertionError("full cache rescan")):
        store.append_batch("e5", {"pair-b": {"similarity": 0.9}})

    assert store.load("e5") == {
        "pair-a": {"similarity": 0.8}, "pair-b": {"similarity": 0.9},
    }


def test_malformed_interior_jsonl_line_is_reported(tmp_path):
    path = tmp_path / "scores.jsonl"
    valid = json.dumps({"metric_fingerprint": "e5", "scores": {"pair": {"score": 1}}})
    path.write_text(valid + "\nnot-json\n" + valid + "\n", encoding="utf-8")

    try:
        SemanticScoreStore(path).load("e5")
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("Malformed interior JSONL row was ignored")


def test_first_append_validates_and_rejects_an_interior_corruption(tmp_path):
    path = tmp_path / "scores.jsonl"
    valid = json.dumps({"metric_fingerprint": "e5", "scores": {"pair": {"score": 1}}})
    path.write_text(valid + "\nnot-json\n" + valid + "\n", encoding="utf-8")

    try:
        SemanticScoreStore(path).append_batch("e5", {"next": {"score": 2}})
    except ValueError as exc:
        assert "line 2" in str(exc)
    else:
        raise AssertionError("First append ignored an interior cache corruption")
