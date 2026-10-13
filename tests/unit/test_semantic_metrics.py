from __future__ import annotations

import unittest

import torch

from mdv_asr_module.evaluation.error_metrics import CERMetric, WERMetric
from mdv_asr_module.evaluation.metric import Metric
from mdv_asr_module.evaluation.similarity_metrics import (
    BERTScoreMetric,
    BERTScoreResult,
    SemSimE5,
    SemSimE5Result,
)


class TinyTokenizer:
    model_max_length = 512

    def num_special_tokens_to_add(self, pair=False):
        del pair
        return 2

    def encode(self, text, add_special_tokens=True, truncation=False):
        del truncation
        ids = [ord(token[0]) % 31 + 3 for token in text.split()]
        return [1, *ids, 2] if add_special_tokens else ids

    def prepare_for_model(self, ids, add_special_tokens=True,
                          return_attention_mask=True,
                          return_special_tokens_mask=False, truncation=False):
        del truncation
        full = [1, *ids, 2] if add_special_tokens else list(ids)
        result = {"input_ids": full}
        if return_attention_mask:
            result["attention_mask"] = [1] * len(full)
        if return_special_tokens_mask:
            result["special_tokens_mask"] = (
                [1, *([0] * len(ids)), 1] if add_special_tokens else [0] * len(ids)
            )
        return result

    def pad(self, features, padding=True, return_tensors="pt"):
        del padding
        width = max(len(feature["input_ids"]) for feature in features)
        values = {}
        for key in features[0]:
            pad_value = 0
            values[key] = [
                feature[key] + [pad_value] * (width - len(feature[key]))
                for feature in features
            ]
        if return_tensors == "pt":
            return {key: torch.tensor(value) for key, value in values.items()}
        return values

    def __call__(self, texts, padding=False, truncation=False,
                 return_tensors=None, add_special_tokens=True):
        del truncation
        rows = [self.encode(text, add_special_tokens=add_special_tokens) for text in texts]
        if not padding:
            return {"input_ids": rows}
        width = max(map(len, rows))
        ids = [row + [0] * (width - len(row)) for row in rows]
        masks = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks)}
        return {"input_ids": ids, "attention_mask": masks}


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vectors = torch.eye(40)
        self.register_buffer("vectors", vectors)
        self.config = type("Config", (), {"_commit_hash": "tiny-revision"})()
        self.seen_ids = set()

    def forward(self, input_ids, attention_mask):
        del attention_mask
        self.seen_ids.update(input_ids.detach().cpu().flatten().tolist())
        return type("Output", (), {"last_hidden_state": self.vectors[input_ids]})()


class TinyBERTScorer:
    def __init__(self):
        self._tokenizer = TinyTokenizer()
        self._model = TinyEncoder()
        self._model.config._commit_hash = "tiny-bert-revision"
        self.device = "cpu"

    def score(self, cands, refs, batch_size, verbose):
        del batch_size, verbose
        count = len(cands)
        return (torch.full((count,), 0.8), torch.full((count,), 0.7),
                torch.full((count,), 0.75))


class SemanticMetricTests(unittest.TestCase):
    def test_metric_is_abstract_and_error_metrics_keep_corpus_rates(self):
        with self.assertRaises(TypeError):
            Metric()

        wer = WERMetric()
        cer = CERMetric()
        wer_score = wer.score("xin chào bạn", "xin chào")
        cer_score = cer.score("xin chào bạn", "xin chào")
        self.assertAlmostEqual(wer_score.rate, 1 / 3)
        self.assertGreater(cer_score.rate, 0)
        self.assertAlmostEqual(wer.aggregate([wer_score])["WER"], wer_score.rate)
        self.assertEqual(wer.metadata()["name"], "WER")

    def test_e5_scores_batches_masks_padding_and_handles_empty_hypothesis(self):
        metric = SemSimE5(
            tokenizer=TinyTokenizer(), encoder=TinyEncoder(), device="cpu", batch_size=2
        )
        scores = metric.score_batch(["a", "b"], ["a", ""])
        self.assertEqual(len(scores), 2)
        self.assertIsInstance(scores[0], SemSimE5Result)
        self.assertAlmostEqual(scores[0].similarity, 1.0, places=6)
        self.assertEqual(scores[1], SemSimE5Result(similarity=0.0, distance=1.0))
        self.assertEqual(metric.metadata()["revision"], "tiny-revision")
        summary = metric.aggregate(scores)
        self.assertEqual(summary["n_samples"], 2)
        self.assertAlmostEqual(summary["SemDist_E5_mean"], 0.5)

    def test_bertscore_batch_and_empty_hypothesis(self):
        metric = BERTScoreMetric(scorer=TinyBERTScorer(), device="cpu", batch_size=2)
        scores = metric.score_batch(["r1", "r2"], ["h1", ""])
        self.assertAlmostEqual(scores[0].precision, 0.8)
        self.assertAlmostEqual(scores[0].recall, 0.7)
        self.assertAlmostEqual(scores[0].f1, 0.75)
        self.assertEqual(scores[1], BERTScoreResult(0.0, 0.0, 0.0))
        self.assertEqual(metric.metadata()["revision"], "tiny-bert-revision")
        self.assertAlmostEqual(metric.aggregate(scores)["BERTScore_F1_mean"], 0.375)

    def test_shared_validation_rejects_nan_empty_reference_and_misaligned_batch(self):
        metric = SemSimE5(tokenizer=TinyTokenizer(), encoder=TinyEncoder(), device="cpu")
        with self.assertRaises(TypeError):
            metric.score(float("nan"), "hypothesis")
        with self.assertRaises(ValueError):
            metric.score("  \t", "hypothesis")
        with self.assertRaises(ValueError):
            metric.score_batch(["ref"], [])

    def test_empty_aggregates_are_none(self):
        e5 = SemSimE5(tokenizer=TinyTokenizer(), encoder=TinyEncoder(), device="cpu")
        bert = BERTScoreMetric(scorer=TinyBERTScorer(), device="cpu")
        self.assertIsNone(e5.aggregate([])["SemSim_E5_mean"])
        self.assertIsNone(bert.aggregate([])["BERTScore_F1_mean"])

    def test_overlong_inputs_raise_instead_of_silent_truncation(self):
        e5 = SemSimE5(
            tokenizer=TinyTokenizer(), encoder=TinyEncoder(), device="cpu", max_length=3
        )
        with self.assertRaisesRegex(ValueError, "instead of truncating"):
            e5.score("a", "a")  # E5 prefix plus special tokens exceeds three tokens.

        bert = BERTScoreMetric(
            scorer=TinyBERTScorer(), device="cpu", max_length=3
        )
        with self.assertRaisesRegex(ValueError, "instead of truncating"):
            bert.score("a b", "a")

    def test_e5_chunk_policy_preserves_tail_and_marks_chunked_pairs(self):
        encoder = TinyEncoder()
        metric = SemSimE5(
            tokenizer=TinyTokenizer(), encoder=encoder, device="cpu",
            batch_size=2, max_length=5, long_text_policy="chunk",
        )
        identical = metric.score("a b c d", "a b c d")
        changed_tail = metric.score("a b c d", "a b c z")

        self.assertTrue(identical.chunked)
        self.assertFalse(metric.score("a b", "a b").chunked)
        self.assertAlmostEqual(identical.similarity, 1.0, places=6)
        self.assertLess(changed_tail.similarity, identical.similarity)
        self.assertIn(32, encoder.seen_ids)  # The final "z" token was encoded.
        self.assertEqual(metric.metadata()["long_text_policy"], "chunk")

    def test_bertscore_chunk_policy_matches_all_content_tokens(self):
        scorer = TinyBERTScorer()
        metric = BERTScoreMetric(
            scorer=scorer, device="cpu", batch_size=2,
            max_length=4, long_text_policy="chunk",
        )
        identical = metric.score("a b c d", "a b c d")
        changed_tail = metric.score("a b c d", "a b c z")

        self.assertTrue(identical.chunked)
        self.assertAlmostEqual(identical.f1, 1.0, places=6)
        self.assertAlmostEqual(changed_tail.precision, 0.75, places=6)
        self.assertAlmostEqual(changed_tail.recall, 0.75, places=6)
        self.assertAlmostEqual(changed_tail.f1, 0.75, places=6)
        self.assertIn(32, scorer._model.seen_ids)  # The final "z" token was encoded.
        self.assertEqual(metric.metadata()["long_text_policy"], "chunk")

    def test_empty_hypothesis_does_not_load_semantic_models(self):
        e5 = SemSimE5()
        bert = BERTScoreMetric()
        self.assertEqual(e5.score("reference", ""), SemSimE5Result(0.0, 1.0))
        self.assertEqual(bert.score("reference", ""), BERTScoreResult(0.0, 0.0, 0.0))
        self.assertIsNone(e5._tokenizer)
        self.assertIsNone(bert._scorer)


if __name__ == "__main__":
    unittest.main()
