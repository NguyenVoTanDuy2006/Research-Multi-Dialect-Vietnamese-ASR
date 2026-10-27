from __future__ import annotations

import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
CBA_ROOT = REPO_ROOT / "src" / "CBA-pipeline" / "Reimplement-CBA"
if str(CBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CBA_ROOT))

from reimplement_cba.adapters.metrics import ViMDMetricsAdapter  # noqa: E402
from reimplement_cba.config import CBAConfig  # noqa: E402


class FakeTensor:
    def __init__(self, shape, values=None):
        self.shape = tuple(shape)
        self.values = values

    def to(self, device):
        del device
        return self

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def __getitem__(self, key):
        rows, column = key
        if rows != slice(None):
            raise IndexError("fake tensor only supports selecting a full column")
        return FakeTensor((self.shape[0],), [row[column] for row in self.values])


class FakeTorch:
    long = "long"

    @staticmethod
    def ones(shape, dtype=None):
        del dtype
        return FakeTensor(shape)

    @staticmethod
    def inference_mode():
        return nullcontext()

    @staticmethod
    def softmax(logits, dim):
        del dim
        return FakeTensor(logits.shape, logits.values)


class PairTokenizer:
    model_max_length = 8

    def __init__(self):
        self.calls = []

    def encode(self, text, *, add_special_tokens=False, truncation=False, verbose=False):
        assert not add_special_tokens and not truncation
        return text.split()

    def num_special_tokens_to_add(self, pair=False):
        assert pair
        return 3

    def __call__(self, premises, claims, **kwargs):
        assert kwargs.get("truncation") is False
        self.calls.extend(zip(premises, claims))
        sequence_length = max(len(premise.split()) + len(claim.split()) + 3
                              for premise, claim in zip(premises, claims))
        shape = (len(premises), sequence_length)
        return {
            "input_ids": FakeTensor(shape, list(zip(premises, claims))),
            "attention_mask": FakeTensor(shape),
        }


class FakeNLIModel:
    config = types.SimpleNamespace(
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        _commit_hash="tiny-revision",
    )

    def __init__(self):
        self.sequence_lengths = []

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        del attention_mask
        self.sequence_lengths.append(input_ids.shape[-1])
        logits = FakeTensor(
            (input_ids.shape[0], 3),
            [[0.2, 0.7, 0.1] if "tail" in (premise + " " + claim).split()
             else [0.8, 0.1, 0.1]
             for premise, claim in input_ids.values],
        )
        return types.SimpleNamespace(logits=logits)


class CbaMetricsTests(unittest.TestCase):
    def _metric(self):
        tokenizer = PairTokenizer()
        model = FakeNLIModel()
        metric = ViMDMetricsAdapter(
            nli_model="tiny-nli", nli_max_length=8, device="cpu",
            nli_tokenizer=tokenizer, nli_model_instance=model,
        )
        return metric, tokenizer, model

    def test_nli_scores_tail_of_overlength_pair_and_averages_chunks(self):
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            metric, tokenizer, model = self._metric()
            scores = metric._score_nli_model(["a b c tail"], ["a b c tail"])

        self.assertAlmostEqual(scores[0], 0.5)
        self.assertEqual(len(tokenizer.calls), 4)
        self.assertTrue(all(length <= 8 for length in model.sequence_lengths))

    def test_nli_scores_short_pair_once(self):
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            metric, tokenizer, _ = self._metric()
            scores = metric._score_nli_model(["a"], ["a"])

        self.assertAlmostEqual(scores[0], 0.8)
        self.assertEqual(tokenizer.calls, [("a", "a"), ("a", "a")])

    def test_nli_penalizes_unmatched_extra_chunk(self):
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            metric, _, _ = self._metric()
            scores = metric._score_nli_model(
                ["a", "a x y z w"], ["a x y z w", "a"]
            )

        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(scores[0], 0.16)
        self.assertAlmostEqual(scores[1], 0.16)

    def test_long_pair_score_is_symmetric_with_repeated_words(self):
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            metric, _, _ = self._metric()
            scores = metric._score_nli_model(
                ["a a a b", "a c b a"], ["a c b a", "a a a b"]
            )

        self.assertAlmostEqual(scores[0], scores[1])

    def test_nli_returns_zero_when_tokenizer_discards_all_content(self):
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            metric, tokenizer, model = self._metric()
            tokenizer.encode = lambda text, **kwargs: []
            scores = metric._score_nli_model(["a"], ["a"])

        self.assertEqual(scores, [0.0])
        self.assertEqual(model.sequence_lengths, [])

    def test_sample_config_enables_bertscore_for_long_transcripts(self):
        config = CBAConfig.from_file(CBA_ROOT / "configs" / "vimd_phowhisper_small.json")
        metric = ViMDMetricsAdapter.from_config(config)
        self.assertEqual(metric.metadata()["bertscore"]["long_text_policy"], "chunk")
        metric.device = "cpu"
        self.assertEqual(metric._get_bertscore_metric().long_text_policy, "chunk")


if __name__ == "__main__":
    unittest.main()
