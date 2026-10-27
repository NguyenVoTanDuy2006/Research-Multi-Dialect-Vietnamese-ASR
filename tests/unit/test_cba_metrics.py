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
        return FakeTensor(
            logits.shape,
            [[0.8, 0.1, 0.1] for _ in range(logits.shape[0])],
        )


class OverlengthPairTokenizer:
    model_max_length = 512

    def __init__(self):
        self.call_kwargs = None

    def __call__(self, premises, claims, **kwargs):
        self.call_kwargs = kwargs
        requested_length = kwargs.get("max_length")
        sequence_length = (
            min(514, requested_length)
            if kwargs.get("truncation") and requested_length is not None
            else 514
        )
        shape = (len(premises), sequence_length)
        return {
            "input_ids": FakeTensor(shape),
            "attention_mask": FakeTensor(shape),
        }


class FakeNLIModel:
    config = types.SimpleNamespace(
        id2label={0: "entailment", 1: "neutral", 2: "contradiction"},
        _commit_hash="tiny-revision",
    )

    def __init__(self):
        self.sequence_length = None

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        del attention_mask
        self.sequence_length = input_ids.shape[-1]
        logits = FakeTensor(
            (input_ids.shape[0], 3),
            [[2.0, 0.0, 0.0] for _ in range(input_ids.shape[0])],
        )
        return types.SimpleNamespace(logits=logits)


class CbaMetricsTests(unittest.TestCase):
    def test_nli_truncates_overlength_pairs_to_configured_limit(self):
        tokenizer = OverlengthPairTokenizer()
        model = FakeNLIModel()
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            metric = ViMDMetricsAdapter(
                nli_model="tiny-nli",
                nli_max_length=512,
                device="cpu",
                nli_tokenizer=tokenizer,
                nli_model_instance=model,
            )
            scores = metric._score_nli_model(["reference"], ["hypothesis"])

        self.assertEqual(tokenizer.call_kwargs["truncation"], True)
        self.assertEqual(tokenizer.call_kwargs["max_length"], 512)
        self.assertEqual(model.sequence_length, 512)
        self.assertEqual(len(scores), 1)
        self.assertGreaterEqual(scores[0], 0.0)
        self.assertLessEqual(scores[0], 1.0)


if __name__ == "__main__":
    unittest.main()
