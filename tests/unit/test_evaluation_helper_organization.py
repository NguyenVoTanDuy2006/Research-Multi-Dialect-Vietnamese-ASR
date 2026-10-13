from __future__ import annotations

import unittest
from importlib.util import find_spec

from mdv_asr_module.evaluation.helper import (
    NormalizedText,
    TextNormalizer,
    WordAligner,
)


class EvaluationHelperOrganizationTests(unittest.TestCase):
    def test_obsolete_evaluation_facades_are_removed(self):
        package = "mdv_asr_module.evaluation"
        for module in ("alignment", "normalization", "metrics"):
            with self.subTest(module=module):
                self.assertIsNone(find_spec(f"{package}.{module}"))

    def test_helper_normalizes_and_aligns_transcripts(self):
        normalizer = TextNormalizer()
        self.assertEqual(normalizer.normalize("  Đường,   đi! "), "đường đi")
        self.assertEqual(
            normalizer.normalize_with_spans("Đường, đi!"),
            NormalizedText("đường đi", ("đường", "đi"), ("đ", "ư", "ờ", "n", "g", "đ", "i")),
        )

        steps = WordAligner(normalizer).align("xin chào bạn", "xin chào")
        self.assertEqual(
            [(step.operation, step.reference_token, step.prediction_token) for step in steps],
            [("match", "xin", "xin"), ("match", "chào", "chào"), ("deletion", "bạn", None)],
        )


if __name__ == "__main__":
    unittest.main()
