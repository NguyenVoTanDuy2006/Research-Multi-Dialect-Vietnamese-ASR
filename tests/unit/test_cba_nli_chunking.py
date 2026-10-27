from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CBA_ROOT = REPO_ROOT / "src" / "CBA-pipeline" / "Reimplement-CBA"
if str(CBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CBA_ROOT))

from reimplement_cba.adapters.nli_chunking import build_nli_chunks  # noqa: E402


class WordTokenizer:
    model_max_length = 8

    def encode(self, text, *, add_special_tokens=False, truncation=False, verbose=False):
        assert not add_special_tokens and not truncation
        return list(range(len(text.split())))

    def num_special_tokens_to_add(self, pair=False):
        assert pair
        return 3


class CharacterTokenizer(WordTokenizer):
    def encode(self, text, *, add_special_tokens=False, truncation=False, verbose=False):
        assert not add_special_tokens and not truncation
        return list(range(len(text.replace(" ", ""))))


class NLIChunkingTests(unittest.TestCase):
    def test_short_pair_stays_whole(self):
        chunks = build_nli_chunks("a b", "a b", WordTokenizer(), max_length=8)
        self.assertEqual([(chunk.reference, chunk.hypothesis) for chunk in chunks],
                         [("a b", "a b")])

    def test_long_pair_preserves_words_and_alignment_after_insertion(self):
        reference = "a b c d e f"
        hypothesis = "a b extra c d e f"
        chunks = build_nli_chunks(reference, hypothesis, WordTokenizer(), max_length=8)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunk.reference for chunk in chunks).split(),
                         reference.split())
        self.assertEqual(" ".join(chunk.hypothesis for chunk in chunks).split(),
                         hypothesis.split())
        for chunk in chunks:
            self.assertEqual(chunk.reference.split(),
                             [word for word in chunk.hypothesis.split() if word != "extra"])
            self.assertLessEqual(
                len(chunk.reference.split()) + len(chunk.hypothesis.split()) + 3, 8
            )

    def test_single_oversized_word_is_split_without_loss(self):
        text = "abcdefghij"
        chunks = build_nli_chunks(text, text, CharacterTokenizer(), max_length=8)

        self.assertGreater(len(chunks), 1)
        self.assertEqual("".join(chunk.reference for chunk in chunks), text)
        self.assertEqual("".join(chunk.hypothesis for chunk in chunks), text)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.reference) + len(chunk.hypothesis) + 3, 8)

    def test_many_chunks_cover_both_transcripts(self):
        reference = " ".join(f"word{i}" for i in range(12))
        chunks = build_nli_chunks(reference, reference, WordTokenizer(), max_length=8)
        self.assertGreater(len(chunks), 2)
        self.assertEqual(" ".join(chunk.reference for chunk in chunks).split(),
                         reference.split())
        self.assertEqual(" ".join(chunk.hypothesis for chunk in chunks).split(),
                         reference.split())
        self.assertTrue(all(len(chunk.reference.split()) + len(chunk.hypothesis.split())
                            + 3 <= 8 for chunk in chunks))

    def test_rejects_length_above_tokenizer_limit(self):
        with self.assertRaisesRegex(ValueError, "model maximum"):
            build_nli_chunks("a", "a", WordTokenizer(), max_length=9)

    def test_tiny_budget_splits_two_single_characters_without_recursion(self):
        chunks = build_nli_chunks("a", "a", CharacterTokenizer(), max_length=4)
        self.assertEqual("".join(chunk.reference for chunk in chunks), "a")
        self.assertEqual("".join(chunk.hypothesis for chunk in chunks), "a")
        self.assertTrue(all(len(chunk.reference) + len(chunk.hypothesis) <= 1
                            for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
