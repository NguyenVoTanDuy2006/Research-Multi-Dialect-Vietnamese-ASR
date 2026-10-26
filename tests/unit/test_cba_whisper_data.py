from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

CBA_ROOT = Path(__file__).resolve().parents[2] / "src" / "CBA-pipeline" / "Reimplement-CBA"
if str(CBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CBA_ROOT))

from reimplement_cba.adapters.whisper_data import PhoWhisperDataCollator  # noqa: E402


class PhoWhisperDataCollatorTests(unittest.TestCase):
    def test_casts_audio_features_to_model_dtype_without_casting_labels(self):
        processor = SimpleNamespace(
            feature_extractor=SimpleNamespace(
                pad=lambda features, return_tensors: {
                    "input_features": torch.tensor([[1.0, 2.0]], dtype=torch.float32)
                }
            ),
            tokenizer=SimpleNamespace(
                pad=lambda features, return_tensors: {
                    "input_ids": torch.tensor([[7, 8]], dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                }
            ),
        )
        collator = PhoWhisperDataCollator(
            processor,
            decoder_start_token_id=7,
            input_features_dtype=torch.float16,
        )

        batch = collator([{"input_features": [1.0, 2.0], "labels": [7, 8]}])

        self.assertEqual(batch["input_features"].dtype, torch.float16)
        self.assertEqual(batch["labels"].dtype, torch.long)
        self.assertEqual(batch["labels"].tolist(), [[8]])


if __name__ == "__main__":
    unittest.main()
