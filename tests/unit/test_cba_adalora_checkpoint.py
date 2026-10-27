from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

CBA_ROOT = Path(__file__).resolve().parents[2] / "src" / "CBA-pipeline" / "Reimplement-CBA"
if str(CBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CBA_ROOT))

from reimplement_cba.stages.adalora_checkpoint import full_rank_save_config  # noqa: E402
from reimplement_cba.stages.training_schedule import (  # noqa: E402
    resolve_training_world_size,
)


class FullRankSaveConfigTests(unittest.TestCase):
    def test_clears_rank_pattern_during_save_then_restores_it(self):
        configuration = types.SimpleNamespace(rank_pattern={"layer": 4})
        model = types.SimpleNamespace(peft_config={"default": configuration})
        with full_rank_save_config(model) as saved_patterns:
            self.assertIsNone(configuration.rank_pattern)
            self.assertEqual(saved_patterns, {"default": {"layer": 4}})
        self.assertEqual(configuration.rank_pattern, {"layer": 4})

    def test_restores_rank_pattern_when_saving_fails(self):
        configuration = types.SimpleNamespace(rank_pattern={"layer": 3})
        model = types.SimpleNamespace(peft_config={"default": configuration})
        with self.assertRaisesRegex(RuntimeError, "disk full"):
            with full_rank_save_config(model):
                raise RuntimeError("disk full")
        self.assertEqual(configuration.rank_pattern, {"layer": 3})


class AdaLoRATrainingWorldSizeTests(unittest.TestCase):
    def test_uses_visible_gpus_for_single_process_dataparallel(self):
        self.assertEqual(
            resolve_training_world_size(
                "cuda", distributed_world_size=1, visible_gpu_count=2
            ),
            2,
        )

    def test_prefers_distributed_world_size(self):
        self.assertEqual(
            resolve_training_world_size(
                "cuda", distributed_world_size=4, visible_gpu_count=2
            ),
            4,
        )


if __name__ == "__main__":
    unittest.main()
