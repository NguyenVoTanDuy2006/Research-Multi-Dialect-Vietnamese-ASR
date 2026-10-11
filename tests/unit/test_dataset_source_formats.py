from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdv_asr_module.config.loader import ConfigLoader
from mdv_asr_module.config.types import DatasetConfig
from mdv_asr_module.data.sources import (
    ArrowDatasetSource,
    DatasetSourceFactory,
    ParquetDatasetSource,
)


FULL_VIMD_ROOT = (
    "/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0/"
    "c89c5d04a2d1025e3aab24345dbe4f23d2cfb666"
)
SMALL_VIMD_ROOT = "/kaggle/input/datasets/nvtd1796/small-vimd"


class DatasetSourceFormatTests(unittest.TestCase):
    def test_package_kind_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported dataset source kind"):
            DatasetConfig(kind="package", root="/data", dataset_id="unit").validate()

    def test_arrow_source_finds_all_three_split_shards(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for split in ("train", "valid", "test"):
                (root / f"vi_md_dataset-{split}-00000-of-00001.arrow").touch()

            source = DatasetSourceFactory.create(
                DatasetConfig(kind="arrow", root=str(root), dataset_id="unit")
            )

            self.assertIsInstance(source, ArrowDatasetSource)
            self.assertEqual(set(source._paths), {"train", "valid", "test"})
            source.close()

    def test_parquet_source_stays_supported(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            for split in ("train", "valid", "test"):
                (root / f"vimd-{split}-00000.parquet").touch()

            source = DatasetSourceFactory.create(
                DatasetConfig(kind="parquet", root=str(root), dataset_id="unit")
            )

            self.assertIsInstance(source, ParquetDatasetSource)
            self.assertEqual(set(source._paths), {"train", "valid", "test"})
            source.close()

    def test_dataset_and_training_configs_use_their_kaggle_arrow_roots(self):
        root = Path(__file__).resolve().parents[2]
        expected = {
            "vimd_full": FULL_VIMD_ROOT,
            "vimd_small": SMALL_VIMD_ROOT,
        }
        config_paths = [
            root / "src/mdv_asr_module/configs/datasets/vimd_full.json",
            root / "src/mdv_asr_module/configs/datasets/vimd_small.json",
            *sorted((root / "src/mdv_asr_module/configs/training").glob("*.json")),
        ]
        loader = ConfigLoader()

        for path in config_paths:
            config = loader.load(path)
            dataset_name = "vimd_small" if "small" in path.name else "vimd_full"
            self.assertEqual(config.dataset.kind, "arrow", path.name)
            self.assertEqual(config.dataset.root, expected[dataset_name], path.name)

    def test_configs_remain_valid_json(self):
        root = Path(__file__).resolve().parents[2]
        for path in [
            root / "src/mdv_asr_module/configs/datasets/vimd_full.json",
            root / "src/mdv_asr_module/configs/datasets/vimd_small.json",
            *sorted((root / "src/mdv_asr_module/configs/training").glob("*.json")),
        ]:
            json.loads(path.read_text(encoding="utf-8"))

    def test_small_training_script_uses_shared_shard_source(self):
        root = Path(__file__).resolve().parents[2]
        script = (root / "src/nb/kaggle_finetune_qwen_small_vimd.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("DatasetSourceFactory.create(config.dataset)", script)
        self.assertIn("source.build_index()", script)
        self.assertIn("qwen_small_vimd_adalora.json", script)
        self.assertNotIn("zipfile", script)
        self.assertNotIn("sample_metadata.json", script)


if __name__ == "__main__":
    unittest.main()
