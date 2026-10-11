from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mdv_asr_module.config.loader import ConfigLoader
from mdv_asr_module.runtime.environment import RuntimeEnvironment


class RuntimePlatformPathTests(unittest.TestCase):
    def test_top_level_platform_is_loaded_and_serialized_flat(self):
        loader = ConfigLoader()
        base = {
            "dataset": {"kind": "arrow", "root": "vimd-small", "dataset_id": "unit"},
            "platform": "kaggle",
            "output_dir": "runs/experiment",
        }

        configured = loader.from_dict(base)
        self.assertEqual(configured.platform, "kaggle")

        with tempfile.TemporaryDirectory() as temporary_dir:
            saved_path = loader.save_resolved(configured, Path(temporary_dir) / "resolved.json")
            saved = json.loads(saved_path.read_text(encoding="utf-8"))

        self.assertEqual(saved["platform"], "kaggle")
        self.assertNotIn("environment", saved)

    def test_kaggle_base_paths_default_inside_runtime_module(self):
        config = ConfigLoader().from_dict({
            "platform": "kaggle",
            "dataset": {"kind": "arrow", "root": "vimd-small", "dataset_id": "unit"},
            "output_dir": "runs/experiment",
        })
        runtime = RuntimeEnvironment(platform="kaggle", project_root="/project")

        resolved = runtime.resolve_config_paths(config)

        self.assertEqual(Path(resolved.dataset.root), Path("/kaggle/input/vimd-small"))
        self.assertEqual(Path(resolved.output_dir), Path("/kaggle/working/runs/experiment"))

    def test_local_relative_paths_are_anchored_at_project_root(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            project_root = Path(temporary_dir)
            config = ConfigLoader().from_dict({
                "platform": "local",
                "dataset": {"kind": "arrow", "root": "data/vimd-small", "dataset_id": "unit"},
                "output_dir": "outputs/run-1",
            })
            runtime = RuntimeEnvironment(platform="local", project_root=project_root)

            resolved = runtime.resolve_config_paths(config)

            self.assertEqual(Path(resolved.dataset.root), project_root / "data/vimd-small")
            self.assertEqual(Path(resolved.output_dir), project_root / "outputs/run-1")

    def test_absolute_paths_are_not_rebased(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            dataset = root / "mounted-data"
            output = root / "mounted-output"
            config = ConfigLoader().from_dict({
                "platform": "kaggle",
                "dataset": {"kind": "arrow", "root": str(dataset), "dataset_id": "unit"},
                "output_dir": str(output),
            })
            resolved = RuntimeEnvironment(platform="kaggle", project_root=root).resolve_config_paths(config)

            self.assertEqual(Path(resolved.dataset.root), dataset)
            self.assertEqual(Path(resolved.output_dir), output)

    def test_all_configs_use_top_level_platform(self):
        root = Path(__file__).resolve().parents[2]
        config_paths = sorted((root / "src/mdv_asr_module/configs").rglob("*.json"))
        loader = ConfigLoader()

        for path in config_paths:
            raw = json.loads(path.read_text(encoding="utf-8"))
            config = loader.load(path)
            self.assertEqual(raw["platform"], "kaggle", path.name)
            self.assertNotIn("environment", raw, path.name)
            self.assertEqual(config.platform, "kaggle", path.name)


if __name__ == "__main__":
    unittest.main()
