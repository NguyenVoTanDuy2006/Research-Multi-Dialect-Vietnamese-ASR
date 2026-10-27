from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO_ROOT / "src/CBA-pipeline/Reimplement-CBA/notebooks/whisper_cl.py"
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


class NotebookCleanupTests(unittest.TestCase):
    def test_removes_entire_checkout_including_internal_outputs(self):
        with TemporaryDirectory() as directory:
            working = Path(directory) / "working"
            checkout = working / "mdv_asr_source"
            legacy_file = checkout / "outputs" / "benchmark_models" / "result.json"
            legacy_file.parent.mkdir(parents=True)
            legacy_file.write_text('{"score": 1}', encoding="utf-8")
            cba_output = working / "outputs" / "cba_whisper" / "run-1"
            cba_output.mkdir(parents=True)
            (cba_output / "run-summary.json").write_text("complete", encoding="utf-8")

            cleanup_cell = NOTEBOOK.read_text(encoding="utf-8").rsplit("# %%\n", 1)[1]
            exec(compile(cleanup_cell, str(NOTEBOOK), "exec"), {
                "KAGGLE_WORKING": working,
                "REPO_DIR": checkout,
                "OUTPUT_DIR": cba_output,
            })

            self.assertFalse(checkout.exists())
            self.assertFalse(legacy_file.exists())
            self.assertEqual((cba_output / "run-summary.json").read_text(), "complete")


if __name__ == "__main__":
    unittest.main()
