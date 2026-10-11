from __future__ import annotations

import re
import unittest
from pathlib import Path


class FineTuningNotebookScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.script_path = cls.root / "src/nb/fine_tuning_qwen.py"
        cls.source = cls.script_path.read_text(encoding="utf-8")
        cls.cells = re.split(r"(?m)^# %%[^\n]*\n", cls.source)

    def test_notebook_script_has_requested_name(self):
        self.assertTrue(self.script_path.is_file())
        self.assertFalse((self.root / "src/nb/kaggle_finetune_qwen_pissa.py").exists())

    def test_config_path_is_in_its_own_editable_cell(self):
        cells = [cell for cell in self.cells if "CONFIG_FILE =" in cell]
        self.assertEqual(len(cells), 1)
        code_lines = [
            line.strip()
            for line in cells[0].splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(len(code_lines), 1)
        self.assertIn("CONFIG_FILE = Path(", code_lines[0])
        self.assertIn("config_loader.load(CONFIG_FILE)", self.source)

    def test_torchao_uninstall_cell_precedes_launcher_initialization(self):
        torchao_cells = [
            index for index, cell in enumerate(self.cells)
            if "pip" in cell and "uninstall" in cell
        ]
        launcher_cells = [
            index for index, cell in enumerate(self.cells)
            if "launcher = WorkerLauncher" in cell
        ]
        self.assertEqual(len(torchao_cells), 1)
        self.assertEqual(len(launcher_cells), 1)
        self.assertLess(torchao_cells[0], launcher_cells[0])
        self.assertIn("find_spec(\"torchao\")", self.cells[torchao_cells[0]])


if __name__ == "__main__":
    unittest.main()
