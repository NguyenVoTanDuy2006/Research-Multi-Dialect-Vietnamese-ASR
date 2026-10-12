from __future__ import annotations

import ast
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


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

    def test_notebook_bootstrap_replaces_existing_source_checkout(self):
        for relative_path in ("src/nb/fine_tuning_qwen.py", "src/nb/benchmark_qwen.py"):
            with self.subTest(notebook=relative_path), tempfile.TemporaryDirectory() as tmp:
                checkout = Path(tmp) / "mdv_asr_source"
                checkout.mkdir()
                stale_file = checkout / "stale.txt"
                stale_file.write_text("old source", encoding="utf-8")

                source = (self.root / relative_path).read_text(encoding="utf-8")
                bootstrap_cell = next(
                    cell for cell in re.split(r"(?m)^# %%[^\n]*\n", source)
                    if "REPO_DIR =" in cell and '"git", "clone"' in cell
                )
                tree = ast.parse(bootstrap_cell)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id == "REPO_DIR"
                        for target in node.targets
                    ):
                        node.value = ast.Call(
                            func=ast.Name(id="Path", ctx=ast.Load()),
                            args=[ast.Constant(value=str(checkout))],
                            keywords=[],
                        )

                commands = []

                def fake_run(command, check):
                    commands.append(command)
                    if command[0] == "git":
                        self.assertFalse(checkout.exists(), "old checkout must be removed before clone")
                        checkout.mkdir()
                        (checkout / "fresh.txt").write_text("new source", encoding="utf-8")
                    return subprocess.CompletedProcess(command, 0)

                previous_sys_path = sys.path[:]
                try:
                    with patch("subprocess.run", side_effect=fake_run):
                        exec(compile(ast.fix_missing_locations(tree), relative_path, "exec"), {})
                finally:
                    sys.path[:] = previous_sys_path

                self.assertEqual(commands[0][0], "git")
                self.assertFalse(stale_file.exists())
                self.assertTrue((checkout / "fresh.txt").is_file())


if __name__ == "__main__":
    unittest.main()
