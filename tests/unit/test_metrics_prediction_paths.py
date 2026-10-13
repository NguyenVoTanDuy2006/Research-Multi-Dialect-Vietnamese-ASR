from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "src/nb/metrics_predictions.py"
)


def load_path_functions():
    """Load the notebook's real path helpers without running metric inference."""
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    names = {
        "is_project_root",
        "find_project_root",
        "resolve_prediction_path",
        "resolve_metrics_output_dir",
    }
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    namespace = {"Path": Path, "os": os}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(SCRIPT_PATH), "exec"), namespace)
    return namespace


class MetricsPredictionPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path_functions = load_path_functions()

    def make_repo(self, root: Path) -> Path:
        (root / "src/mdv_asr_module").mkdir(parents=True)
        (root / "pyproject.toml").touch()
        return root

    def test_finds_repo_cloned_as_child_of_kaggle_working_directory(self):
        with self.subTest("repo clone under Kaggle working directory"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as temp_dir:
                working = Path(temp_dir)
                repo = self.make_repo(working / "mdv_asr_source")
                found = self.path_functions["find_project_root"](
                    cwd=working, kaggle_working_dir=working
                )
                self.assertEqual(found, repo.resolve())

    def test_explicit_prediction_path_overrides_repo_default(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = self.make_repo(root / "repo")
            default = repo / "outputs/benchmark_models/qwen_vs_finetuning/predictions_wide.csv"
            default.parent.mkdir(parents=True)
            default.write_text("default", encoding="utf-8")
            override = root / "input/predictions_wide.csv"
            override.parent.mkdir()
            override.write_text("override", encoding="utf-8")
            found = self.path_functions["resolve_prediction_path"](
                repo, configured_path=override
            )
            self.assertEqual(found, override.resolve())

    def test_discovers_one_prediction_file_attached_as_kaggle_input(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = self.make_repo(root / "repo")
            prediction = root / "kaggle/input/prediction-data/predictions_wide.csv"
            prediction.parent.mkdir(parents=True)
            prediction.write_text("csv", encoding="utf-8")
            found = self.path_functions["resolve_prediction_path"](
                repo, kaggle_input_root=root / "kaggle/input"
            )
            self.assertEqual(found, prediction.resolve())

    def test_ambiguous_kaggle_prediction_files_require_explicit_path(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repo = self.make_repo(root / "repo")
            input_root = root / "kaggle/input"
            for dataset in ("first", "second"):
                prediction = input_root / dataset / "predictions_wide.csv"
                prediction.parent.mkdir(parents=True)
                prediction.write_text("csv", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "multiple"):
                self.path_functions["resolve_prediction_path"](
                    repo, kaggle_input_root=input_root
                )

    def test_output_is_writable_kaggle_working_path_not_input_directory(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction = root / "kaggle/input/data/predictions_wide.csv"
            output = self.path_functions["resolve_metrics_output_dir"](
                prediction, is_kaggle=True, kaggle_working_dir=root / "kaggle/working"
            )
            self.assertEqual(
                output, (root / "kaggle/working/outputs/metrics_predictions").resolve()
            )

    def test_local_output_stays_next_to_prediction_file(self):
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            prediction = Path(temp_dir) / "predictions_wide.csv"
            output = self.path_functions["resolve_metrics_output_dir"](
                prediction, is_kaggle=False
            )
            self.assertEqual(output, (Path(temp_dir) / "metrics").resolve())

    def test_prediction_and_output_overrides_are_in_one_editable_notebook_cell(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        cells = re.split(r"(?m)^# %%[^\n]*\n", source)
        matching_cells = [
            cell for cell in cells
            if "PREDICTION_PATH_OVERRIDE =" in cell and "OUTPUT_DIR_OVERRIDE =" in cell
        ]
        self.assertEqual(len(matching_cells), 1)
        self.assertNotIn("BERTScoreMetric(", matching_cells[0])

    def test_notebook_processes_prediction_csv_and_writes_metric_reports(self):
        notebook_path = SCRIPT_PATH.with_suffix(".ipynb")
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            prediction = root / "predictions_wide.csv"
            output = root / "metrics"
            with prediction.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "sample_id", "region", "province_name", "speaker_gender", "reference",
                    "base", "ada-lora_small", "ada-lora_full", "pissa_small", "pissa_full",
                ])
                writer.writeheader()
                writer.writerow({
                    "sample_id": "sample-1", "region": "South", "province_name": "An Giang",
                    "speaker_gender": "female", "reference": "xin chào",
                    "base": "xin chào", "ada-lora_small": "xin chào",
                    "ada-lora_full": "xin chao", "pissa_small": "xin chào",
                    "pissa_full": "xin chào",
                })

            class FakeSemanticMetric:
                def __init__(self, **kwargs):
                    self._encoder = None
                    self._tokenizer = None
                    self._scorer = None

                def score_batch(self, references, predictions):
                    return [SimpleNamespace(
                        similarity=0.9, distance=0.1, chunked=False,
                        precision=0.9, recall=0.9, f1=0.9,
                    ) for _ in references]

                def metadata(self):
                    return {"model": "test-double"}

            old_sys_path = sys.path[:]
            try:
                source_dir = str(SCRIPT_PATH.resolve().parents[1])
                if source_dir not in sys.path:
                    sys.path.insert(0, source_dir)
                namespace = {
                    "IS_KAGGLE": False,
                    "REPO_DIR": SCRIPT_PATH.resolve().parents[2],
                    "PROJECT_ROOT": SCRIPT_PATH.resolve().parents[2],
                    "KAGGLE_WORKING_DIR": root / "kaggle-working",
                    "Path": Path,
                    "os": os,
                    "subprocess": subprocess,
                    "sys": sys,
                }
                config_overrides_set = False
                # Don't clone a repository or install packages in this isolated run.
                with patch("importlib.util.find_spec", return_value=object()), redirect_stdout(io.StringIO()):
                    for cell_index, code in enumerate(code_cells):
                        if cell_index == 0:
                            continue
                        exec(compile(code, f"metrics notebook cell {cell_index}", "exec"), namespace)
                        if "PREDICTION_PATH_OVERRIDE" in namespace and not config_overrides_set:
                            namespace["PREDICTION_PATH_OVERRIDE"] = prediction
                            namespace["OUTPUT_DIR_OVERRIDE"] = output
                            config_overrides_set = True
                        if "SemSimE5" in namespace:
                            namespace["SemSimE5"] = FakeSemanticMetric
                            namespace["BERTScoreMetric"] = FakeSemanticMetric
            finally:
                sys.path[:] = old_sys_path

            with (output / "metrics_per_sample.csv").open(encoding="utf-8-sig") as stream:
                per_sample = list(csv.DictReader(stream))
            with (output / "metrics_overall.csv").open(encoding="utf-8-sig") as stream:
                overall = list(csv.DictReader(stream))
            self.assertEqual(len(per_sample), 5)
            self.assertEqual(len(overall), 5)
            self.assertEqual({row["model"] for row in per_sample}, {
                "base", "ada-lora_small", "ada-lora_full", "pissa_small", "pissa_full",
            })
            self.assertTrue((output / "metrics_by_province.csv").is_file())
            self.assertTrue((output / "metrics_config.json").is_file())


if __name__ == "__main__":
    unittest.main()
