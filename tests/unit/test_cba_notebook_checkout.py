from __future__ import annotations

import ast
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace


NOTEBOOK = (
    Path(__file__).resolve().parents[2]
    / "src/CBA-pipeline/Reimplement-CBA/notebooks/whisper_cl.py"
)
BRANCH = "research-CBA-Whisper"


def git(*args: object) -> None:
    subprocess.run(["git", *(str(arg) for arg in args)], check=True, capture_output=True)


def run_checkout_cell(
    working: Path, origin: Path, *, loaded_modules: dict[str, object] | None = None
) -> None:
    source = ast.parse(NOTEBOOK.read_text(encoding="utf-8"))
    kaggle_block = next(
        node for node in source.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call)
        and isinstance(node.test.func, ast.Attribute)
        and isinstance(node.test.func.value, ast.Name)
        and node.test.func.value.id == "KAGGLE_WORKING"
    )
    checkout_statements = []
    for statement in kaggle_block.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "project_dir"
            for target in statement.targets
        ):
            break
        checkout_statements.append(statement)
    code = compile(ast.Module(body=checkout_statements, type_ignores=[]), str(NOTEBOOK), "exec")
    scope = {
        "REPO_DIR": working / "mdv_asr_source",
        "KAGGLE_WORKING": working,
        "REPO_URL": str(origin),
        "REPO_BRANCH": BRANCH,
        "PROJECT_RELATIVE": Path("src/CBA-pipeline/Reimplement-CBA"),
        "running_from_clone": False,
        "subprocess": subprocess,
        "sys": SimpleNamespace(modules=loaded_modules or {}),
    }
    exec(code, scope)


class NotebookCheckoutTests(unittest.TestCase):
    def test_refuses_to_update_source_in_kernel_with_loaded_cba_modules(self):
        with TemporaryDirectory() as directory:
            working = Path(directory) / "working"
            (working / "mdv_asr_source").mkdir(parents=True)

            with self.assertRaisesRegex(RuntimeError, "Restart kernel"):
                run_checkout_cell(
                    working,
                    Path(directory) / "origin.git",
                    loaded_modules={"reimplement_cba": object()},
                )

    def test_existing_checkout_fast_forwards_and_preserves_untracked_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            origin = root / "origin.git"
            publisher = root / "publisher"
            working = root / "working"
            working.mkdir()
            git("init", "--bare", f"--initial-branch={BRANCH}", origin)
            git("clone", origin, publisher)
            git("-C", publisher, "config", "user.name", "Notebook Test")
            git("-C", publisher, "config", "user.email", "test@example.com")
            (publisher / "version.txt").write_text("one\n", encoding="utf-8")
            git("-C", publisher, "add", "version.txt")
            git("-C", publisher, "commit", "-m", "initial")
            git("-C", publisher, "push", "-u", "origin", BRANCH)
            checkout = working / "mdv_asr_source"
            git("clone", "--branch", BRANCH, origin, checkout)
            output = checkout / "outputs" / "checkpoint.bin"
            output.parent.mkdir()
            output.write_bytes(b"saved checkpoint")

            (publisher / "version.txt").write_text("two\n", encoding="utf-8")
            git("-C", publisher, "commit", "-am", "update")
            git("-C", publisher, "push")

            run_checkout_cell(working, origin)

            self.assertEqual((checkout / "version.txt").read_text(), "two\n")
            self.assertEqual(output.read_bytes(), b"saved checkpoint")


if __name__ == "__main__":
    unittest.main()
