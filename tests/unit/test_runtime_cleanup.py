import tempfile
import unittest
from pathlib import Path

from mdv_asr_module.runtime.cleanup import NotebookOutputCleaner


class NotebookOutputCleanerTests(unittest.TestCase):
    def test_removes_only_source_checkout_and_preserves_notebook_output(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            working = Path(temporary_dir) / "working"
            checkout = working / "mdv_asr_source"
            output = working / "vimd_qwen_pissa_full_2epoch"
            (checkout / "src").mkdir(parents=True)
            (checkout / "src" / "code.py").write_text("source", encoding="utf-8")
            output.mkdir()
            (output / "checkpoint.bin").write_bytes(b"checkpoint")

            cleaner = NotebookOutputCleaner(working_dir=working)
            removed = cleaner.remove_source_checkout(checkout, preserve_paths=[output])

            self.assertIs(removed, True)
            self.assertFalse(checkout.exists())
            self.assertEqual((output / "checkpoint.bin").read_bytes(), b"checkpoint")

    def test_refuses_to_remove_path_outside_working_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            working = Path(temporary_dir) / "working"
            outside = Path(temporary_dir) / "mdv_asr_source"
            outside.mkdir(parents=True)
            cleaner = NotebookOutputCleaner(working_dir=working)

            with self.assertRaisesRegex(ValueError, "direct child"):
                cleaner.remove_source_checkout(outside)

            self.assertTrue(outside.is_dir())

    def test_refuses_to_remove_checkout_when_output_is_inside_it(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            working = Path(temporary_dir) / "working"
            checkout = working / "mdv_asr_source"
            output = checkout / "outputs" / "run"
            output.mkdir(parents=True)
            cleaner = NotebookOutputCleaner(working_dir=working)

            with self.assertRaisesRegex(ValueError, "protected output"):
                cleaner.remove_source_checkout(checkout, preserve_paths=[output])

            self.assertTrue(output.is_dir())

    def test_missing_checkout_is_a_safe_noop(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            working = Path(temporary_dir) / "working"
            working.mkdir()
            cleaner = NotebookOutputCleaner(working_dir=working)

            self.assertIs(cleaner.remove_source_checkout(working / "mdv_asr_source"), False)
