from __future__ import annotations

import json
import sys
import types
import unittest
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
CBA_ROOT = REPO_ROOT / "src" / "CBA-pipeline" / "Reimplement-CBA"
if str(CBA_ROOT) not in sys.path:
    sys.path.insert(0, str(CBA_ROOT))

# Keep this optional-inference test independent from the workspace's audio
# extras, which are not installed in the minimal test environment.
adapter_package = types.ModuleType("reimplement_cba.adapters")
adapter_package.__path__ = [str(CBA_ROOT / "reimplement_cba" / "adapters")]
sys.modules.setdefault("reimplement_cba.adapters", adapter_package)

from reimplement_cba.adapters.inference import (  # noqa: E402
    FasterWhisperAdapter,
    WhisperXAdapter,
    _ct2_runtime_settings,
    create_transcriber,
)
from reimplement_cba.adapters.whisper import PhoWhisperAdapter  # noqa: E402
from reimplement_cba.config import CBAConfig  # noqa: E402
from reimplement_cba.cli import _parser  # noqa: E402
from reimplement_cba.stages.ct2_conversion import prepare_ct2_checkpoint  # noqa: E402


CONFIG_PATH = CBA_ROOT / "configs" / "vimd_phowhisper_medium.json"


class DummyAudio:
    waveform = [0.0] * 16000
    sample_id = "a"

    def __init__(self, duration_seconds=1.0, sample_rate=16000):
        self.duration_seconds = duration_seconds
        self.sample_rate = sample_rate

    def validate(self):
        return None


def make_config(**updates) -> CBAConfig:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["model"]["device"] = "cpu"
    payload["inference"] = {
        "backend": "transformers",
        "batch_size": 8,
        "vad": {"method": "pyannote", "chunk_size_seconds": 30,
                  "onset": 0.3, "offset": 0.3},
    }
    for key, value in updates.items():
        payload["inference"][key] = value
    return CBAConfig(CONFIG_PATH, payload)


class InferenceBackendConfigTests(unittest.TestCase):
    def test_cli_exposes_ct2_preparation_and_split_inference(self):
        parser = _parser()
        prepare_args = parser.parse_args([
            "prepare-ct2", "--output", "ct2-model", "--adapter-path", "adapter"
        ])
        inference_args = parser.parse_args(["infer", "--split", "test"])
        self.assertEqual(prepare_args.command, "prepare-ct2")
        self.assertEqual(inference_args.command, "infer")
        self.assertEqual(inference_args.split, "test")

    def test_config_accepts_each_inference_backend(self):
        for backend in ("transformers", "faster_whisper", "whisperx"):
            with self.subTest(backend=backend):
                config = make_config(backend=backend)
                config.validate()

    def test_config_rejects_unknown_inference_backend(self):
        config = make_config(backend="openai_whisper")
        with self.assertRaisesRegex(ValueError, "inference.backend"):
            config.validate()

    def test_factory_uses_configured_backend(self):
        config = make_config(backend="whisperx")
        with TemporaryDirectory() as model_dir:
            adapter = create_transcriber(config, ct2_model_path=model_dir)
            self.assertIsInstance(adapter, WhisperXAdapter)

    def test_ct2_backends_require_converted_checkpoint(self):
        for backend in ("faster_whisper", "whisperx"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "CTranslate2 checkpoint"):
                    create_transcriber(make_config(backend=backend))

    def test_ct2_artifact_rejects_different_adapter_provenance(self):
        config = make_config(backend="faster_whisper")
        with TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "converted"
            artifact.mkdir()
            adapter = Path(temporary) / "adapter-a"
            adapter.mkdir()
            (artifact / "reimplement_cba.json").write_text(
                json.dumps({"adapter_path": str(adapter)}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "provenance"):
                create_transcriber(
                    config,
                    ct2_model_path=artifact,
                    adapter_path=Path(temporary) / "adapter-b",
                )

    def test_ct2_artifact_rejects_different_base_model(self):
        config = make_config(backend="whisperx")
        with TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "converted"
            artifact.mkdir()
            (artifact / "reimplement_cba.json").write_text(
                json.dumps({"base_model": "another/model", "adapter_path": None}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "base checkpoint"):
                create_transcriber(config, ct2_model_path=artifact)

    def test_ct2_artifact_rejects_different_model_revision(self):
        config = make_config(backend="whisperx")
        config.values["model"]["revision"] = "configured-revision"
        with TemporaryDirectory() as temporary:
            artifact = Path(temporary) / "converted"
            artifact.mkdir()
            (artifact / "reimplement_cba.json").write_text(
                json.dumps({
                    "base_model": config.values["model"]["base_model"],
                    "model_revision": "other-revision",
                    "adapter_path": None,
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "revision"):
                create_transcriber(config, ct2_model_path=artifact)

    def test_auto_device_resolves_ct2_device_and_supported_dtype(self):
        config = make_config(backend="faster_whisper")
        config.values["model"]["device"] = "auto"
        with patch.dict(sys.modules, {
            "ctranslate2": types.SimpleNamespace(get_cuda_device_count=lambda: 0)
        }):
            self.assertEqual(_ct2_runtime_settings(config), ("cpu", "float32"))


class FasterWhisperAdapterTests(unittest.TestCase):
    def test_transcribe_joins_segments_and_uses_vietnamese_options(self):
        calls = {}

        class FakeModel:
            def __init__(self, model_path, **kwargs):
                calls["load"] = (model_path, kwargs)

            def transcribe(self, waveform, **kwargs):
                calls["transcribe"] = (waveform, kwargs)
                return iter([types.SimpleNamespace(text=" xin chào"),
                             types.SimpleNamespace(text=" bạn")]), object()

        fake_module = types.SimpleNamespace(WhisperModel=FakeModel)
        config = make_config(backend="faster_whisper")
        with TemporaryDirectory() as model_dir:
            expected_model_path = str(Path(model_dir).resolve())
            adapter = FasterWhisperAdapter(config, model_dir)
            with patch.dict(sys.modules, {"faster_whisper": fake_module}):
                adapter.load()
                text = adapter.transcribe(DummyAudio())

        self.assertEqual(text, "xin chào bạn")
        self.assertEqual(calls["load"][0], expected_model_path)
        self.assertEqual(calls["transcribe"][1]["language"], "vi")
        self.assertEqual(calls["transcribe"][1]["task"], "transcribe")
        self.assertFalse(calls["transcribe"][1]["vad_filter"])
        self.assertEqual(adapter.metadata()["backend"], "faster_whisper")
        self.assertEqual(adapter.metadata()["ct2_model_path"], expected_model_path)

    def test_transcribe_accepts_long_form_and_rejects_over_configured_limit(self):
        class FakeModel:
            def transcribe(self, waveform, **kwargs):
                return iter([]), object()

        config = make_config(backend="faster_whisper")
        with TemporaryDirectory() as model_dir:
            adapter = FasterWhisperAdapter(config, model_dir)
            adapter.model = FakeModel()
            self.assertEqual(adapter.transcribe(DummyAudio(duration_seconds=31.0)), "")
            with self.assertRaisesRegex(ValueError, "600 seconds"):
                adapter.transcribe(DummyAudio(duration_seconds=601.0))


class WhisperXAdapterTests(unittest.TestCase):
    def test_transcribe_uses_configured_vad_and_joins_segments(self):
        calls = {}

        class FakePipeline:
            def transcribe(self, waveform, **kwargs):
                calls["transcribe"] = (waveform, kwargs)
                return {"segments": [{"text": " xin chào"}, {"text": " bạn"}]}

        def load_model(model_path, device, **kwargs):
            calls["load"] = (model_path, device, kwargs)
            return FakePipeline()

        fake_module = types.SimpleNamespace(load_model=load_model)
        config = make_config(backend="whisperx")
        with TemporaryDirectory() as model_dir:
            expected_model_path = str(Path(model_dir).resolve())
            adapter = WhisperXAdapter(config, model_dir)
            with patch.dict(sys.modules, {"whisperx": fake_module}):
                adapter.load()
                text = adapter.transcribe(DummyAudio())

        self.assertEqual(text, "xin chào bạn")
        self.assertEqual(calls["load"][0], expected_model_path)
        self.assertEqual(calls["load"][2]["language"], "vi")
        self.assertEqual(calls["load"][2]["vad_options"], {
            "chunk_size": 30, "vad_onset": 0.3, "vad_offset": 0.3
        })
        self.assertEqual(calls["transcribe"][1]["batch_size"], 8)


class PhoWhisperGenerationSettingsTests(unittest.TestCase):
    def test_short_form_generation_uses_new_token_limit_without_stored_max_length(self):
        class FakeInputFeatures:
            def to(self, device, dtype):
                del device, dtype
                return self

        class FakeProcessor:
            def __call__(self, waveform, **kwargs):
                del waveform, kwargs
                return {"input_features": FakeInputFeatures()}

            def batch_decode(self, token_ids, skip_special_tokens):
                del token_ids, skip_special_tokens
                return ["xin chào"]

        class FakeModel:
            generation_config = types.SimpleNamespace(max_length=448)

            def parameters(self):
                return iter([types.SimpleNamespace(dtype="float16")])

            def generate(self, **kwargs):
                self.max_length_during_generation = self.generation_config.max_length
                self.generation_kwargs = kwargs
                return [[1, 2]]

        class FakeTorch:
            @staticmethod
            def inference_mode():
                return nullcontext()

        adapter = PhoWhisperAdapter.__new__(PhoWhisperAdapter)
        adapter.model = FakeModel()
        adapter.processor = FakeProcessor()
        adapter.device = "cpu"
        adapter.min_duration_seconds = 0.5
        adapter.max_duration_seconds = 600.0
        adapter.language = "vi"
        adapter.task = "transcribe"
        adapter.max_new_tokens = 256
        adapter.num_beams = 1

        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            result = adapter.transcribe(DummyAudio())

        self.assertEqual(result, "xin chào")
        self.assertIsNone(adapter.model.max_length_during_generation)
        self.assertEqual(adapter.model.generation_config.max_length, 448)
        self.assertEqual(adapter.model.generation_kwargs["max_new_tokens"], 256)


class CTranslate2PreparationTests(unittest.TestCase):
    def _fake_dependencies(self):
        class FakeModel:
            def save_pretrained(self, path, **kwargs):
                Path(path, "config.json").write_text("{}", encoding="utf-8")

        class FakeProcessor:
            def save_pretrained(self, path):
                Path(path, "tokenizer.json").write_text("{}", encoding="utf-8")
                Path(path, "preprocessor_config.json").write_text("{}", encoding="utf-8")

        class FakeWhisperModel:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return FakeModel()

        class FakeWhisperProcessor:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return FakeProcessor()

        fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
        fake_transformers = types.SimpleNamespace(
            WhisperForConditionalGeneration=FakeWhisperModel,
            WhisperProcessor=FakeWhisperProcessor,
        )
        return {"torch": fake_torch, "transformers": fake_transformers}

    def test_prepares_base_model_and_writes_conversion_metadata(self):
        calls = {}

        def runner(command, **kwargs):
            calls["command"] = command
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir()
            (output_dir / "model.bin").write_bytes(b"ct2")

        config = make_config()
        with TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "base-ct2"
            with (
                patch.dict(sys.modules, self._fake_dependencies()),
                patch("reimplement_cba.stages.ct2_conversion.shutil.which", return_value="converter"),
            ):
                result = prepare_ct2_checkpoint(config, output_dir, runner=runner)

            self.assertEqual(result, output_dir.resolve())
            self.assertTrue((result / "model.bin").is_file())
            metadata = json.loads((result / "reimplement_cba.json").read_text(encoding="utf-8"))
            self.assertIsNone(metadata["adapter_path"])
            self.assertIn("--quantization", calls["command"])

    def test_merges_adapter_before_conversion(self):
        calls = {"merged": False}

        class FakePeftModel:
            @classmethod
            def from_pretrained(cls, model, path, **kwargs):
                calls["adapter_path"] = path
                return cls()

            def merge_and_unload(self, **kwargs):
                calls["merged"] = True
                return self

            def save_pretrained(self, path, **kwargs):
                Path(path, "config.json").write_text("{}", encoding="utf-8")

        def runner(command, **kwargs):
            output_dir = Path(command[command.index("--output_dir") + 1])
            output_dir.mkdir()
            (output_dir / "model.bin").write_bytes(b"ct2")

        dependencies = self._fake_dependencies()
        with TemporaryDirectory() as temporary:
            adapter_dir = Path(temporary) / "adapter"
            adapter_dir.mkdir()
            output_dir = Path(temporary) / "adapter-ct2"
            dependencies["peft"] = types.SimpleNamespace(PeftModel=FakePeftModel)
            with (
                patch.dict(sys.modules, dependencies),
                patch("reimplement_cba.stages.ct2_conversion.shutil.which", return_value="converter"),
            ):
                prepare_ct2_checkpoint(
                    make_config(), output_dir, adapter_path=adapter_dir, runner=runner
                )

        self.assertTrue(calls["merged"])
        self.assertEqual(calls["adapter_path"], str(adapter_dir.resolve()))


if __name__ == "__main__":
    unittest.main()
