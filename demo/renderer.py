from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import soundfile as sf

from mdv_asr.evaluation.metrics import ErrorCounter

from demo.settings import DemoConfig
from demo.theme import DemoTheme


class DemoRenderer:
    """Render saved ASR predictions; ASR inference is never run during rendering."""

    def __init__(self, config: DemoConfig, predictions_path: str | Path,
                 audio_root: str | Path, output_dir: str | Path):
        config.validate()
        self.config = config
        self.predictions_path = Path(predictions_path)
        self.audio_root = Path(audio_root)
        self.output_dir = Path(output_dir)
        self.theme = DemoTheme.from_config({"font": config.font, "palette": config.palette})

    def _predictions(self) -> list[dict]:
        with self.predictions_path.open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def validate_inputs(self, case_ids=None) -> list[dict]:
        if not self.predictions_path.is_file():
            raise FileNotFoundError(self.predictions_path)
        rows = self._predictions()
        baseline, adapted = self.config.baseline_key, self.config.adapted_key
        by_sample: dict[str, dict[str, dict]] = {}
        for row in rows:
            if row.get("error"):
                continue
            key = str(row.get("sample_id") or row.get("filename") or "")
            if key:
                by_sample.setdefault(key, {})[str(row.get("model_key") or row.get("model"))] = row
        selected_ids = list(case_ids or self.config.case_ids)
        if not selected_ids:
            selected_ids = [key for key, models in by_sample.items()
                            if baseline in models and adapted in models][:3]
        if not selected_ids:
            raise ValueError("No paired samples found in prediction file")
        cases = []
        counter = ErrorCounter()
        for sample_id in selected_ids:
            models = by_sample.get(str(sample_id))
            if models is None:
                models = next((value for value in by_sample.values()
                               if any(row.get("filename") == sample_id for row in value.values())), None)
            if models is None or baseline not in models or adapted not in models:
                raise KeyError(f"Need predictions from {baseline} and {adapted} for {sample_id}")
            left, right = models[baseline], models[adapted]
            if left.get("reference") != right.get("reference"):
                raise ValueError(f"Reference mismatch for {sample_id}")
            filename = left.get("filename") or str(sample_id)
            audio_path = self._resolve_audio(filename, sample_id=str(sample_id))
            info = sf.info(str(audio_path))
            if info.duration <= 0:
                raise ValueError(f"Audio is empty: {audio_path}")
            metric = {}
            for key, row in (("baseline", left), ("adapted", right)):
                counts = counter.count(row.get("reference", ""), row.get("prediction", ""))
                metric[key] = {"CER": counts.cer * 100, "WER": counts.wer * 100}
            cases.append({"sample_id": str(sample_id), "filename": filename,
                          "audio_path": str(audio_path.resolve()), "reference": left["reference"],
                          "place": left.get("province_name") or left.get("region") or filename,
                          "models": {
                              "baseline": {"name": self.config.baseline_key,
                                           "prediction": left["prediction"], "metrics": metric["baseline"]},
                              "adapted": {"name": self.config.adapted_key,
                                          "prediction": right["prediction"], "metrics": metric["adapted"]},
                          }})
        return cases

    def _resolve_audio(self, filename: str, *, sample_id: str | None = None) -> Path:
        override = (self.config.audio_overrides.get(sample_id) if sample_id else None)
        override = override or self.config.audio_overrides.get(Path(filename).name)
        if override:
            path = Path(override).expanduser()
            if not path.is_absolute():
                path = self.audio_root / path
            if not path.is_file():
                raise FileNotFoundError(f"Configured demo audio does not exist: {path}")
            return path
        direct = self.audio_root / Path(filename).name
        if direct.is_file():
            return direct
        matches = list(self.audio_root.rglob(Path(filename).name))
        if len(matches) != 1:
            raise FileNotFoundError(f"Expected one audio named {filename!r} under {self.audio_root}, found {matches}")
        return matches[0]

    def build_case_manifest(self, case: dict) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sample_id = self._safe_sample_id(case["sample_id"])
        path = self.output_dir / f"{sample_id}.demo.json"
        payload = {**case, "theme": {"font": self.config.font, "palette": self.config.palette}}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @staticmethod
    def _safe_sample_id(sample_id: str) -> str:
        value = str(sample_id)
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"Unsafe demo sample_id: {sample_id!r}")
        return value

    def _render(self, case: dict, *, preview: bool = False) -> Path:
        self.theme.validate()
        destination = self.output_dir / ("preview_" if preview else "") / "videos"
        destination.mkdir(parents=True, exist_ok=True)
        sample_id = self._safe_sample_id(case["sample_id"])
        name = f"{sample_id}_preview" if preview else sample_id
        video = destination / f"{name}.mp4"
        if video.exists():
            raise FileExistsError(f"Refusing to overwrite existing demo video: {video}")
        payload = self.build_case_manifest(case)
        env = os.environ.copy()
        env["VIMD_ASR_DEMO_PAYLOAD"] = str(payload.resolve())
        project_root = Path(__file__).resolve().parents[1]
        package_paths = [str(project_root), str(project_root / "src")]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            package_paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(package_paths)
        env.setdefault("PYGLET_HEADLESS", "1")
        command = [sys.executable, "-m", "manimlib",
                   str(Path(__file__).with_name("scenes.py")), "ASRComparisonScene", "-w",
                   "-r", f"{self.config.width}x{self.config.height}",
                   "--video_dir", str(destination), "--file_name", name]
        subprocess.run(command, check=True, env=env)
        if not video.is_file():
            raise FileNotFoundError(f"ManimGL finished but expected video is missing: {video}")
        return video

    def render_case(self, case_id: str) -> Path:
        cases = self.validate_inputs([case_id])
        return self._render(cases[0])

    def render_all(self, case_ids=None) -> list[Path]:
        return [self._render(case) for case in self.validate_inputs(case_ids)]

    def render_preview(self, case_id: str | None = None) -> Path:
        cases = self.validate_inputs([case_id] if case_id else None)
        return self._render(cases[0], preview=True)

    def validate_outputs(self, paths) -> dict:
        values = [Path(path) for path in paths]
        missing = [str(path) for path in values if not path.is_file()]
        return {"count": len(values), "missing": missing, "valid": not missing}
