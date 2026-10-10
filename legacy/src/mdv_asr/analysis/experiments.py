from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from mdv_asr.core.artifacts import ArtifactStore
from mdv_asr.evaluation.metrics import MetricAggregator
from mdv_asr.evaluation.records import PredictionRecord


class ExperimentAnalyzer:
    """Load and compare benchmark/training artifacts without rerunning models."""

    def load_results(self, paths: Iterable[str | Path]) -> list[PredictionRecord]:
        rows: list[PredictionRecord] = []
        for path in paths:
            path = Path(path)
            if path.suffix.casefold() == ".jsonl":
                with path.open(encoding="utf-8") as stream:
                    payload = [json.loads(line) for line in stream if line.strip()]
            elif path.suffix.casefold() == ".csv":
                rows.extend(self.import_legacy_results(path))
                continue
            else:
                raise ValueError(f"Unsupported result format: {path}")
            rows.extend(PredictionRecord.from_dict(item) for item in payload)
        return rows

    def import_legacy_results(self, path: str | Path) -> list[PredictionRecord]:
        frame = pd.read_csv(path)
        rows = []
        for index, item in enumerate(frame.to_dict("records")):
            rows.append(PredictionRecord(
                dataset_id=str(item.get("dataset_id") or "legacy"),
                sample_id=str(item.get("sample_id") or item.get("filename") or index),
                model_key=str(item.get("model_key") or item.get("model") or "unknown"),
                reference=str(item.get("reference") or ""),
                prediction=str(item.get("prediction") or ""),
                split=str(item.get("split") or "test"),
                filename=str(item.get("filename") or index),
                region=_optional(item.get("region")),
                province_name=_optional(item.get("province_name")),
                speaker_id=_optional(item.get("speaker_id", item.get("speakerID"))),
                inference_seconds=_number(item.get("inference_seconds")),
                error=str(item.get("error") or ""),
            ))
        return rows

    @staticmethod
    def validate_provenance(manifests: Iterable[str | Path]) -> dict:
        entries = []
        for path in manifests:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
            entries.append({"path": str(Path(path).resolve()),
                            "config_fingerprint": value.get("config_fingerprint"),
                            "provenance": value.get("provenance", {})})
        fingerprints = {entry["config_fingerprint"] for entry in entries
                        if entry["config_fingerprint"] is not None}
        return {"entries": entries, "same_config": len(fingerprints) <= 1}

    def compare_asr_models(self, predictions: Iterable[PredictionRecord], baseline_key: str) -> dict:
        return MetricAggregator().compare_models(predictions, baseline_key)

    @staticmethod
    def compare_training_histories(paths: Iterable[str | Path]) -> pd.DataFrame:
        frames = []
        for path in paths:
            frame = pd.read_csv(path)
            frame.insert(0, "run", Path(path).parent.name)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def plot_comparisons(frame: pd.DataFrame, *, x: str, y: str, hue: str | None = None):
        import matplotlib.pyplot as plt

        fig, axis = plt.subplots(figsize=(8, 4))
        if hue and hue in frame:
            for label, group in frame.groupby(hue):
                axis.plot(group[x], group[y], marker="o", label=str(label))
            axis.legend()
        else:
            axis.plot(frame[x], frame[y], marker="o")
        axis.set(xlabel=x, ylabel=y)
        fig.tight_layout()
        return fig

    @staticmethod
    def export_report(output_dir: str | Path, payload: dict) -> Path:
        return ArtifactStore(output_dir).write_json("experiment_comparison.json", payload)


def _optional(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def _number(value):
    try:
        return None if pd.isna(value) else float(value)
    except (TypeError, ValueError):
        return None
