from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd

from mdv_asr.data.index import DatasetIndex


class DatasetAnalyzer:
    def __init__(self, output_dir: str | Path | None = None):
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def split_summary(self, index: DatasetIndex) -> pd.DataFrame:
        rows = []
        for split in sorted({record.split for record in index.records}):
            records = list(index.for_split(split))
            rows.append({"split": split, 
                         "clips": len(records),
                         "hours": sum(row.duration_seconds for row in records) / 3600,
                         "speakers": len({row.speaker_id for row in records if row.speaker_id}),
                         "provinces": len({row.province_name for row in records if row.province_name})})
        return pd.DataFrame(rows)

    def province_summary(self, index: DatasetIndex) -> pd.DataFrame:
        frame = index.to_frame()
        if frame.empty:
            return frame
        return frame.groupby(["split", "province_name"], dropna=False).agg(
            clips=("sample_id", "count"), hours=("duration_seconds", lambda values: values.sum() / 3600),
            speakers=("speaker_id", "nunique"),
        ).reset_index()

    def region_summary(self, index: DatasetIndex) -> pd.DataFrame:
        frame = index.to_frame()
        return frame.groupby(["split", "region"], dropna=False).agg(
            clips=("sample_id", "count"), hours=("duration_seconds", lambda values: values.sum() / 3600),
            speakers=("speaker_id", "nunique"),
        ).reset_index()

    def gender_summary(self, index: DatasetIndex) -> pd.DataFrame:
        frame = index.to_frame()
        return frame.groupby(["split", "gender"], dropna=False).agg(
            clips=("sample_id", "count"), speakers=("speaker_id", "nunique")
        ).reset_index()

    def speaker_summary(self, index: DatasetIndex) -> pd.DataFrame:
        frame = index.to_frame()
        return frame.groupby(["split", "speaker_id"], dropna=False).agg(
            clips=("sample_id", "count"), hours=("duration_seconds", lambda values: values.sum() / 3600)
        ).reset_index()

    def duration_distribution(self, index: DatasetIndex) -> pd.DataFrame:
        return index.to_frame()[["split", "sample_id", "duration_seconds", "region", "province_name"]]

    def compare_sample(self, source_index: DatasetIndex, sample_index: DatasetIndex) -> dict:
        source = {(row.dataset_id, row.sample_id): row for row in source_index.records}
        sample = {(row.dataset_id, row.sample_id): row for row in sample_index.records}
        return {"source_count": len(source), 
                "sample_count": len(sample),
                "missing_source_keys": sorted(sample.keys() - source.keys()),
                "duplicate_count": len(sample_index.records) - len(sample)}

    def plot_distributions(self, index: DatasetIndex):
        import matplotlib.pyplot as plt

        frame = index.to_frame()
        axes = frame.boxplot(column="duration_seconds", by="split", rot=0, return_type="axes")
        plt.suptitle("Audio duration by split")
        return axes

    def export(self, index: DatasetIndex, name: str = "dataset_summary") -> dict[str, Path]:
        if self.output_dir is None:
            raise ValueError("output_dir is required for export")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tables = {"splits": self.split_summary(index), "provinces": self.province_summary(index),
                  "regions": self.region_summary(index), "genders": self.gender_summary(index)}
        result = {}
        for key, frame in tables.items():
            path = self.output_dir / f"{name}_{key}.csv"
            frame.to_csv(path, index=False, encoding="utf-8-sig")
            result[key] = path
        return result

