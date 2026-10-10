from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from mdv_asr.core.errors import AmbiguousSampleError

from .records import SampleRecord


class DatasetIndex:
    def __init__(self, records: Iterable[SampleRecord], *, source_fingerprint: str | None = None):
        self.records = tuple(records)
        self.source_fingerprint = source_fingerprint
        self._by_key: dict[tuple[str, str], SampleRecord] = {}
        self._by_split: dict[str, list[SampleRecord]] = defaultdict(list)
        self._by_filename: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in self.records:
            record.validate()
            if record.key in self._by_key:
                raise ValueError(f"Duplicate sample key: {record.key}")
            self._by_key[record.key] = record
            self._by_split[record.split].append(record)
            self._by_filename[Path(record.filename).name.casefold()].append(record)

    def __len__(self) -> int:
        return len(self.records)

    def get(self, sample_id: str, dataset_id: str | None = None) -> SampleRecord:
        if dataset_id is not None:
            return self._by_key[(dataset_id, sample_id)]
        hits = [record for (ds_id, sid), record in self._by_key.items() if sid == sample_id]
        if len(hits) != 1:
            if not hits:
                raise KeyError(sample_id)
            raise AmbiguousSampleError(f"sample_id={sample_id!r} appears in multiple datasets")
        return hits[0]

    def for_split(self, split: str) -> tuple[SampleRecord, ...]:
        return tuple(self._by_split.get(split, ()))

    def filter(self, *, region: str | None = None, provinces: Iterable[str] | None = None,
               speakers: Iterable[str] | None = None, splits: Iterable[str] | None = None) -> "DatasetIndex":
        province_set = set(provinces) if provinces is not None else None
        speaker_set = set(speakers) if speakers is not None else None
        split_set = set(splits) if splits is not None else None
        region_aliases = {
            "north": {"north", "miền bắc", "mien bac"},
            "central": {"central", "miền trung", "mien trung"},
            "south": {"south", "miền nam", "mien nam"},
        }
        region_values = None
        if region is not None:
            key = str(region).casefold().strip()
            region_values = region_aliases.get(key, {key})
        rows = [record for record in self.records
                if (region_values is None or str(record.region or "").casefold() in region_values)
                and (province_set is None or record.province_name in province_set)
                and (speaker_set is None or record.speaker_id in speaker_set)
                and (split_set is None or record.split in split_set)]
        return DatasetIndex(rows, source_fingerprint=self.source_fingerprint)

    def find_by_filename(self, filename: str, split: str | None = None) -> SampleRecord:
        target = Path(str(filename)).name.casefold()
        matches = list(self._by_filename.get(target, ()))
        if not matches:
            matches = [record for record in self.records
                       if Path(record.filename).name.casefold().endswith("_" + target)]
        if split is not None:
            matches = [record for record in matches if record.split == split]
        if not matches:
            raise FileNotFoundError(f"No dataset sample matches filename {filename!r}")
        if len(matches) != 1:
            keys = [record.key for record in matches]
            raise AmbiguousSampleError(f"Filename {filename!r} has multiple matches: {keys}")
        return matches[0]

    def find_many(self, filenames: Iterable[str], split: str | None = None) -> list[SampleRecord]:
        return [self.find_by_filename(filename, split=split) for filename in filenames]

    def to_frame(self):
        import pandas as pd

        return pd.DataFrame([{
            "dataset_id": row.dataset_id, "sample_id": row.sample_id, "split": row.split,
            "filename": row.filename, "text": row.text, "duration_seconds": row.duration_seconds,
            "region": row.region, "province_name": row.province_name,
            "province_code": row.province_code, "speaker_id": row.speaker_id,
            "gender": row.gender,
        } for row in self.records])

    def validate(self) -> None:
        for record in self.records:
            record.validate()

