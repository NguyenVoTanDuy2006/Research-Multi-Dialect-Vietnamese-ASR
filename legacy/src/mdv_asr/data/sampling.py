from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace
from typing import Iterable

from mdv_asr.config.types import SamplingConfig

from .index import DatasetIndex
from .records import SampleRecord, SelectionResult


class SpeakerGenderResolver:
    def resolve(self, records: Iterable[SampleRecord]) -> list[SampleRecord]:
        records = list(records)
        observed: dict[str, set[str]] = defaultdict(set)
        for record in records:
            if record.speaker_id and record.gender in {"female", "male"}:
                observed[record.speaker_id].add(record.gender)
        conflicts = {speaker for speaker, genders in observed.items() if len(genders) > 1}
        return [replace(record, gender="unknown") if record.speaker_id in conflicts else record
                for record in records]

    def gender_for(self, speaker_id: str | None, records: Iterable[SampleRecord]) -> str:
        if not speaker_id:
            return "unknown"
        seen = {record.gender for record in records if record.speaker_id == speaker_id
                and record.gender in {"female", "male"}}
        return next(iter(seen)) if len(seen) == 1 else "unknown"

    def conflicts(self, records: Iterable[SampleRecord]) -> dict[str, list[str]]:
        observed: dict[str, set[str]] = defaultdict(set)
        for record in records:
            if record.speaker_id and record.gender in {"female", "male"}:
                observed[record.speaker_id].add(record.gender)
        return {speaker: sorted(genders) for speaker, genders in observed.items() if len(genders) > 1}

    def report(self, records: Iterable[SampleRecord]) -> dict[str, int]:
        records = list(records)
        observed: dict[str, set[str]] = defaultdict(set)
        for record in records:
            if record.speaker_id and record.gender in {"female", "male"}:
                observed[record.speaker_id].add(record.gender)
        speakers = {speaker: next(iter(genders)) if len(genders) == 1 else "unknown"
                    for speaker, genders in observed.items()}
        return {
            "female_speakers": sum(value == "female" for value in speakers.values()),
            "male_speakers": sum(value == "male" for value in speakers.values()),
            "unknown_speakers": sum(value == "unknown" for value in speakers.values()),
            "rows_without_speaker_id": sum(not row.speaker_id for row in records),
            "conflicting_speakers": len(self.conflicts(records)),
        }


class StratifiedSampler:
    def __init__(self, gender_resolver: SpeakerGenderResolver | None = None):
        self.gender_resolver = gender_resolver or SpeakerGenderResolver()

    @staticmethod
    def allocate_quotas(records: Iterable[SampleRecord], total_seconds: float,
                        province_stratified: bool = True) -> dict[str, float]:
        records = list(records)
        if total_seconds <= 0:
            raise ValueError("total_seconds must be positive")
        if not province_stratified:
            return {"__all__": total_seconds}
        totals: dict[str, float] = defaultdict(float)
        for record in records:
            totals[record.province_name or "Unknown"] += record.duration_seconds
        total_available = sum(totals.values())
        if total_available <= 0:
            return {"Unknown": total_seconds}
        return {province: total_seconds * duration / total_available
                for province, duration in totals.items()}

    @staticmethod
    def _round_robin_speakers(records: list[SampleRecord], rng: random.Random) -> list[SampleRecord]:
        groups: dict[str, dict[str, list[SampleRecord]]] = {
            "female": defaultdict(list), "male": defaultdict(list), "unknown": defaultdict(list),
        }
        anonymous: list[SampleRecord] = []
        for record in records:
            if not record.speaker_id:
                anonymous.append(record)
                continue
            groups[record.gender if record.gender in {"female", "male"} else "unknown"][record.speaker_id].append(record)
        for group in groups.values():
            for rows in group.values():
                rng.shuffle(rows)
        keys = {gender: list(speakers) for gender, speakers in groups.items()}
        for values in keys.values():
            rng.shuffle(values)
        order: list[SampleRecord] = []
        speaker_rows = {gender: {speaker: list(rows) for speaker, rows in group.items()}
                        for gender, group in groups.items()}
        queues = {gender: list(values) for gender, values in keys.items()}
        while queues["female"] or queues["male"]:
            for gender in ("female", "male"):
                if queues[gender]:
                    speaker = queues[gender].pop()
                    order.append(speaker_rows[gender][speaker].pop())
                    if speaker_rows[gender][speaker]:
                        queues[gender].insert(0, speaker)
        while queues["unknown"]:
            speaker = queues["unknown"].pop()
            order.append(speaker_rows["unknown"][speaker].pop())
            if speaker_rows["unknown"][speaker]:
                queues["unknown"].insert(0, speaker)
        rng.shuffle(anonymous)
        order.extend(anonymous)
        return order

    def _select_from_group(self, records: list[SampleRecord], quota: float,
                           rng: random.Random, balance_gender: bool) -> list[SampleRecord]:
        candidates = self._round_robin_speakers(records, rng) if balance_gender else list(records)
        if not balance_gender:
            rng.shuffle(candidates)
        selected: list[SampleRecord] = []
        seconds = 0.0
        for record in candidates:
            if seconds >= quota:
                break
            selected.append(record)
            seconds += record.duration_seconds
        return selected

    def select_train(self, records: Iterable[SampleRecord], target_seconds: float,
                     seed: int, province_stratified: bool = True,
                     balance_gender: bool = True) -> list[SampleRecord]:
        records = self.gender_resolver.resolve(records)
        quotas = self.allocate_quotas(records, target_seconds, province_stratified)
        rng = random.Random(seed)
        selected: list[SampleRecord] = []
        for province, quota in sorted(quotas.items()):
            group = records if province == "__all__" else [
                row for row in records if (row.province_name or "Unknown") == province
            ]
            selected.extend(self._select_from_group(group, quota, rng, balance_gender))
        return selected

    def select_evaluation(self, records: Iterable[SampleRecord], target_seconds: float,
                          seed: int, province_stratified: bool = True) -> list[SampleRecord]:
        records = list(records)
        quotas = self.allocate_quotas(records, target_seconds, province_stratified)
        rng = random.Random(seed)
        selected: list[SampleRecord] = []
        for province, quota in sorted(quotas.items()):
            group = records if province == "__all__" else [
                row for row in records if (row.province_name or "Unknown") == province
            ]
            selected.extend(self._select_from_group(group, quota, rng, balance_gender=False))
        return selected

    @staticmethod
    def exclude_overlapping_speakers(records: Iterable[SampleRecord], other: Iterable[SampleRecord]):
        other_speakers = {row.speaker_id for row in other if row.speaker_id}
        return [row for row in records if not row.speaker_id or row.speaker_id not in other_speakers]

    def sample(self, index: DatasetIndex, config: SamplingConfig) -> SelectionResult:
        config.validate()
        records = self.gender_resolver.resolve(index.records)
        by_split: dict[str, list[SampleRecord]] = defaultdict(list)
        for record in records:
            by_split[record.split].append(record)
        selected: list[SampleRecord] = []
        requested: dict[str, float] = {}
        test_speakers = {row.speaker_id for row in by_split.get("test", []) if row.speaker_id}
        for split, target in config.target_seconds.items():
            requested[split] = float(target)
            candidates = list(by_split.get(split, []))
            if split == "valid" and config.exclude_valid_speakers_in_test:
                candidates = [row for row in candidates if not row.speaker_id or row.speaker_id not in test_speakers]
            if split == "train":
                chosen = self.select_train(candidates, float(target), config.seed,
                                           config.province_stratified, config.balance_train_gender)
            else:
                chosen = self.select_evaluation(candidates, float(target), config.seed,
                                                config.province_stratified)
            selected.extend(chosen)
        result = SelectionResult(selected, config.seed, requested)
        result.validate()
        return result

    @staticmethod
    def compare_distribution(source: Iterable[SampleRecord], selection: Iterable[SampleRecord]) -> dict:
        def summarize(rows):
            rows = list(rows)
            speakers = {row.speaker_id: row.gender for row in rows if row.speaker_id}
            return {
                "clips": len(rows),
                "duration_seconds": sum(row.duration_seconds for row in rows),
                "unique_speakers": len(speakers),
                "female_speakers": sum(value == "female" for value in speakers.values()),
                "male_speakers": sum(value == "male" for value in speakers.values()),
                "unknown_speakers": sum(value == "unknown" for value in speakers.values()),
            }
        return {"source": summarize(source), "selection": summarize(selection)}

