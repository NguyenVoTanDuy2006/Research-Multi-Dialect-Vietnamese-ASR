from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from mdv_asr.config.types import LexiconConfig
from mdv_asr.core.artifacts import ArtifactStore
from mdv_asr.data.index import DatasetIndex
from mdv_asr.evaluation.normalization import TextNormalizer


class LexicalTokenizer:
    token_re = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

    def __init__(self, stopwords=None):
        self.stopwords = {str(word).casefold() for word in (stopwords or set())}
        self.normalizer = TextNormalizer()

    def tokenize(self, text: str) -> list[str]:
        value = self.normalizer.normalize(text)
        return [match.group(0) for match in self.token_re.finditer(value)
                if len(match.group(0)) >= 2 and match.group(0) not in self.stopwords]

    def unique_tokens(self, text: str) -> set[str]:
        return set(self.tokenize(text))

    @staticmethod
    def describe_policy() -> str:
        return "Unicode letter sequences, lowercase NFC, punctuation/numbers removed, no word segmentation"


class LexicalStatisticsBuilder:
    def __init__(self, tokenizer: LexicalTokenizer | None = None):
        self.tokenizer = tokenizer or LexicalTokenizer()

    def build_token_counts(self, records):
        by_province = defaultdict(Counter)
        for record in records:
            if not record.province_name:
                continue
            by_province[record.province_name].update(self.tokenizer.tokenize(record.text))
        return dict(by_province)

    def build_document_counts(self, records):
        by_province = defaultdict(Counter)
        for record in records:
            if record.province_name:
                by_province[record.province_name].update(self.tokenizer.unique_tokens(record.text))
        return dict(by_province)

    def build_speaker_presence(self, records):
        by_province = defaultdict(lambda: defaultdict(set))
        unknown_speaker_records = 0
        for record in records:
            if not record.province_name:
                continue
            tokens = self.tokenizer.unique_tokens(record.text)
            if not record.speaker_id:
                unknown_speaker_records += 1
                continue
            for token in tokens:
                by_province[record.province_name][token].add(record.speaker_id)
        return {province: dict(words) for province, words in by_province.items()}, unknown_speaker_records

    def build(self, records):
        records = list(records)
        speaker_presence, unknown_rows = self.build_speaker_presence(records)
        return {"token_counts": self.build_token_counts(records),
                "document_counts": self.build_document_counts(records),
                "speaker_presence": speaker_presence,
                "province_speakers": {
                    province: {record.speaker_id for record in records
                               if record.province_name == province and record.speaker_id}
                    for province in sorted({record.province_name for record in records if record.province_name})
                },
                "unknown_speaker_rows": unknown_rows,
                "province_documents": Counter(record.province_name for record in records
                                               if record.province_name)}


class ProvinceSpecificityScorer:
    def __init__(self, prior_alpha: float = 0.01):
        if prior_alpha <= 0:
            raise ValueError("prior_alpha must be positive")
        self.prior_alpha = prior_alpha

    def score(self, statistics: dict) -> pd.DataFrame:
        counts = statistics["token_counts"]
        docs = statistics["document_counts"]
        speakers = statistics["speaker_presence"]
        provinces = sorted(counts)
        global_counts = Counter()
        for value in counts.values():
            global_counts.update(value)
        total_tokens = sum(global_counts.values())
        vocab_size = max(1, len(global_counts))
        rows = []
        for province in provinces:
            local = counts[province]
            local_total = sum(local.values())
            rest_total = total_tokens - local_total
            for word, count in local.items():
                count_rest = global_counts[word] - count
                province_frequency = sum(word in province_counts for province_counts in counts.values())
                tf_per_million = count / local_total * 1_000_000 if local_total else 0.0
                ipf = math.log((len(provinces) + 1) / (province_frequency + 1))
                alpha = self.prior_alpha
                p_local = (count + alpha) / (local_total + alpha * vocab_size)
                p_rest = (count_rest + alpha) / (rest_total + alpha * vocab_size) if rest_total else 0.0
                log_odds = math.log(max(p_local, 1e-15) / max(1 - p_local, 1e-15)) - math.log(
                    max(p_rest, 1e-15) / max(1 - p_rest, 1e-15)
                )
                pmi = math.log2(count * total_tokens / (global_counts[word] * local_total)) if local_total else 0.0
                rows.append({"province_name": province, "word": word,
                             "token_count": count, "document_count": docs[province][word],
                             "speaker_count": len(speakers.get(province, {}).get(word, set())),
                             "province_token_total": local_total,
                             "province_frequency": province_frequency,
                             "tf_per_million": tf_per_million, "ipf": ipf,
                             "tf_ipf": tf_per_million * ipf,
                             "log_odds": log_odds, "pmi": pmi})
        return pd.DataFrame(rows)

    def tf_ipf(self, statistics):
        return self.score(statistics)[["province_name", "word", "tf_ipf"]]

    def log_odds(self, statistics, prior: float | None = None):
        scorer = self if prior is None else ProvinceSpecificityScorer(prior)
        return scorer.score(statistics)[["province_name", "word", "log_odds"]]

    def pmi(self, statistics):
        return self.score(statistics)[["province_name", "word", "pmi"]]


class LexicalSignificanceTester:
    def __init__(self, p_value_threshold: float = 0.05):
        if not 0 < p_value_threshold <= 1:
            raise ValueError("p_value_threshold must be in (0, 1]")
        self.p_value_threshold = p_value_threshold

    @staticmethod
    def build_speaker_table(word: str, province: str, speaker_presence: dict,
                            all_province_speakers: dict[str, set[str]]) -> tuple[int, int, int, int]:
        target = set(all_province_speakers.get(province, set()))
        present_target = set(speaker_presence.get(province, {}).get(word, set()))
        rest_speakers = set().union(*(set(speakers) for name, speakers in all_province_speakers.items()
                                      if name != province))
        present_rest = set().union(*(set(words.get(word, set())) for name, words in speaker_presence.items()
                                     if name != province))
        a = len(target & present_target)
        b = len(target - present_target)
        c = len(rest_speakers & present_rest)
        d = len(rest_speakers - present_rest)
        return a, b, c, d

    def test(self, table: tuple[int, int, int, int]) -> float:
        from scipy.stats import fisher_exact

        a, b, c, d = table
        if a + b == 0 or c + d == 0:
            return 1.0
        return float(fisher_exact([[a, b], [c, d]], alternative="greater").pvalue)

    @staticmethod
    def adjust_fdr(p_values) -> list[float]:
        values = [float(value) for value in p_values]
        if not values:
            return []
        order = sorted(range(len(values)), key=values.__getitem__)
        adjusted = [1.0] * len(values)
        running = 1.0
        for reverse_rank, index in enumerate(reversed(order), start=1):
            rank = len(values) - reverse_rank + 1
            running = min(running, values[index] * len(values) / rank)
            adjusted[index] = min(1.0, running)
        return adjusted

    def apply(self, statistics: pd.DataFrame, source_statistics: dict) -> pd.DataFrame:
        known_speakers = source_statistics.get("province_speakers", {})
        p_values, tables = [], []
        for row in statistics.itertuples(index=False):
            table = self.build_speaker_table(row.word, row.province_name,
                                             source_statistics["speaker_presence"], known_speakers)
            tables.append(table)
            p_values.append(self.test(table))
        result = statistics.copy()
        result["speaker_presence_table"] = [str(value) for value in tables]
        result["p_value_speaker"] = p_values
        result["p_value_speaker_fdr"] = self.adjust_fdr(p_values)
        result["is_candidate"] = ((result["log_odds"] > 0)
                                  & (result["p_value_speaker_fdr"] < self.p_value_threshold))
        return result


class LexiconAnalyzer:
    def __init__(self, tokenizer: LexicalTokenizer | None = None):
        self.tokenizer = tokenizer or LexicalTokenizer()

    def run(self, index: DatasetIndex, config: LexiconConfig) -> pd.DataFrame:
        config.validate()
        region_aliases = {
            "central": {"central", "miền trung", "mien trung"},
            "north": {"north", "miền bắc", "mien bac"},
            "south": {"south", "miền nam", "mien nam"},
        }
        requested_regions = set()
        for value in config.regions:
            key = value.casefold().strip()
            requested_regions.update(region_aliases.get(key, {key}))
        records = [row for row in index.records if row.split in config.splits
                   and (not requested_regions or (row.region or "").casefold() in requested_regions)]
        if config.stopwords_path:
            words = {line.strip().casefold() for line in Path(config.stopwords_path).read_text(
                encoding="utf-8").splitlines() if line.strip()}
            self.tokenizer = LexicalTokenizer(words)
        source_stats = LexicalStatisticsBuilder(self.tokenizer).build(records)
        scored = ProvinceSpecificityScorer(config.prior_alpha).score(source_stats)
        tester = LexicalSignificanceTester(config.p_value_threshold)
        return tester.apply(scored, source_stats)

    @staticmethod
    def rank_candidates(statistics: pd.DataFrame, top_k: int | None = None) -> pd.DataFrame:
        candidates = statistics[statistics["is_candidate"]].sort_values(
            ["province_name", "token_count", "document_count", "speaker_count", "log_odds"],
            ascending=[True, False, False, False, False], kind="stable",
        ).copy()
        candidates["rank_in_province"] = candidates.groupby("province_name").cumcount() + 1
        return candidates if top_k is None else candidates[candidates["rank_in_province"] <= top_k].copy()

    def collect_examples(self, candidates: pd.DataFrame, index: DatasetIndex,
                         examples_per_word: int = 5) -> pd.DataFrame:
        wanted = {(row.province_name, row.word) for row in candidates.itertuples(index=False)}
        wanted_by_province = defaultdict(set)
        for province, word in wanted:
            wanted_by_province[province].add(word)
        rows = []
        for record in index.records:
            province_words = wanted_by_province.get(record.province_name)
            if not province_words:
                continue
            tokens = self.tokenizer.unique_tokens(record.text)
            for word in province_words & tokens:
                rows.append({"dataset_id": record.dataset_id, "sample_id": record.sample_id,
                             "filename": record.filename, "split": record.split,
                             "province_name": record.province_name, "word": word,
                             "speaker_id": record.speaker_id, "text": record.text,
                             "audio_locator": record.audio_locator.to_dict()})
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.groupby(["province_name", "word"], group_keys=False).head(examples_per_word)
        return frame

    def export(self, statistics: pd.DataFrame, output_dir: str | Path,
               top_k: int = 250) -> dict[str, Path]:
        output = Path(output_dir)
        store = ArtifactStore(output)
        candidates = self.rank_candidates(statistics)
        top = self.rank_candidates(statistics, top_k)
        paths = {
            "statistics": store.write_csv("province_word_statistics.csv", statistics.to_dict("records")),
            "candidates": store.write_csv("province_dialect_candidates.csv", candidates.to_dict("records")),
            "top": store.write_csv("province_top_words.csv", top.to_dict("records")),
        }
        return paths

