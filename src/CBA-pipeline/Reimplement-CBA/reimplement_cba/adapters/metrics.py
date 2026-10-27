"""Vietnamese ASR metrics used by curriculum scoring and checkpoint selection.

SemScore is a weighted mean of symmetric multilingual-NLI entailment,
multilingual BERTScore-F1, and region-aware phonetic similarity. WER remains a
separate corpus metric and is reported as a ratio in [0, 1]. Transformer models
are loaded lazily, so importing this adapter does not download model weights.
"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .nli_chunking import build_nli_chunks
from .vietnamese_phonetics import VietnamesePhoneticEncoder


@dataclass(frozen=True)
class SemScoreComponents:
    """Normalized component values and their 0..100 composite score."""

    nli: float
    bertscore_f1: float
    phonetic_similarity: float
    semscore: float


class ViMDMetricsAdapter:
    """Batch-capable WER and Vietnamese SemScore implementation.

    ``score_pair`` retains the scaffold's original float-returning interface.
    Pass the sample's province or macro-dialect as ``region`` to enable the
    article-derived dialect rules in the phonetic component. If region is
    missing or unknown, the phonetic component uses canonical phones.

    The default SemScore weights are equal. They are an explicit starting
    configuration for this reimplementation, not weights claimed by the paper.
    """

    DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    DEFAULT_BERTSCORE_MODEL = "FacebookAI/xlm-roberta-large"
    DEFAULT_BERTSCORE_LAYERS = 17
    DEFAULT_WEIGHTS = {
        "nli": 1.0,
        "bertscore_f1": 1.0,
        "phonetic_similarity": 1.0,
    }

    def __init__(
        self,
        *,
        weights: Mapping[str, float] | None = None,
        nli_model: str = DEFAULT_NLI_MODEL,
        nli_revision: str | None = None,
        nli_max_length: int = 256,
        bertscore_model: str = DEFAULT_BERTSCORE_MODEL,
        bertscore_layers: int = DEFAULT_BERTSCORE_LAYERS,
        bertscore_long_text_policy: str = "chunk",
        batch_size: int = 8,
        device: str | None = None,
        nli_scorer: Any | None = None,
        nli_tokenizer: Any | None = None,
        nli_model_instance: Any | None = None,
        bertscore_scorer: Any | None = None,
        phonetic_encoder: VietnamesePhoneticEncoder | None = None,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if nli_max_length < 4:
            raise ValueError("nli_max_length must be at least 4")
        if bertscore_layers < 1:
            raise ValueError("bertscore_layers must be positive")
        if bertscore_long_text_policy not in {"chunk", "error"}:
            raise ValueError("bertscore_long_text_policy must be 'chunk' or 'error'")
        if (nli_tokenizer is None) != (nli_model_instance is None):
            raise ValueError("Pass both nli_tokenizer and nli_model_instance, or neither")

        self.weights = self._normalize_weights(weights)
        self.nli_model_name = nli_model
        self.nli_revision = nli_revision
        self.nli_max_length = int(nli_max_length)
        self.bertscore_model_name = bertscore_model
        self.bertscore_layers = int(bertscore_layers)
        self.bertscore_long_text_policy = bertscore_long_text_policy
        self.batch_size = int(batch_size)
        self.device = device
        self._nli_scorer = nli_scorer
        self._nli_tokenizer = nli_tokenizer
        self._nli_model = nli_model_instance
        self._bertscore_scorer = bertscore_scorer
        self._bertscore_metric = None
        self._nli_resolved_revision = None
        self._phonetics = phonetic_encoder or VietnamesePhoneticEncoder()

        if self._nli_model is not None:
            self._prepare_nli_model()

    @classmethod
    def from_config(cls, config) -> "ViMDMetricsAdapter":
        """Build the adapter from optional ``metrics`` settings in CBAConfig."""
        settings = config.values.get("metrics", {})
        if not isinstance(settings, dict):
            raise ValueError("metrics config must be an object")
        nli = settings.get("nli", {})
        bertscore = settings.get("bertscore", {})
        phonetic = settings.get("phonetic", {})
        if not all(isinstance(section, dict) for section in (nli, bertscore, phonetic)):
            raise ValueError("metrics.nli, metrics.bertscore, and metrics.phonetic must be objects")

        def resolve_resource_path(raw_path):
            if raw_path is None:
                return None
            path = Path(raw_path).expanduser()
            return path if path.is_absolute() else (config.path.parent / path).resolve()

        return cls(
            weights=settings.get("weights"),
            nli_model=nli.get("model", cls.DEFAULT_NLI_MODEL),
            nli_revision=nli.get("revision"),
            nli_max_length=int(nli.get("max_length", 256)),
            bertscore_model=bertscore.get("model", cls.DEFAULT_BERTSCORE_MODEL),
            bertscore_layers=int(bertscore.get("num_layers", cls.DEFAULT_BERTSCORE_LAYERS)),
            bertscore_long_text_policy=bertscore.get("long_text_policy", "chunk"),
            batch_size=int(settings.get("batch_size", 8)),
            device=settings.get("device"),
            phonetic_encoder=VietnamesePhoneticEncoder(
                vocabulary_path=resolve_resource_path(phonetic.get("vocabulary_path")),
                transformations_path=resolve_resource_path(phonetic.get("transformations_path")),
            ),
        )

    @classmethod
    def _normalize_weights(cls, weights: Mapping[str, float] | None) -> dict[str, float]:
        if weights is None:
            weights = cls.DEFAULT_WEIGHTS
        allowed = set(cls.DEFAULT_WEIGHTS)
        unknown = set(weights) - allowed
        if unknown:
            raise ValueError(f"Unknown SemScore weight(s): {sorted(unknown)}")
        normalized = {key: float(weights.get(key, 0.0)) for key in allowed}
        if any(not math.isfinite(value) or value < 0 for value in normalized.values()):
            raise ValueError("SemScore weights must be finite and non-negative")
        total = sum(normalized.values())
        if total <= 0:
            raise ValueError("At least one SemScore weight must be positive")
        return {key: value / total for key, value in normalized.items()}

    @staticmethod
    def _clean_pair(reference: str, hypothesis: str) -> tuple[str, str]:
        if not isinstance(reference, str):
            raise TypeError(f"reference must be a string, got {type(reference).__name__}")
        if not isinstance(hypothesis, str):
            raise TypeError(f"hypothesis must be a string, got {type(hypothesis).__name__}")
        reference = " ".join(unicodedata.normalize("NFC", reference).split())
        hypothesis = " ".join(unicodedata.normalize("NFC", hypothesis).split())
        if not reference:
            raise ValueError("Reference must not be empty")
        return reference, hypothesis

    @staticmethod
    def _validate_batches(references, hypotheses) -> list[tuple[str, str]]:
        if isinstance(references, str) or isinstance(hypotheses, str):
            raise TypeError("Batch inputs must be iterables of strings, not a single string")
        try:
            references, hypotheses = list(references), list(hypotheses)
        except TypeError as exc:
            raise TypeError("Batch inputs must be iterable") from exc
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Reference/hypothesis batch sizes differ: {len(references)} != {len(hypotheses)}"
            )
        return [ViMDMetricsAdapter._clean_pair(ref, hyp)
                for ref, hyp in zip(references, hypotheses)]

    def _resolve_device(self, torch) -> str:
        if self.device and self.device != "auto":
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _prepare_nli_model(self) -> None:
        import torch

        self.device = self._resolve_device(torch)
        self._nli_model = self._nli_model.to(self.device)
        self._nli_model.eval()
        config = getattr(self._nli_model, "config", None)
        self._nli_resolved_revision = getattr(config, "_commit_hash", None)
        id2label = getattr(config, "id2label", {}) or {}
        labels = {int(index): str(label).casefold() for index, label in id2label.items()}
        entailment_ids = [index for index, label in labels.items() if "entail" in label]
        if entailment_ids:
            self._entailment_id = entailment_ids[0]
        elif self.nli_model_name == self.DEFAULT_NLI_MODEL:
            # The selected model card documents logits ordered as entailment,
            # neutral, contradiction when config labels are generic.
            self._entailment_id = 0
        else:
            raise ValueError(
                "Could not identify the entailment logit from model.config.id2label; "
                "use an NLI model with named labels or adapt the model config."
            )

    def _load_nli(self) -> None:
        if self._nli_model is not None:
            if not hasattr(self, "_entailment_id"):
                self._prepare_nli_model()
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "NLI scoring requires the semantic extra: install with `pip install -e '.[semantic]'`."
            ) from exc

        kwargs = {"revision": self.nli_revision} if self.nli_revision else {}
        self._nli_tokenizer = AutoTokenizer.from_pretrained(self.nli_model_name, **kwargs)
        self._nli_model = AutoModelForSequenceClassification.from_pretrained(
            self.nli_model_name, **kwargs
        )
        self.device = self._resolve_device(torch)
        self._prepare_nli_model()

    def _score_nli_model(self, references: list[str], hypotheses: list[str]) -> list[float]:
        import torch

        self._load_nli()
        if self._nli_tokenizer is None:
            raise RuntimeError("NLI tokenizer was not loaded")
        chunks_by_pair = [
            build_nli_chunks(ref, hyp, self._nli_tokenizer, max_length=self.nli_max_length)
            for ref, hyp in zip(references, hypotheses)
        ]
        chunk_scores = [[0.0] * len(chunks) for chunks in chunks_by_pair]
        active = [
            (pair_index, chunk_index, chunk)
            for pair_index, chunks in enumerate(chunks_by_pair)
            for chunk_index, chunk in enumerate(chunks)
            if chunk.weight and chunk.reference and chunk.hypothesis
        ]
        for start in range(0, len(active), self.batch_size):
            batch = active[start:start + self.batch_size]
            premises = [text for _, _, chunk in batch
                        for text in (chunk.reference, chunk.hypothesis)]
            claims = [text for _, _, chunk in batch
                      for text in (chunk.hypothesis, chunk.reference)]
            encoded = self._nli_tokenizer(
                premises,
                claims,
                padding=True,
                truncation=False,
                return_tensors="pt",
            )
            sequence_length = int(encoded["input_ids"].shape[-1])
            if sequence_length > self.nli_max_length:
                raise ValueError(
                    f"NLI aligned chunk uses {sequence_length} tokens, exceeding "
                    f"metrics.nli.max_length={self.nli_max_length}; check tokenizer token counting."
                )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self._nli_model(**encoded)
                logits = output.logits if hasattr(output, "logits") else output[0]
                probabilities = torch.softmax(logits, dim=-1)[:, self._entailment_id]
            directional = probabilities.detach().float().cpu().tolist()
            for index, (pair_index, chunk_index, _) in enumerate(batch):
                chunk_scores[pair_index][chunk_index] = (
                    float(directional[2 * index]) + float(directional[2 * index + 1])
                ) / 2
        results = []
        for chunks, scores in zip(chunks_by_pair, chunk_scores):
            total_weight = sum(chunk.weight + chunk.unmatched_weight for chunk in chunks)
            weighted_score = sum(chunk.weight * score
                                 for chunk, score in zip(chunks, scores))
            results.append(weighted_score / total_weight if total_weight else 0.0)
        return results

    def _score_nli(self, references: list[str], hypotheses: list[str]) -> list[float]:
        if self._nli_scorer is None:
            return self._score_nli_model(references, hypotheses)
        scorer = self._nli_scorer
        if hasattr(scorer, "score_batch"):
            values = scorer.score_batch(references, hypotheses)
        elif callable(scorer):
            values = scorer(references, hypotheses)
        else:
            raise TypeError("nli_scorer must be callable or implement score_batch")
        values = [float(value) for value in values]
        if len(values) != len(references):
            raise ValueError("nli_scorer returned a different number of values than input pairs")
        return [self._unit_score(value, name="NLI") for value in values]

    def _get_bertscore_metric(self):
        if self._bertscore_metric is None:
            try:
                from mdv_asr_module.evaluation.similarity_metrics import BERTScoreMetric
            except ImportError as exc:
                raise ImportError(
                    "BERTScore requires the semantic extra: install with `pip install -e '.[semantic]'`."
                ) from exc
            if not self.device or self.device == "auto":
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._bertscore_metric = BERTScoreMetric(
                model_name=self.bertscore_model_name,
                num_layers=self.bertscore_layers,
                device=self.device,
                batch_size=self.batch_size,
                long_text_policy=self.bertscore_long_text_policy,
                scorer=self._bertscore_scorer,
            )
        return self._bertscore_metric

    @staticmethod
    def _unit_score(value: float, *, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} returned a non-finite score: {value}")
        return min(1.0, max(0.0, value))

    def _blend(self, nli: float, bertscore_f1: float,
               phonetic_similarity: float) -> SemScoreComponents:
        components = {
            "nli": self._unit_score(nli, name="NLI"),
            "bertscore_f1": self._unit_score(bertscore_f1, name="BERTScore-F1"),
            "phonetic_similarity": self._unit_score(phonetic_similarity, name="phonetic similarity"),
        }
        value = 100.0 * sum(self.weights[key] * components[key] for key in components)
        return SemScoreComponents(**components, semscore=min(100.0, max(0.0, value)))

    def score_batch(
        self,
        references: Sequence[str],
        hypotheses: Sequence[str],
        *,
        regions: Sequence[str | None] | str | None = None,
    ) -> list[SemScoreComponents]:
        pairs = self._validate_batches(references, hypotheses)
        if isinstance(regions, str) or regions is None:
            region_values = [regions] * len(pairs)
        else:
            region_values = list(regions)
            if len(region_values) != len(pairs):
                raise ValueError(
                    f"Region batch size differs from text batch: {len(region_values)} != {len(pairs)}"
                )

        nli_values = [0.0] * len(pairs)
        bert_values = [0.0] * len(pairs)
        active = [(index, ref, hyp) for index, (ref, hyp) in enumerate(pairs) if hyp]
        if active:
            active_refs = [ref for _, ref, _ in active]
            active_hyps = [hyp for _, _, hyp in active]
            active_nli = self._score_nli(active_refs, active_hyps)
            for (index, _, _), value in zip(active, active_nli):
                nli_values[index] = value

            bert_metric = self._get_bertscore_metric()
            active_bert = bert_metric.score_batch(active_refs, active_hyps)
            if len(active_bert) != len(active):
                raise ValueError("BERTScore returned a different number of values than input pairs")
            for (index, _, _), value in zip(active, active_bert):
                bert_values[index] = self._unit_score(value.f1, name="BERTScore-F1")

        results = []
        for index, (reference, hypothesis) in enumerate(pairs):
            phonetic = (self._phonetics.similarity(reference, hypothesis, region_values[index])
                        if hypothesis else 0.0)
            results.append(self._blend(nli_values[index], bert_values[index], phonetic))
        return results

    def score_components(
        self,
        reference: str,
        hypothesis: str,
        *,
        region: str | None = None,
    ) -> SemScoreComponents:
        return self.score_batch([reference], [hypothesis], regions=[region])[0]

    def score_pair(self, reference: str, hypothesis: str,
                   *, region: str | None = None) -> float:
        """Return a curriculum-compatible SemScore in the range 0..100."""
        return self.score_components(reference, hypothesis, region=region).semscore

    def score(self, reference: str, hypothesis: str,
              *, region: str | None = None) -> float:
        """Alias implementing the shared PairScorer protocol."""
        return self.score_pair(reference, hypothesis, region=region)

    @staticmethod
    def _wer_tokens(text: str) -> list[str]:
        normalized = unicodedata.normalize("NFC", text).casefold()
        cleaned = "".join(
            " " if unicodedata.category(char).startswith("P") else char
            for char in normalized
        )
        return cleaned.split()

    @staticmethod
    def _word_errors(reference: list[str], hypothesis: list[str]) -> int:
        if len(reference) < len(hypothesis):
            reference, hypothesis = hypothesis, reference
        previous = list(range(len(hypothesis) + 1))
        for i, ref_word in enumerate(reference, start=1):
            current = [i]
            for j, hyp_word in enumerate(hypothesis, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (ref_word != hyp_word),
                ))
            previous = current
        return previous[-1]

    def evaluate_corpus(
        self,
        references: Sequence[str],
        hypotheses: Sequence[str],
        *,
        regions: Sequence[str | None] | str | None = None,
    ) -> dict[str, float | int | None]:
        """Return micro-WER and macro per-sample SemScore/component means.

        WER is a 0..1 ratio over whitespace-delimited Vietnamese syllables.
        SemScore is 0..100 and averaged per sample, matching curriculum scoring.
        """
        pairs = self._validate_batches(references, hypotheses)
        if isinstance(regions, str) or regions is None:
            region_values = [regions] * len(pairs)
        else:
            region_values = list(regions)
            if len(region_values) != len(pairs):
                raise ValueError(
                    f"Region batch size differs from text batch: {len(region_values)} != {len(pairs)}"
                )
        if not pairs:
            return {
                "n_samples": 0, "WER": None, "semscore": None, "SemScore": None,
                "NLI_mean": None, "BERTScore_F1_mean": None,
                "PhoneticSimilarity_mean": None,
            }

        components = self.score_batch(
            [ref for ref, _ in pairs], [hyp for _, hyp in pairs], regions=region_values
        )
        errors = 0
        reference_words = 0
        sentence_wer: list[float] = []
        for reference, hypothesis in pairs:
            ref_words = self._wer_tokens(reference)
            hyp_words = self._wer_tokens(hypothesis)
            if not ref_words:
                raise ValueError("Reference has no words after WER punctuation normalization")
            row_errors = self._word_errors(ref_words, hyp_words)
            errors += row_errors
            reference_words += len(ref_words)
            sentence_wer.append(row_errors / len(ref_words))
        def mean(attribute: str) -> float:
            return sum(getattr(item, attribute) for item in components) / len(components)

        semscore = mean("semscore")
        return {
            "n_samples": len(pairs),
            "wer_errors": errors,
            "reference_words": reference_words,
            "WER": errors / reference_words if reference_words else None,
            "WER_sentence_mean": sum(sentence_wer) / len(sentence_wer),
            "semscore": semscore,
            "SemScore": semscore,
            "NLI_mean": mean("nli"),
            "BERTScore_F1_mean": mean("bertscore_f1"),
            "PhoneticSimilarity_mean": mean("phonetic_similarity"),
        }

    def metadata(self) -> dict[str, Any]:
        if self._bertscore_metric is None:
            bertscore_metadata = {
                "name": "BERTScore", "model": self.bertscore_model_name,
                "revision": None, "num_layers": self.bertscore_layers,
                "long_text_policy": self.bertscore_long_text_policy,
                "status": "not_loaded",
            }
        else:
            bertscore_metadata = self._bertscore_metric.metadata()
        return {
            "name": "ViMD SemScore",
            "formula": "100 * (w_nli * symmetric_entailment + w_bertscore * BERTScore_F1 + w_phonetic * regional_phone_similarity)",
            "weights": dict(self.weights),
            "nli": {
                "model": self.nli_model_name,
                "requested_revision": self.nli_revision,
                "revision": self._nli_resolved_revision,
                "max_length": self.nli_max_length,
                "symmetry": "mean(P(entailment|ref,hyp), P(entailment|hyp,ref)) per aligned chunk",
                "long_pair_strategy": "word-aligned token-budgeted chunks; unmatched tokens have zero-weighted score; token-weighted mean",
                "status": "injected" if self._nli_scorer is not None else (
                    "loaded" if self._nli_model is not None else "not_loaded"
                ),
                "transformers_version": self._package_version("transformers"),
            },
            "bertscore": bertscore_metadata,
            "phonetic": self._phonetics.metadata(),
            "wer_normalization": "NFC, casefold, Unicode punctuation-to-space, whitespace tokenization",
        }

    @staticmethod
    def _package_version(package: str) -> str | None:
        try:
            return version(package)
        except PackageNotFoundError:
            return None
