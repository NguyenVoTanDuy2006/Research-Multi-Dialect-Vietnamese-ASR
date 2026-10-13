"""Optional semantic similarity metrics for ASR transcript pairs."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from collections.abc import Sequence
from typing import Any, Literal

from .helper import distribution_summary, validate_text_pairs
from .metric import Metric


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _as_float_list(values) -> list[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return [float(value) for value in values]


def _content_token_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer.encode(
        text, add_special_tokens=False, truncation=False
    ))


def _special_token_count(tokenizer) -> int:
    count = getattr(tokenizer, "num_special_tokens_to_add", None)
    if count is not None:
        return int(count(pair=False))
    return len(tokenizer.build_inputs_with_special_tokens([]))


def _split_content_tokens(token_ids: list[int], *, max_length: int,
                          special_tokens: int, prefix_ids: list[int] | None = None
                          ) -> list[list[int]]:
    prefix_ids = prefix_ids or []
    capacity = max_length - special_tokens - len(prefix_ids)
    if capacity < 1:
        raise ValueError(
            "The configured token limit leaves no room for transcript content "
            "after model prefixes and special tokens."
        )
    return [token_ids[start:start + capacity]
            for start in range(0, len(token_ids), capacity)] or [[]]


def _prepare_token_batch(tokenizer, sequences: list[list[int]], *,
                         include_special_mask: bool = False):
    features = [
        tokenizer.prepare_for_model(
            sequence, add_special_tokens=True, truncation=False,
            return_attention_mask=True,
            return_special_tokens_mask=include_special_mask,
        )
        for sequence in sequences
    ]
    batch = dict(tokenizer.pad(features, padding=True, return_tensors="pt"))
    special_mask = batch.pop("special_tokens_mask", None)
    return batch, special_mask


def _model_hidden_states(model, batch, device):
    batch = {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask"}
    }
    output = model(**batch)
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None:
        hidden = output[0]
    return hidden, batch["attention_mask"]


@dataclass(frozen=True)
class SemSimE5Result:
    similarity: float
    distance: float
    chunked: bool = False


class SemSimE5(Metric[SemSimE5Result]):
    """Symmetric cosine similarity using multilingual E5 masked-mean embeddings."""

    DEFAULT_MODEL = "intfloat/multilingual-e5-base"
    MAX_TOKENS = 512
    name = "SemSim-E5"

    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str | None = None,
                 batch_size: int = 16, max_length: int = MAX_TOKENS,
                 revision: str | None = None, tokenizer=None, encoder=None,
                 long_text_policy: Literal["error", "chunk"] = "error"):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 1 <= max_length <= self.MAX_TOKENS:
            raise ValueError(f"max_length must be between 1 and {self.MAX_TOKENS}")
        if (tokenizer is None) != (encoder is None):
            raise ValueError("Pass both tokenizer and encoder for dependency injection, or neither")
        if long_text_policy not in {"error", "chunk"}:
            raise ValueError("long_text_policy must be 'error' or 'chunk'")
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.requested_revision = revision
        self._tokenizer = tokenizer
        self._encoder = encoder
        self._resolved_revision = None
        self.long_text_policy = long_text_policy

    def _load(self):
        if self._tokenizer is not None and self._encoder is not None:
            if self.device is None:
                try:
                    self.device = str(next(self._encoder.parameters()).device)
                except (AttributeError, StopIteration, TypeError):
                    self.device = "cpu"
            if self._resolved_revision is None:
                self._resolved_revision = getattr(
                    getattr(self._encoder, "config", None), "_commit_hash", None
                )
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, revision=self.requested_revision
        )
        self._encoder = AutoModel.from_pretrained(
            self.model_name, revision=self.requested_revision
        ).to(self.device)
        self._encoder.eval()
        self._resolved_revision = getattr(self._encoder.config, "_commit_hash", None)

    def _effective_max_length(self) -> int:
        tokenizer_limit = getattr(self._tokenizer, "model_max_length", self.MAX_TOKENS)
        if not isinstance(tokenizer_limit, int) or tokenizer_limit > 1_000_000:
            tokenizer_limit = self.MAX_TOKENS
        return min(self.max_length, tokenizer_limit)

    def _encode(self, texts: list[str]):
        import torch
        import torch.nn.functional as functional

        encoded = self._tokenizer(
            texts, padding=False, truncation=False, add_special_tokens=True
        )["input_ids"]
        limit = self._effective_max_length()
        lengths = [len(row) for row in encoded]
        if any(length > limit for length in lengths):
            index = next(i for i, length in enumerate(lengths) if length > limit)
            raise ValueError(
                f"E5 input has {lengths[index]} tokens including prefix/special tokens; "
                f"maximum is {limit}. Choose an explicit long-text policy instead of truncating."
            )

        batch = self._tokenizer(
            texts, padding=True, truncation=False, return_tensors="pt",
            add_special_tokens=True,
        )
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            output = self._encoder(**batch)
            hidden = output.last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            return functional.normalize(embeddings, p=2, dim=1)

    def _encode_token_sequences(self, sequences: list[list[int]]):
        import torch.nn.functional as functional

        batch, _ = _prepare_token_batch(self._tokenizer, sequences)
        hidden, attention_mask = _model_hidden_states(self._encoder, batch, self.device)
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return functional.normalize(embeddings, p=2, dim=1)

    def _is_overlong(self, text: str) -> bool:
        prepared = self._tokenizer(
            [f"query: {text}"], padding=False, truncation=False,
            add_special_tokens=True,
        )["input_ids"][0]
        return len(prepared) > self._effective_max_length()

    def _encode_long_texts(self, texts: list[str]):
        import torch
        import torch.nn.functional as functional

        prefix_ids = _content_token_ids(self._tokenizer, "query: ")
        special_tokens = _special_token_count(self._tokenizer)
        sequences, ranges = [], []
        for text in texts:
            content_ids = _content_token_ids(self._tokenizer, text)
            chunks = _split_content_tokens(
                content_ids, max_length=self._effective_max_length(),
                special_tokens=special_tokens, prefix_ids=prefix_ids,
            )
            start = len(sequences)
            sequences.extend([prefix_ids + chunk for chunk in chunks])
            ranges.append((start, len(sequences), [len(chunk) for chunk in chunks]))

        chunk_embeddings = self._encode_token_sequences(sequences)
        text_embeddings = []
        for start, end, weights in ranges:
            weight_tensor = torch.tensor(weights, device=self.device, dtype=chunk_embeddings.dtype)
            combined = (chunk_embeddings[start:end] * weight_tensor[:, None]).sum(dim=0)
            combined = combined / weight_tensor.sum().clamp(min=1)
            text_embeddings.append(functional.normalize(combined, p=2, dim=0))
        return torch.stack(text_embeddings)

    def score(self, reference: str, prediction: str) -> SemSimE5Result:
        return self.score_batch([reference], [prediction])[0]

    def score_batch(self, references: Sequence[str],
                    predictions: Sequence[str]) -> list[SemSimE5Result]:
        pairs = validate_text_pairs(references, predictions)
        if not pairs:
            return []
        results: list[SemSimE5Result | None] = [None] * len(pairs)
        active = [(index, ref, pred) for index, (ref, pred) in enumerate(pairs) if pred]
        for start in range(0, len(active), self.batch_size):
            chunk = active[start:start + self.batch_size]
            self._load()
            texts = [text for _, ref, pred in chunk for text in (ref, pred)]
            is_long = [self._is_overlong(text) for text in texts]
            if any(is_long) and self.long_text_policy == "error":
                self._encode([f"query: {text}" for text in texts])

            embeddings = [None] * len(texts)
            short_indices = [index for index, overlong in enumerate(is_long) if not overlong]
            if short_indices:
                short_embeddings = self._encode([
                    f"query: {texts[index]}" for index in short_indices
                ])
                for index, embedding in zip(short_indices, short_embeddings):
                    embeddings[index] = embedding
            long_indices = [index for index, overlong in enumerate(is_long) if overlong]
            if long_indices:
                long_embeddings = self._encode_long_texts([texts[index] for index in long_indices])
                for index, embedding in zip(long_indices, long_embeddings):
                    embeddings[index] = embedding

            for offset, (index, _, _) in enumerate(chunk):
                similarity = float(
                    (embeddings[offset * 2] * embeddings[offset * 2 + 1]).sum().clamp(-1, 1)
                )
                results[index] = SemSimE5Result(
                    similarity, 1.0 - similarity,
                    chunked=is_long[offset * 2] or is_long[offset * 2 + 1],
                )
        for index, (_, prediction) in enumerate(pairs):
            if not prediction:
                results[index] = SemSimE5Result(0.0, 1.0)
        return [result for result in results if result is not None]

    def aggregate(self, scores: Sequence[SemSimE5Result]) -> dict[str, Any]:
        scores = list(scores)
        similarity = distribution_summary(score.similarity for score in scores)
        distance = distribution_summary(score.distance for score in scores)
        return {
            "n_samples": len(scores),
            "SemSim_E5_mean": similarity["mean"],
            "SemSim_E5_median": similarity["median"],
            "SemSim_E5_p05": similarity["p05"],
            "SemDist_E5_mean": distance["mean"],
            "SemDist_E5_median": distance["median"],
            "SemDist_E5_p05": distance["p05"],
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name, "model": self.model_name,
            "requested_revision": self.requested_revision,
            "revision": self._resolved_revision,
            "pooling": "attention_masked_mean_then_l2_normalize",
            "prefix": "query: ", "prefix_applied_to": "both reference and prediction",
            "configured_max_tokens_including_prefix_and_special_tokens": self.max_length,
            "effective_max_tokens": (self._effective_max_length()
                                      if self._tokenizer is not None else self.max_length),
            "batch_size": self.batch_size,
            "transformers_version": _package_version("transformers"),
            "long_text_policy": self.long_text_policy,
            "long_text_strategy": "query-prefixed token chunks; content-token-weighted normalized mean",
        }


@dataclass(frozen=True)
class BERTScoreResult:
    precision: float
    recall: float
    f1: float
    chunked: bool = False


class BERTScoreMetric(Metric[BERTScoreResult]):
    """Official BERTScore with the selected multilingual XLM-R-large layer."""

    DEFAULT_MODEL = "FacebookAI/xlm-roberta-large"
    DEFAULT_LAYERS = 17
    MAX_TOKENS = 512
    name = "BERTScore"

    def __init__(self, model_name: str = DEFAULT_MODEL, *, num_layers: int = DEFAULT_LAYERS,
                 device: str | None = None, batch_size: int = 8,
                 max_length: int = MAX_TOKENS, scorer=None,
                 long_text_policy: Literal["error", "chunk"] = "error"):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if not 1 <= max_length <= self.MAX_TOKENS:
            raise ValueError(f"max_length must be between 1 and {self.MAX_TOKENS}")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if long_text_policy not in {"error", "chunk"}:
            raise ValueError("long_text_policy must be 'error' or 'chunk'")
        self.model_name = model_name
        self.num_layers = int(num_layers)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self._scorer = scorer
        self._resolved_revision = None
        self.long_text_policy = long_text_policy

    def _load(self):
        if self._scorer is not None:
            return
        try:
            from bert_score import BERTScorer
        except ImportError as exc:
            raise ImportError(
                "BERTScore is optional. Install it with `pip install -e '.[semantic]'`."
            ) from exc
        self._scorer = BERTScorer(
            model_type=self.model_name, num_layers=self.num_layers, idf=False,
            device=self.device, rescale_with_baseline=False, use_fast_tokenizer=True,
        )
        config = getattr(getattr(self._scorer, "_model", None), "config", None)
        self._resolved_revision = getattr(config, "_commit_hash", None)

    def _effective_max_length(self) -> int:
        tokenizer = getattr(self._scorer, "_tokenizer", None)
        if tokenizer is None:
            return self.max_length
        tokenizer_limit = getattr(tokenizer, "model_max_length", self.MAX_TOKENS)
        if not isinstance(tokenizer_limit, int) or tokenizer_limit > 1_000_000:
            tokenizer_limit = self.MAX_TOKENS
        return min(self.max_length, tokenizer_limit)

    def _check_length(self, texts: list[str]) -> None:
        tokenizer = getattr(self._scorer, "_tokenizer", None)
        if tokenizer is None:
            return
        limit = self._effective_max_length()
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
            if len(ids) > limit:
                raise ValueError(
                    f"BERTScore input has {len(ids)} tokens including special tokens; "
                    f"maximum is {limit}. Choose an explicit long-text policy instead of truncating."
                )

    def _is_overlong(self, text: str) -> bool:
        tokenizer = self._scorer._tokenizer
        ids = tokenizer.encode(text, add_special_tokens=True, truncation=False)
        return len(ids) > self._effective_max_length()

    def _encode_content_tokens(self, text: str):
        import torch
        import torch.nn.functional as functional

        tokenizer = self._scorer._tokenizer
        model = self._scorer._model
        content_ids = _content_token_ids(tokenizer, text)
        if not content_ids:
            raise ValueError("BERTScore cannot compare text with no content tokens")
        chunks = _split_content_tokens(
            content_ids, max_length=self._effective_max_length(),
            special_tokens=_special_token_count(tokenizer),
        )
        batch, special_mask = _prepare_token_batch(
            tokenizer, chunks, include_special_mask=True
        )
        device = self.device or getattr(self._scorer, "device", None) or "cpu"
        hidden, attention_mask = _model_hidden_states(model, batch, device)
        if special_mask is None:
            raise RuntimeError("Tokenizer did not return a special_tokens_mask for chunked BERTScore")
        content_mask = attention_mask.bool() & ~special_mask.to(device).bool()
        pieces = [
            hidden[index][content_mask[index]]
            for index in range(len(chunks))
        ]
        embeddings = functional.normalize(torch.cat(pieces, dim=0), p=2, dim=-1)
        return embeddings

    def _score_long_pair(self, reference: str, prediction: str) -> BERTScoreResult:
        import torch

        reference_embeddings = self._encode_content_tokens(reference)
        prediction_embeddings = self._encode_content_tokens(prediction)
        similarities = prediction_embeddings @ reference_embeddings.transpose(0, 1)
        precision = similarities.max(dim=1).values.mean()
        recall = similarities.max(dim=0).values.mean()
        denominator = precision + recall
        f1 = 2 * precision * recall / denominator if float(denominator) != 0.0 else torch.zeros_like(denominator)
        return BERTScoreResult(
            float(precision), float(recall), float(f1), chunked=True
        )

    def score(self, reference: str, prediction: str) -> BERTScoreResult:
        return self.score_batch([reference], [prediction])[0]

    def score_batch(self, references: Sequence[str],
                    predictions: Sequence[str]) -> list[BERTScoreResult]:
        pairs = validate_text_pairs(references, predictions)
        if not pairs:
            return []
        results: list[BERTScoreResult | None] = [None] * len(pairs)
        active = [(index, ref, pred) for index, (ref, pred) in enumerate(pairs) if pred]
        for start in range(0, len(active), self.batch_size):
            chunk = active[start:start + self.batch_size]
            self._load()
            refs = [ref for _, ref, _ in chunk]
            hyps = [pred for _, _, pred in chunk]
            is_long = [
                self._is_overlong(ref) or self._is_overlong(hyp)
                for ref, hyp in zip(refs, hyps)
            ]
            if any(is_long) and self.long_text_policy == "error":
                self._check_length(refs + hyps)

            short_offsets = [offset for offset, overlong in enumerate(is_long) if not overlong]
            if short_offsets:
                precision, recall, f1 = self._scorer.score(
                    cands=[hyps[offset] for offset in short_offsets],
                    refs=[refs[offset] for offset in short_offsets],
                    batch_size=self.batch_size, verbose=False,
                )
                columns = [_as_float_list(values) for values in (precision, recall, f1)]
                for column_offset, pair_offset in enumerate(short_offsets):
                    row_index = chunk[pair_offset][0]
                    results[row_index] = BERTScoreResult(
                        *(column[column_offset] for column in columns), chunked=False
                    )

            for offset, (row_index, reference, prediction) in enumerate(chunk):
                if is_long[offset]:
                    results[row_index] = self._score_long_pair(reference, prediction)
        for index, (_, prediction) in enumerate(pairs):
            if not prediction:
                results[index] = BERTScoreResult(0.0, 0.0, 0.0)
        return [result for result in results if result is not None]

    def aggregate(self, scores: Sequence[BERTScoreResult]) -> dict[str, Any]:
        scores = list(scores)
        summaries = {
            "precision": distribution_summary(score.precision for score in scores),
            "recall": distribution_summary(score.recall for score in scores),
            "F1": distribution_summary(score.f1 for score in scores),
        }
        result: dict[str, Any] = {"n_samples": len(scores)}
        for name, summary in summaries.items():
            for stat, value in summary.items():
                result[f"BERTScore_{name}_{stat}"] = value
        return result

    def metadata(self) -> dict[str, Any]:
        config = getattr(getattr(self._scorer, "_model", None), "config", None)
        revision = self._resolved_revision or getattr(config, "_commit_hash", None)
        return {
            "name": self.name, "model": self.model_name,
            "revision": revision, "num_layers": self.num_layers,
            "idf": False, "rescale_with_baseline": False,
            "use_fast_tokenizer": True,
            "configured_max_tokens_including_special_tokens": self.max_length,
            "effective_max_tokens": (self._effective_max_length()
                                     if self._scorer is not None else self.max_length),
            "batch_size": self.batch_size,
            "bert_score_version": _package_version("bert-score"),
            "long_text_policy": self.long_text_policy,
            "long_text_strategy": (
                "contextual token chunks; global greedy cosine matching over all content tokens"
            ),
        }
