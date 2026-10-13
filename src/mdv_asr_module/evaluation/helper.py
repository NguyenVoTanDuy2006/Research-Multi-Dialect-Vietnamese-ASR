"""Shared evaluation utilities for normalization, alignment and metric helpers."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedText:
    text: str
    tokens: tuple[str, ...]
    characters: tuple[str, ...]


class TextNormalizer:
    """Normalize transcripts for Vietnamese WER/CER and alignment."""

    version = "vi-nfc-lower-punctuation-space-v1"

    @staticmethod
    def _missing(value) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and math.isnan(value):
            return True
        try:
            import pandas as pd

            missing = pd.isna(value)
            return bool(missing) if not hasattr(missing, "__len__") else False
        except (ImportError, TypeError, ValueError):
            return False

    def normalize(self, text) -> str:
        if self._missing(text):
            return ""
        value = unicodedata.normalize("NFC", str(text)).lower()
        value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
        return " ".join(value.split())

    def tokens(self, text) -> list[str]:
        return self.normalize(text).split()

    def characters(self, text) -> list[str]:
        return list(self.normalize(text).replace(" ", ""))

    def normalize_with_spans(self, text) -> NormalizedText:
        normalized = self.normalize(text)
        return NormalizedText(
            normalized, tuple(normalized.split()), tuple(normalized.replace(" ", ""))
        )


def edit_distance(reference, prediction) -> int:
    """Compute Levenshtein distance for token or character sequences."""
    if len(reference) < len(prediction):
        reference, prediction = prediction, reference
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


@dataclass(frozen=True)
class AlignmentStep:
    operation: str
    reference_index: int | None
    prediction_index: int | None
    reference_token: str | None
    prediction_token: str | None


class WordAligner:
    """Align normalized reference and hypothesis words with edit operations."""

    def __init__(self, normalizer: TextNormalizer | None = None):
        self.normalizer = normalizer or TextNormalizer()

    def align(self, reference: str, prediction: str) -> list[AlignmentStep]:
        ref = self.normalizer.tokens(reference)
        hyp = self.normalizer.tokens(prediction)
        rows, cols = len(ref), len(hyp)
        dp = [[0] * (cols + 1) for _ in range(rows + 1)]
        for i in range(rows + 1):
            dp[i][0] = i
        for j in range(cols + 1):
            dp[0][j] = j
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                               dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
        result = []
        i, j = rows, cols
        while i or j:
            if i and j and ref[i - 1] == hyp[j - 1] and dp[i][j] == dp[i - 1][j - 1]:
                result.append(AlignmentStep("match", i - 1, j - 1, ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif i and j and dp[i][j] == dp[i - 1][j - 1] + 1:
                result.append(AlignmentStep("substitution", i - 1, j - 1, ref[i - 1], hyp[j - 1]))
                i -= 1
                j -= 1
            elif i and dp[i][j] == dp[i - 1][j] + 1:
                result.append(AlignmentStep("deletion", i - 1, None, ref[i - 1], None))
                i -= 1
            else:
                result.append(AlignmentStep("insertion", None, j - 1, None, hyp[j - 1]))
                j -= 1
        return list(reversed(result))

    def prediction_spans(self, reference: str, prediction: str) -> list[tuple[str, bool]]:
        return [(str(step.prediction_token), step.operation == "match")
                for step in self.align(reference, prediction)
                if step.prediction_token is not None]

    def displayed_prediction_status(self, reference: str, prediction: str) -> list[bool]:
        """Return one correctness flag per whitespace-delimited displayed token.

        Punctuation can make one displayed token normalize to a token sequence (or
        no token), so direct alignment indexes are not necessarily display indexes.
        """
        display_words = str(prediction).split()
        normalized_words = []
        origins = []
        for display_index, word in enumerate(display_words):
            tokens = self.normalizer.tokens(word)
            normalized_words.extend(tokens)
            origins.extend([display_index] * len(tokens))
        expected = self.normalizer.tokens(reference)
        predicted = normalized_words
        rows, columns = len(expected), len(predicted)
        distance = [[0] * (columns + 1) for _ in range(rows + 1)]
        for i in range(rows + 1):
            distance[i][0] = i
        for j in range(columns + 1):
            distance[0][j] = j
        for i in range(1, rows + 1):
            for j in range(1, columns + 1):
                distance[i][j] = min(distance[i - 1][j] + 1,
                                     distance[i][j - 1] + 1,
                                     distance[i - 1][j - 1] + (expected[i - 1] != predicted[j - 1]))
        status = [True] * len(display_words)
        i, j = rows, columns
        while i or j:
            if i and j and expected[i - 1] == predicted[j - 1] and distance[i][j] == distance[i - 1][j - 1]:
                i, j = i - 1, j - 1
            elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
                status[origins[j - 1]] = False
                i, j = i - 1, j - 1
            elif j and distance[i][j] == distance[i][j - 1] + 1:
                status[origins[j - 1]] = False
                j -= 1
            else:
                i -= 1
        return status

    def deletions(self, reference: str, prediction: str) -> list[str]:
        return [str(step.reference_token) for step in self.align(reference, prediction)
                if step.operation == "deletion"]


def _clean_text(value: str, *, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string, got {type(value).__name__}")
    return " ".join(unicodedata.normalize("NFC", value).split())


def validate_text_pair(reference: str, prediction: str) -> tuple[str, str]:
    """Validate and NFC/whitespace-normalize one reference/hypothesis pair."""
    reference = _clean_text(reference, field="reference")
    prediction = _clean_text(prediction, field="prediction")
    if not reference:
        raise ValueError("Reference must not be empty")
    return reference, prediction


def validate_text_pairs(references, predictions) -> list[tuple[str, str]]:
    """Validate aligned batches; reject bare strings and mismatched lengths."""
    if isinstance(references, str) or isinstance(predictions, str):
        raise TypeError("Batch inputs must be iterables of strings, not a single string")
    try:
        references, predictions = list(references), list(predictions)
    except TypeError as exc:
        raise TypeError("Batch inputs must be iterable") from exc
    if len(references) != len(predictions):
        raise ValueError(
            f"Reference/prediction batch sizes differ: {len(references)} != {len(predictions)}"
        )
    return [validate_text_pair(ref, pred) for ref, pred in zip(references, predictions)]


def distribution_summary(values: Iterable[float]) -> dict[str, float | None]:
    """Return sentence-level mean, median and interpolated 5th percentile."""
    values = [float(value) for value in values]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Metric scores must be finite")
    if not values:
        return {"mean": None, "median": None, "p05": None}

    ordered = sorted(values)
    position = 0.05 * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    p05 = ordered[lower] * (1 - fraction) + ordered[upper] * fraction
    return {
        "mean": sum(ordered) / len(ordered),
        "median": (ordered[(len(ordered) - 1) // 2] + ordered[len(ordered) // 2]) / 2,
        "p05": p05,
    }
