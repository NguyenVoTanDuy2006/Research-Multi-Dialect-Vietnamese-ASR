from __future__ import annotations

from dataclasses import dataclass

from .metrics import edit_distance
from .normalization import TextNormalizer


@dataclass(frozen=True)
class AlignmentStep:
    operation: str
    reference_index: int | None
    prediction_index: int | None
    reference_token: str | None
    prediction_token: str | None


class WordAligner:
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

