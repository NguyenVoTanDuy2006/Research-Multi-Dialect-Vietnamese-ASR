from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mdv_asr.evaluation.alignment import WordAligner


@dataclass(frozen=True)
class WordSpan:
    text: str
    line: int
    start: int
    end: int
    correct: bool | None = None


class TranscriptLayout:
    """Stable-baseline word wrapping and prediction/reference alignment."""

    def __init__(self, max_chars_per_line: int = 48):
        if max_chars_per_line < 1:
            raise ValueError("max_chars_per_line must be positive")
        self.max_chars_per_line = max_chars_per_line

    def build_paragraph(self, text: str) -> list[str]:
        lines, current = [], ""
        for word in str(text).split():
            candidate = f"{current} {word}".strip()
            if current and len(candidate) > self.max_chars_per_line:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        return lines

    def word_spans(self, text: str, reference: str | None = None) -> list[WordSpan]:
        words = str(text).split()
        statuses = None
        if reference is not None:
            statuses = WordAligner().displayed_prediction_status(reference, text)
            if len(statuses) != len(words):
                raise ValueError("Alignment token count differs from displayed prediction words")
        output, line, column = [], 0, 0
        for index, word in enumerate(words):
            added = len(word) + int(column > 0)
            if column and column + added > self.max_chars_per_line:
                line, column = line + 1, 0
                added = len(word)
            start = column + int(column > 0)
            end = start + len(word)
            output.append(WordSpan(word, line, start, end,
                                   statuses[index] if statuses is not None else None))
            column = end
        return output

    def reveal(self, text: str, visible_words: int) -> str:
        words = str(text).split()
        visible_words = max(0, min(int(visible_words), len(words)))
        return " ".join(words[:visible_words])

    def highlight(self, reference: str, prediction: str) -> list[WordSpan]:
        return self.word_spans(prediction, reference)

    def validate_bounds(self, spans: list[WordSpan], max_lines: int) -> None:
        if any(span.line >= max_lines for span in spans):
            raise ValueError("Transcript exceeds available layout height")


class WaveformVisualizer:
    def __init__(self, levels: int = 160):
        if levels < 1:
            raise ValueError("levels must be positive")
        self.level_count = levels

    def compute_levels(self, waveform) -> np.ndarray:
        signal = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if signal.size == 0 or not np.isfinite(signal).all():
            raise ValueError("Waveform is empty or non-finite")
        chunks = np.array_split(signal, min(self.level_count, signal.size))
        return np.asarray([np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) for chunk in chunks],
                          dtype=np.float32)

    def build(self, waveform, width: float, height: float) -> list[tuple[float, float]]:
        levels = self.compute_levels(waveform)
        peak = max(float(levels.max()), 1e-8)
        x = np.linspace(-width / 2, width / 2, len(levels))
        return list(zip(x.tolist(), (levels / peak * height / 2).tolist()))

    @staticmethod
    def duration_seconds(waveform, sample_rate: int) -> float:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        return len(waveform) / sample_rate
