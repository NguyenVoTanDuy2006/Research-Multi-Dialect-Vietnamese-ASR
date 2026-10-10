"""ManimGL scenes. This module is imported only by DemoRenderer at render time."""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from manimlib import FadeIn, Line, RoundedRectangle, Scene, Text, VGroup, ValueTracker, linear

from mdv_asr.evaluation.alignment import WordAligner
from demo.layout import WaveformVisualizer
from demo.theme import DemoTheme


def _left_text(value, x, y, size, color, font):
    obj = Text(str(value), font=font, font_size=size, color=color)
    obj.move_to([x + obj.get_width() / 2, y, 0])
    return obj


def _panel(width, height, x, y, color, border):
    obj = RoundedRectangle(width=width, height=height, corner_radius=0.12)
    obj.set_fill(color, opacity=1)
    obj.set_stroke(border, width=1.2, opacity=0.8)
    obj.move_to([x, y, 0])
    return obj


def _word_layout(text, left, right, top, bottom, theme):
    words = str(text).split()
    if not words:
        raise ValueError("Prediction cannot be empty")
    paragraph_text = " ".join(words)
    spans = [match.span() for match in re.finditer(r"\S+", paragraph_text)]
    for size in range(theme.base_font_size, theme.min_font_size - 1, -2):
        paragraph = Text(paragraph_text, font=theme.font, font_size=size,
                          line_width=right - left - 0.16, alignment="LEFT",
                          color=theme.palette["text"], local_configs={span: {} for span in spans})
        if paragraph.get_width() > right - left or paragraph.get_height() > top - bottom:
            continue
        parts = [paragraph.select_part(span) for span in spans]
        if len(parts) != len(words):
            raise ValueError("ManimGL could not map displayed word spans")
        paragraph.move_to([left + paragraph.get_width() / 2,
                           top - paragraph.get_height() / 2, 0])
        for word in parts:
            word.set_opacity(0)
        return paragraph, parts
    raise ValueError("Prediction does not fit the configured panel")


def _word_status(reference, prediction):
    return WordAligner().displayed_prediction_status(reference, prediction)


class ASRComparisonScene(Scene):
    """Two-model comparison with one shared playback timeline and token alignment."""

    def construct(self):
        payload_path = os.environ.get("VIMD_ASR_DEMO_PAYLOAD")
        if not payload_path:
            raise RuntimeError("VIMD_ASR_DEMO_PAYLOAD environment variable is required")
        case = json.loads(Path(payload_path).read_text(encoding="utf-8"))
        theme = DemoTheme.from_config(case.get("theme"))
        theme.validate()
        palette = theme.palette
        self.camera.background_color = palette["background"]
        audio_path = Path(case["audio_path"])
        signal, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
        waveform = signal.mean(axis=1)
        duration = len(waveform) / sample_rate
        levels = WaveformVisualizer(150).compute_levels(waveform)

        title = _left_text(case.get("title", "So sánh nhận dạng tiếng nói"), -6.45, 3.42,
                           32, palette["text"], theme.font)
        place = _left_text(case.get("place", case["sample_id"]), 3.45, 3.42,
                           18, palette["muted"], theme.font)
        wave_panel = _panel(13.0, 1.06, 0, 2.20, palette["panel"], palette["border"])
        reference_panel = _panel(13.0, 1.34, 0, 0.73, palette["panel"], palette["border"])
        left_panel = _panel(6.35, 3.68, -3.32, -1.91, palette["panel"], palette["border"])
        right_panel = _panel(6.35, 3.68, 3.32, -1.91, palette["panel"], palette["border"])
        audio_label = _left_text("AUDIO", -6.30, 2.52, 15, palette["teal"], theme.font)
        audio_duration = _left_text(f"{duration:.1f} s", 5.42, 2.52, 15,
                                     palette["muted"], theme.font)
        reference_label = _left_text("TRANSCRIPT CHUẨN", -6.30, 1.17, 15,
                                     palette["teal"], theme.font)
        reference = Text(case["reference"], font=theme.font, font_size=20,
                         line_width=12.25, alignment="LEFT", color=palette["text"])
        reference.move_to([0, 0.57, 0])
        left_name = _left_text(case["models"]["baseline"]["name"], -6.16, -0.43,
                               20, palette["teal"], theme.font)
        right_name = _left_text(case["models"]["adapted"]["name"], 0.51, -0.43,
                                20, palette["orange"], theme.font)
        divider = VGroup(
            Line([-6.2, -0.88, 0], [-0.42, -0.88, 0], color=palette["border"]),
            Line([0.42, -0.88, 0], [6.2, -0.88, 0], color=palette["border"]),
        )
        left_paragraph, left_words = _word_layout(
            case["models"]["baseline"]["prediction"], -6.15, -0.48, -1.08, -3.42, theme)
        right_paragraph, right_words = _word_layout(
            case["models"]["adapted"]["prediction"], 0.48, 6.15, -1.08, -3.42, theme)

        tracker = ValueTracker(0.0)
        wave_left, wave_right = -6.18, 6.18
        bars = VGroup()
        for index, level in enumerate(levels):
            x = wave_left + (index + 0.5) * (wave_right - wave_left) / len(levels)
            height = 0.08 + 0.55 * float(level / max(float(levels.max()), 1e-8))
            bar = Line([x, 1.88 - height / 2, 0], [x, 1.88 + height / 2, 0])
            bar.set_stroke(palette["wave_inactive"], width=2.5)
            bars.add(bar)
        cursor = Line([wave_left, 1.55, 0], [wave_left, 2.25, 0],
                      color=palette["orange"], stroke_width=2)
        revealed = {"left": 0, "right": 0, "wave": 0}

        def reveal(words, side, accent):
            target = min(len(words), int(tracker.get_value() * len(words) + 1e-8))
            while revealed[side] < target:
                words[revealed[side]].set_opacity(1)
                words[revealed[side]].set_color(accent)
                revealed[side] += 1

        left_paragraph.add_updater(lambda _mob: reveal(left_words, "left", palette["teal"]))
        right_paragraph.add_updater(lambda _mob: reveal(right_words, "right", palette["orange"]))

        def update_wave(_mob):
            target = min(len(bars), int(tracker.get_value() * len(bars) + 1e-8))
            while revealed["wave"] < target:
                bars[revealed["wave"]].set_color(palette["teal"])
                revealed["wave"] += 1
            cursor.set_x(wave_left + tracker.get_value() * (wave_right - wave_left))

        bars.add_updater(update_wave)
        self.add(wave_panel, reference_panel, left_panel, right_panel, title, place,
                 audio_label, audio_duration, reference_label, reference,
                 left_name, right_name, divider, bars, cursor, left_paragraph, right_paragraph)
        self.wait(0.25)
        self.add_sound(str(audio_path))
        self.play(tracker.animate.set_value(1), run_time=duration, rate_func=linear)
        left_paragraph.clear_updaters()
        right_paragraph.clear_updaters()
        bars.clear_updaters()
        for paragraph, words, model_key in (
            (left_paragraph, left_words, "baseline"),
            (right_paragraph, right_words, "adapted"),
        ):
            status = _word_status(case["reference"], case["models"][model_key]["prediction"])
            if len(status) != len(words):
                raise ValueError("Prediction alignment differs from displayed word count")
            for word, correct in zip(words, status):
                word.set_opacity(1)
                word.set_color(palette["correct"] if correct else palette["incorrect"])
        left_metrics = case["models"]["baseline"]["metrics"]
        right_metrics = case["models"]["adapted"]["metrics"]
        score_items = []
        for x, metrics, color in ((-3.32, left_metrics, palette["teal"]),
                                  (3.32, right_metrics, palette["orange"])):
            score = Text(f"CER {metrics['CER']:.2f}%     WER {metrics['WER']:.2f}%",
                         font=theme.font, font_size=18, color=color)
            score.move_to([x, -3.80, 0])
            score_items.append(score)
        self.play(FadeIn(VGroup(*score_items)), run_time=0.45)
        self.wait(1.5)
