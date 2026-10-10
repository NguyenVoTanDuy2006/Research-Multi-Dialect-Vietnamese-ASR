"""ManimGL scenes comparing pretrained and adapted ASR predictions."""

from __future__ import annotations

import csv
import os
import re
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf

# ManimGL uses a standalone ModernGL context when rendering to a file.  In a
# headless session the EGL backend works without an X server.
if os.environ.get("PYGLET_HEADLESS") == "1":
    import moderngl

    _create_standalone_context = moderngl.create_standalone_context

    def _create_egl_context(*args, **kwargs):
        kwargs.setdefault("backend", "egl")
        return _create_standalone_context(*args, **kwargs)

    moderngl.create_standalone_context = _create_egl_context

from manimlib import FadeIn, Line, RoundedRectangle, Scene, Text, VGroup, ValueTracker, linear


DEMO_DIR = Path(__file__).resolve().parent
CSV_PATH = DEMO_DIR / "predictions.csv"
AUDIO_DIR = DEMO_DIR / "audio"

PALETTE = {
    "background": "#0F1624",
    "panel": "#18243A",
    "raised": "#1E2B42",
    "border": "#354860",
    "text": "#EEF1F5",
    "muted": "#9AA7BA",
    "teal": "#2BB9AD",
    "orange": "#F4A259",
    "correct": "#4DCCA5",
    "incorrect": "#FF7777",
    "wave_inactive": "#41526A",
}
FONT = "Noto Sans"

CASES = {
    "30_0309.wav": {"place": "Hà Nội", "region": "Miền Bắc"},
    "59_0313.wav": {"place": "TP. Hồ Chí Minh", "region": "Miền Nam"},
    "37_0338.wav": {"place": "Nghệ An", "region": "Miền Trung"},
    "test_super_hard_000000": {
        "place": "Quảng Ngãi",
        "region": "Miền Trung",
        "audio_path": DEMO_DIR.parent / "outputs/super_hard_instance/audio/super_hard.wav",
    },
}


def _normalized_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split()


def _prediction_word_status(reference: str, prediction: str) -> tuple[list[bool], int]:
    """Minimum-edit word alignment; classify each displayed prediction word."""
    displayed = prediction.split()
    predicted = []
    origins = []
    for index, word in enumerate(displayed):
        parts = _normalized_words(word)
        predicted.extend(parts)
        origins.extend([index] * len(parts))
    expected = _normalized_words(reference)
    rows, columns = len(expected), len(predicted)
    distance = [[0] * (columns + 1) for _ in range(rows + 1)]
    for i in range(rows + 1):
        distance[i][0] = i
    for j in range(columns + 1):
        distance[0][j] = j
    for i, reference_word in enumerate(expected, 1):
        for j, predicted_word in enumerate(predicted, 1):
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                distance[i - 1][j - 1] + (reference_word != predicted_word),
            )

    correct = [True] * len(displayed)
    i, j = rows, columns
    while i or j:
        if (i and j and expected[i - 1] == predicted[j - 1]
                and distance[i][j] == distance[i - 1][j - 1]):
            i -= 1
            j -= 1
        elif i and j and distance[i][j] == distance[i - 1][j - 1] + 1:
            correct[origins[j - 1]] = False
            i -= 1
            j -= 1
        elif j and distance[i][j] == distance[i][j - 1] + 1:
            correct[origins[j - 1]] = False
            j -= 1
        else:
            i -= 1  # A reference word was omitted; no prediction word can mark it.
    return correct, distance[rows][columns]


def _read_case(filename: str) -> dict:
    if filename not in CASES:
        raise KeyError(f"Unknown demo case: {filename}")
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["filename"] == filename]
    selected = {}
    for model in ("pretrained", "epoch_3_adapter"):
        found = [row for row in rows if row["model"] == model]
        if len(found) != 1:
            raise ValueError(f"Expected one {model} row for {filename}; found {len(found)}")
        row = found[0]
        if row.get("error") or not row.get("prediction"):
            raise ValueError(f"Missing valid prediction: {filename}, {model}")
        for unit in ("cer", "wer"):
            errors = int(row[f"{unit}_errors"])
            units = int(row[f"{unit}_units"])
            if not (0 <= errors and units > 0):
                raise ValueError(f"Invalid {unit} counts: {filename}, {model}")
            row[f"{unit}_percent"] = 100 * errors / units
        row["word_is_correct"], aligned_errors = _prediction_word_status(
            row["reference"], row["prediction"]
        )
        if aligned_errors != int(row["wer_errors"]):
            raise ValueError(f"Word alignment disagrees with saved WER: {filename}, {model}")
        selected[model] = row
    if selected["pretrained"]["reference"] != selected["epoch_3_adapter"]["reference"]:
        raise ValueError(f"Reference differs between models for {filename}")
    case_config = CASES[filename]
    audio_path = case_config.get("audio_path", AUDIO_DIR / filename)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Missing audio: {audio_path}")
    return {
        "filename": filename,
        "audio_path": audio_path,
        "reference": selected["pretrained"]["reference"],
        "models": selected,
        "place": case_config["place"],
        "region": case_config["region"],
    }


def _left_text(content: str, left: float, center_y: float, size: int, color: str,
               weight: str = "NORMAL") -> Text:
    item = Text(content, font=FONT, font_size=size, color=color, weight=weight)
    item.move_to([left + item.get_width() / 2, center_y, 0])
    return item


def _panel(width: float, height: float, center_x: float, center_y: float) -> RoundedRectangle:
    card = RoundedRectangle(width=width, height=height, corner_radius=0.14)
    card.set_fill(PALETTE["panel"], opacity=1)
    card.set_stroke(PALETTE["border"], width=1.5, opacity=0.72)
    card.move_to([center_x, center_y, 0])
    return card


def _word_layout(content: str, left: float, right: float, top: float,
                 bottom: float) -> tuple[Text, list[VGroup], int]:
    """Use Pango's shared baselines, then reveal the words of that layout."""
    tokens = content.split()
    if not tokens:
        raise ValueError("Prediction is empty")
    available_width = right - left
    available_height = top - bottom
    paragraph_text = " ".join(tokens)
    word_spans = [match.span() for match in re.finditer(r"\S+", paragraph_text)]
    # Render the complete paragraph once.  Separate Text objects per word have
    # different ink bounding boxes and cannot share a typographic baseline.
    for size in (27, 25, 23, 21, 19, 17):
        paragraph = Text(
            paragraph_text, font=FONT, font_size=size,
            line_width=available_width - 0.28, alignment="LEFT",
            color=PALETTE["text"],
            # ManimGL's regex substring selection can miss glyph paths after
            # Vietnamese diacritics.  Label the exact word spans in the SVG.
            local_configs={span: {} for span in word_spans},
        )
        if paragraph.get_width() > available_width or paragraph.get_height() > available_height:
            continue
        words = [paragraph.select_part(span) for span in word_spans]
        if len(words) != len(tokens):
            raise ValueError(f"Word selection mismatch: {len(words)} != {len(tokens)}")
        selected_glyphs = {id(glyph) for word in words for glyph in word.submobjects}
        if len(selected_glyphs) != len(paragraph.submobjects):
            raise ValueError("Some rendered glyphs were not assigned to a word")
        paragraph.move_to([
            left + paragraph.get_width() / 2,
            top - paragraph.get_height() / 2,
            0,
        ])
        for word in words:
            word.set_opacity(0)
        return paragraph, words, size
    raise ValueError("Prediction does not fit inside the transcript panel")


def _reference_text(reference: str) -> Text:
    for size in (24, 23, 22, 21):
        text = Text(reference, font=FONT, font_size=size, line_width=12.0,
                    alignment="LEFT", color=PALETTE["text"])
        if text.get_width() <= 12.42 and text.get_height() <= 0.94:
            text.move_to([-6.35 + text.get_width() / 2, 1.00 - text.get_height() / 2, 0])
            return text
    raise ValueError("Reference does not fit inside its panel")


def _wave_levels(audio_path: Path, count: int = 150) -> tuple[np.ndarray, float]:
    signal, rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    mono = signal.mean(axis=1)
    if not len(mono) or rate <= 0 or not np.isfinite(mono).all():
        raise ValueError(f"Invalid WAV: {audio_path}")
    levels = np.array([
        np.sqrt(np.mean(block * block)) if len(block) else 0.0
        for block in np.array_split(mono, count)
    ], dtype=np.float32)
    scale = max(float(np.percentile(levels, 96)), 1e-6)
    levels = np.clip(levels / scale, 0.06, 1.0)
    return levels, len(mono) / rate


def _score_chip(label: str, percent: float, color: str, center_x: float,
                center_y: float) -> VGroup:
    box = RoundedRectangle(width=2.55, height=0.42, corner_radius=0.08)
    box.set_fill(PALETTE["raised"], opacity=1)
    box.set_stroke(color, width=1.2, opacity=0.85)
    box.move_to([center_x, center_y, 0])
    value = Text(f"{label} {percent:.2f}%", font=FONT, font_size=18, color=color,
                 weight="BOLD")
    value.move_to(box.get_center())
    return VGroup(box, value)


class ASRComparison(Scene):
    filename = ""
    default_camera_config = {"background_color": PALETTE["background"]}

    def construct(self) -> None:
        case = _read_case(self.filename)
        levels, audio_seconds = _wave_levels(case["audio_path"])

        title = _left_text("So sánh nhận dạng giọng địa phương", -6.50, 3.43,
                           38, PALETTE["text"], "BOLD")
        location = _left_text(f"{case['place']}  ·  {case['region']}",
                              3.05, 3.43, 20, PALETTE["muted"])

        wave_panel = _panel(13.0, 1.14, 0, 2.16)
        reference_panel = _panel(13.0, 1.52, 0, 0.76)
        left_panel = _panel(6.35, 3.62, -3.33, -1.98)
        right_panel = _panel(6.35, 3.62, 3.33, -1.98)
        wave_label = _left_text("AUDIO", -6.34, 2.52, 18, PALETTE["teal"], "BOLD")
        duration_label = _left_text(f"{audio_seconds:.1f} s", 5.39, 2.52,
                                    18, PALETTE["muted"])
        ref_label = _left_text("TRANSCRIPT CHUẨN", -6.34, 1.29, 18,
                               PALETTE["teal"], "BOLD")
        reference = _reference_text(case["reference"])

        model_labels = VGroup(
            _left_text("Qwen3-ASR 1.7B", -6.26, -0.58, 23, PALETTE["teal"], "BOLD"),
            _left_text("Model đang phát triển", 0.40, -0.39, 23,
                       PALETTE["orange"], "BOLD"),
        )
        separators = VGroup(
            Line([-6.28, -1.08, 0], [-0.39, -1.08, 0],
                 color=PALETTE["border"], stroke_width=1.5),
            Line([0.39, -1.08, 0], [6.28, -1.08, 0],
                 color=PALETTE["border"], stroke_width=1.5),
        )

        left_words, left_list, left_size = _word_layout(
            case["models"]["pretrained"]["prediction"],
            -6.27, -0.40, -1.17, -3.64,
        )
        right_words, right_list, right_size = _word_layout(
            case["models"]["epoch_3_adapter"]["prediction"],
            0.39, 6.27, -1.17, -3.64,
        )
        print(f"{case['filename']}: words {len(left_list)}/{len(right_list)}, "
              f"font sizes {left_size}/{right_size}, audio {audio_seconds:.2f}s")

        # The same tracker drives the waveform cursor and both transcript panels.
        progress = ValueTracker(0.0)
        wave_left, wave_right, wave_y = -6.24, 6.24, 1.98
        wave_bars = VGroup()
        for index, level in enumerate(levels):
            x = wave_left + (index + 0.5) * (wave_right - wave_left) / len(levels)
            height = 0.11 + 0.54 * float(level)
            bar = Line([x, wave_y - height / 2, 0], [x, wave_y + height / 2, 0])
            bar.set_stroke(PALETTE["wave_inactive"], width=3.0, opacity=1)
            wave_bars.add(bar)
        cursor = Line([wave_left, 1.58, 0], [wave_left, 2.36, 0],
                      color=PALETTE["orange"], stroke_width=2.4)

        revealed = {"left": 0, "right": 0, "wave": 0}

        def reveal_words(group: Text, words: list[VGroup], key: str,
                         accent: str) -> None:
            target = min(len(words), int(progress.get_value() * len(words) + 1e-8))
            while revealed[key] < target:
                if revealed[key] > 0:
                    words[revealed[key] - 1].set_color(PALETTE["text"])
                words[revealed[key]].set_opacity(1)
                words[revealed[key]].set_color(accent)
                revealed[key] += 1

        left_words.add_updater(lambda group: reveal_words(group, left_list, "left", PALETTE["teal"]))
        right_words.add_updater(lambda group: reveal_words(group, right_list, "right", PALETTE["orange"]))

        def update_wave(_group: VGroup) -> None:
            target = min(len(wave_bars), int(progress.get_value() * len(wave_bars) + 1e-8))
            while revealed["wave"] < target:
                wave_bars[revealed["wave"]].set_color(PALETTE["teal"])
                revealed["wave"] += 1
            cursor.set_x(wave_left + progress.get_value() * (wave_right - wave_left))

        wave_bars.add_updater(update_wave)
        self.add(wave_panel, reference_panel, left_panel, right_panel, title,
                 location, wave_label, duration_label, ref_label, reference,
                 model_labels, separators, wave_bars, cursor, left_words, right_words)
        self.wait(0.45)
        self.add_sound(str(case["audio_path"]))
        self.play(progress.animate.set_value(1.0), run_time=audio_seconds, rate_func=linear)
        left_words.clear_updaters()
        right_words.clear_updaters()
        wave_bars.clear_updaters()
        for word, is_correct in zip(left_list,
                                    case["models"]["pretrained"]["word_is_correct"]):
            word.set_color(PALETTE["correct"] if is_correct else PALETTE["incorrect"])
            word.set_opacity(1)
        for word, is_correct in zip(right_list,
                                    case["models"]["epoch_3_adapter"]["word_is_correct"]):
            word.set_color(PALETTE["correct"] if is_correct else PALETTE["incorrect"])
            word.set_opacity(1)
        for bar in wave_bars:
            bar.set_color(PALETTE["teal"])
        cursor.set_x(wave_right)

        left_model = case["models"]["pretrained"]
        right_model = case["models"]["epoch_3_adapter"]
        left_scores = VGroup(
            _score_chip("CER", left_model["cer_percent"], PALETTE["teal"], -4.96, -0.82),
            _score_chip("WER", left_model["wer_percent"], PALETTE["teal"], -1.70, -0.82),
        )
        right_scores = VGroup(
            _score_chip("CER", right_model["cer_percent"], PALETTE["orange"], 1.70, -0.82),
            _score_chip("WER", right_model["wer_percent"], PALETTE["orange"], 4.96, -0.82),
        )
        # Scores and token correctness become visible together after audio ends.
        self.play(FadeIn(left_scores), FadeIn(right_scores), run_time=0.55)
        self.wait(2.65)


class DemoHaNoi(ASRComparison):
    filename = "30_0309.wav"


class DemoHoChiMinh(ASRComparison):
    filename = "59_0313.wav"


class DemoNgheAn(ASRComparison):
    filename = "37_0338.wav"


class DemoSuperHard(ASRComparison):
    filename = "test_super_hard_000000"


class PreviewPretrainedAlignment(Scene):
    """One-panel render of the screenshot's 30_0309 pretrained prediction."""

    default_camera_config = {"background_color": PALETTE["background"]}

    def construct(self) -> None:
        case = _read_case("30_0309.wav")
        model = case["models"]["pretrained"]
        audio_seconds = sf.info(str(case["audio_path"])).duration
        title = _left_text("Kiểm tra căn dòng prediction", -6.2, 3.26,
                           36, PALETTE["text"], "BOLD")
        subtitle = _left_text("30_0309.wav · Qwen3-ASR 1.7B", -2.94, 2.31,
                              22, PALETTE["muted"])
        panel = _panel(6.35, 3.62, 0, 0)
        label = _left_text("Qwen3-ASR 1.7B", -2.94, 1.43,
                           23, PALETTE["teal"], "BOLD")
        separator = Line([-2.94, 1.06, 0], [2.94, 1.06, 0],
                         color=PALETTE["border"], stroke_width=1.5)
        paragraph, words, size = _word_layout(model["prediction"],
                                               -2.94, 2.94, 0.97, -1.50)
        print(f"Preview: {len(words)} words, font size {size}, {audio_seconds:.2f}s")

        progress = ValueTracker(0.0)
        revealed = 0

        def reveal(_paragraph: Text) -> None:
            nonlocal revealed
            target = min(len(words), int(progress.get_value() * len(words) + 1e-8))
            while revealed < target:
                if revealed:
                    words[revealed - 1].set_color(PALETTE["text"])
                words[revealed].set_opacity(1)
                words[revealed].set_color(PALETTE["teal"])
                revealed += 1

        paragraph.add_updater(reveal)
        self.add(title, subtitle, panel, label, separator, paragraph)
        self.wait(0.35)
        self.add_sound(str(case["audio_path"]))
        self.play(progress.animate.set_value(1), run_time=audio_seconds, rate_func=linear)
        paragraph.clear_updaters()
        for word, is_correct in zip(words, model["word_is_correct"]):
            word.set_opacity(1)
            word.set_color(PALETTE["correct"] if is_correct else PALETTE["incorrect"])
        self.wait(2.0)
