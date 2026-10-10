from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DemoTheme:
    palette: dict[str, str] = field(default_factory=lambda: {
        "background": "#0F1624", "panel": "#18243A", "raised": "#1E2B42",
        "border": "#354860", "text": "#EEF1F5", "muted": "#9AA7BA",
        "teal": "#2BB9AD", "orange": "#F4A259", "correct": "#4DCCA5",
        "incorrect": "#FF7777", "wave_inactive": "#41526A",
    })
    font: str = "Noto Sans"
    base_font_size: int = 28
    min_font_size: int = 16
    line_spacing: float = 0.34
    horizontal_padding: float = 0.22
    vertical_padding: float = 0.18

    @classmethod
    def from_config(cls, value: dict[str, Any] | None = None) -> "DemoTheme":
        value = dict(value or {})
        defaults = cls()
        return cls(palette={**defaults.palette, **value.get("palette", {})},
                   font=value.get("font", defaults.font),
                   base_font_size=int(value.get("base_font_size", defaults.base_font_size)),
                   min_font_size=int(value.get("min_font_size", defaults.min_font_size)),
                   line_spacing=float(value.get("line_spacing", defaults.line_spacing)),
                   horizontal_padding=float(value.get("horizontal_padding", defaults.horizontal_padding)),
                   vertical_padding=float(value.get("vertical_padding", defaults.vertical_padding)))

    def validate(self) -> None:
        required = {"background", "panel", "text", "correct", "incorrect"}
        missing = required - self.palette.keys()
        if missing:
            raise ValueError(f"Theme palette is missing: {sorted(missing)}")
        if self.min_font_size < 1 or self.base_font_size < self.min_font_size:
            raise ValueError("Invalid theme font sizes")

    def validate_fonts(self) -> dict[str, bool]:
        from matplotlib import font_manager

        available = {font.name for font in font_manager.fontManager.ttflist}
        return {self.font: self.font in available}
