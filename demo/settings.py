from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path


@dataclass(frozen=True)
class DemoConfig:
    case_ids: tuple[str, ...] = ()
    baseline_key: str = "qwen_base"
    adapted_key: str = "qwen_epoch_3"
    audio_overrides: dict[str, str] = field(default_factory=dict)
    width: int = 1920
    height: int = 1080
    fps: int = 30
    font: str = "Noto Sans"
    palette: dict[str, str] = field(default_factory=dict)
    output_dir: str = "demo/rendered"

    def validate(self) -> None:
        if min(self.width, self.height, self.fps) < 1:
            raise ValueError("Demo dimensions and fps must be positive")
        if any(not key or not value for key, value in self.audio_overrides.items()):
            raise ValueError("Audio override keys and paths must be nonempty")

    @classmethod
    def load(cls, path: str | Path) -> "DemoConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Demo config must be a JSON object")
        allowed = {item.name for item in fields(cls)}
        unknown = set(payload) - allowed
        if unknown:
            raise ValueError(f"Unknown demo config fields: {sorted(unknown)}")
        values = dict(payload)
        if isinstance(values.get("case_ids"), list):
            values["case_ids"] = tuple(values["case_ids"])
        config = cls(**values)
        config.validate()
        return config
