from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedText:
    text: str
    tokens: tuple[str, ...]
    characters: tuple[str, ...]


class TextNormalizer:
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
        return NormalizedText(normalized, tuple(normalized.split()), tuple(normalized.replace(" ", "")))

