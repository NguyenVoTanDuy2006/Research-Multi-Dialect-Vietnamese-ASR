from __future__ import annotations

import unicodedata


class VietnamesePostprocessor:
    """Normalize display text while preserving the decoded spoken content."""

    def process(self, transcript: str) -> str:
        if not isinstance(transcript, str):
            raise TypeError(f"Transcript must be a string, got {type(transcript).__name__}")
        return " ".join(unicodedata.normalize("NFC", transcript).split())
