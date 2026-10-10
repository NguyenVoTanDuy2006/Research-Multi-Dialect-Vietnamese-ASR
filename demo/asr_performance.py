"""Compatibility entrypoint for the ManimGL ASR comparison scene."""

from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "src"))
from demo.scenes import ASRComparisonScene

__all__ = ["ASRComparisonScene"]
