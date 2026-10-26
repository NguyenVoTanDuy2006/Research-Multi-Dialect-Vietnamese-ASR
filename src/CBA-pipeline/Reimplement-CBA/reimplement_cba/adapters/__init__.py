"""Adapters to the shared ViMD data/model/metric packages.

Keep optional audio, training, and scoring dependencies lazy so config
inspection can run before the full model environment is installed.
"""

from __future__ import annotations

from importlib import import_module


_EXPORT_MODULES = {
    "SemScoreComponents": ".metrics",
    "ViMDMetricsAdapter": ".metrics",
    "PhoWhisperDataCollator": ".whisper_data",
    "PhoWhisperTrainingDataset": ".whisper_data",
}

__all__ = [
    "PhoWhisperDataCollator",
    "PhoWhisperTrainingDataset",
    "SemScoreComponents",
    "ViMDMetricsAdapter",
]


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
