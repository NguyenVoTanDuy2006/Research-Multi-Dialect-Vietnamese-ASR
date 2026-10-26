from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def full_rank_save_config(model: Any) -> Iterator[dict[str, Any]]:
    """Save AdaLoRA factor tensors at their initialized dimensions."""
    configurations = getattr(model, "peft_config", None)
    if not isinstance(configurations, dict) or not configurations:
        raise RuntimeError("Cannot save adapter: model does not expose PEFT configurations")
    rank_patterns = {
        name: getattr(configuration, "rank_pattern", None)
        for name, configuration in configurations.items()
    }
    for configuration in configurations.values():
        configuration.rank_pattern = None
    try:
        yield rank_patterns
    finally:
        for name, configuration in configurations.items():
            configuration.rank_pattern = rank_patterns[name]
