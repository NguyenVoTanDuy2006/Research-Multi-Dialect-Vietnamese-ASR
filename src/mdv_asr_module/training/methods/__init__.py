from .base import FineTuningMethod
from .targets import ModuleTargetSelector
from .ada_lora import AdaLoRAMethod
from .pissa import PiSSAMethod


def create_method(config):
    if config.method == "adalora":
        return AdaLoRAMethod(config.adalora)
    if config.method == "pissa":
        return PiSSAMethod(config.pissa)
    raise ValueError(f"Unknown fine-tuning method: {config.method}")


__all__ = ["FineTuningMethod", "ModuleTargetSelector", "AdaLoRAMethod", "PiSSAMethod", "create_method"]
