"""Compatibility import; implementations live under training.methods."""
from mdv_asr_module.training.methods.ada_lora import AdaLoRAMethod


class AdaLoraManager(AdaLoRAMethod):
    """Backward-compatible name for AdaLoRAMethod."""
