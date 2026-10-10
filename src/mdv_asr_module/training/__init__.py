"""Lazy exports: method classes do not import model-specific trainers."""
from importlib import import_module

_EXPORTS = {
    "CheckpointManager": ".checkpoints", "QwenTrainingCollator": ".collator",
    "ASRAudioDataset": ".dataset", "ASRDataLoaderFactory": ".dataset",
    "OptimizerStepController": ".optimization", "StepResult": ".optimization",
    "TrainProgress": ".optimization", "QwenAdaLoraTrainer": ".ada_lora",
    "QwenTrainer": ".qwen", "PiSSAMethod": ".methods", "AdaLoRAMethod": ".methods",
    "FineTuningMethod": ".methods",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value

