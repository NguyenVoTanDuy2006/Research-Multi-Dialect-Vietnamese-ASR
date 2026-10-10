"""Compatibility entrypoint; the shared Qwen training loop lives in qwen.py."""
from .qwen import QwenTrainer
from .methods import AdaLoRAMethod


class QwenAdaLoraTrainer(QwenTrainer):
    def __init__(self, **kwargs):
        if "method" not in kwargs:
            kwargs["method"] = AdaLoRAMethod(kwargs["train_config"].adalora)
        super().__init__(**kwargs)

    @property
    def ada_manager(self):
        return self.method
