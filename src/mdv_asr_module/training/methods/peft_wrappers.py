"""Generic multimodal PEFT forward preserving AdaLoRA's native regularizer."""
from peft import PeftModel


class AdaLoRAPeftModel(PeftModel):
    def forward(self, *args, **kwargs):
        # Generic PeftModel.forward calls get_base_model() directly, bypassing
        # AdaLoraModel.forward and its orthogonality regularization. Delegate to
        # the tuner, without a CAUSAL_LM wrapper injecting text-only arguments.
        with self._enable_peft_forward_hooks(*args, **kwargs):
            kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
            return self.base_model(*args, **kwargs)
