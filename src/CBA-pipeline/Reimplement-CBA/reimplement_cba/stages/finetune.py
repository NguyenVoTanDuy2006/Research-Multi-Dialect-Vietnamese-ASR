from reimplement_cba.config import CBAConfig


class AdaLoRATrainingStage:
    """Trainer boundary for one AdaLoRA curriculum round."""

    def __init__(self, config: CBAConfig):
        self.config = config

    def run(self, records, *, resume_adapter: str | None = None) -> str:
        raise NotImplementedError(
            "TODO: build the Whisper Seq2Seq dataset and train AdaLoRA"
        )
