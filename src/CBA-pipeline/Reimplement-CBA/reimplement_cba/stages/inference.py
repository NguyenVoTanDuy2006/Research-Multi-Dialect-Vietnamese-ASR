from reimplement_cba.adapters.whisper import PhoWhisperAdapter
from reimplement_cba.config import CBAConfig


class AdaptedInferenceStage:
    """Run the selected fine-tuned checkpoint to obtain current predictions."""

    def __init__(self, config: CBAConfig, adapter_path: str):
        self.model = PhoWhisperAdapter(config, adapter_path=adapter_path)

    def run(self, records, audio_reader):
        raise NotImplementedError(
            "TODO: generate current hypotheses and persist them with sample IDs"
        )
