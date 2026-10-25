from reimplement_cba.adapters.whisper import PhoWhisperAdapter
from reimplement_cba.config import CBAConfig


class InitialInferenceStage:
    """Run the unadapted PhoWhisper baseline over labelled ViMD samples."""

    def __init__(self, config: CBAConfig):
        self.model = PhoWhisperAdapter(config)

    def run(self, records, audio_reader):
        raise NotImplementedError(
            "TODO: transcribe all selected samples and persist baseline hypotheses"
        )
