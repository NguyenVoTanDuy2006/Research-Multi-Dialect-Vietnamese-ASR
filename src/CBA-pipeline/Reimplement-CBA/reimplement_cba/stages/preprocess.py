class AudioPreprocessor:
    """CBA audio preprocessing boundary for ViMD audio."""

    def process(self, audio):
        raise NotImplementedError(
            "TODO: add loudness normalization and optional VAD/cut-and-merge"
        )
