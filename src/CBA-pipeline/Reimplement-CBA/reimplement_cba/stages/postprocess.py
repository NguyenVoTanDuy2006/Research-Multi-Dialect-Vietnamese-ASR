class VietnamesePostprocessor:
    """Vietnamese-safe final transcript postprocessing boundary."""

    def process(self, transcript: str) -> str:
        raise NotImplementedError(
            "TODO: add repetition cleanup and Vietnamese normalization without stripping diacritics"
        )
