class ViMDMetricsAdapter:
    """Metric boundary for per-sample curriculum and validation evaluation."""

    def score_pair(self, reference: str, hypothesis: str) -> float:
        raise NotImplementedError(
            "TODO: connect a Vietnamese-compatible SemScore implementation"
        )

    def evaluate_corpus(self, references: list[str], hypotheses: list[str]) -> dict[str, float]:
        raise NotImplementedError(
            "TODO: return aggregate WER and SemScore on validation data"
        )
