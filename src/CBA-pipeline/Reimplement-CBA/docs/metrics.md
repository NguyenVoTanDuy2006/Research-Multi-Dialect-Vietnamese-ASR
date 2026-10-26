# ASR metrics

`reimplement_cba.adapters.metrics.ViMDMetricsAdapter` provides sentence-level SemScore and corpus-level WER/SemScore for the ViMD CBA pipeline.

## SemScore components

Each component is normalized to `[0, 1]`; SemScore is reported on `[0, 100]`:

```text
SemScore = 100 × (w_nli × NLI + w_bert × BERTScore_F1 + w_phone × PhoneticSimilarity)
```

The default weights are equal and can be changed in `configs/vimd_phowhisper_medium.json`. They are a configurable starting point for this reimplementation; the CBA paper does not specify these Vietnamese component weights.

- **NLI:** multilingual mDeBERTa-v3 MNLI/XNLI. Score the entailment probability in both directions (`reference → hypothesis` and `hypothesis → reference`) and take their mean. The selected model card covers Vietnamese among its evaluated XNLI languages; the reported Vietnamese XNLI accuracy is 0.795, so this is a useful starting scorer, not a validated ViMD metric. [Model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli)
- **BERTScore-F1:** multilingual XLM-R-large BERTScore F1 without baseline rescaling, using layer 17. Values are clamped to `[0, 1]` before entering the composite score.
- **Phonetic similarity:** `1 - normalized Levenshtein distance` over atomic IPA phones, tone IDs, and syllable boundaries after converting Quốc ngữ with the vocabulary in `resources/phonetics/`. When a known province or macro-dialect is passed, the converter applies the matching implemented regional rules. With no region, it compares canonical phones.

The phonetic converter uses the paper-derived spelling inventory and explicit transformations. Ambiguous surface rules that the PDF does not specify reliably are not guessed. Unsupported syllables remain as marked grapheme tokens, so they do not disappear from the score.

## WER

Corpus WER is returned as a ratio in `[0, 1]`, computed over whitespace-delimited Vietnamese syllables after NFC normalization, case-folding, and punctuation removal. The metric returns both corpus micro-WER and mean sentence WER. SemScore and its three component means are macro-averaged over aligned sample pairs.

## Use

```python
from reimplement_cba.adapters.metrics import ViMDMetricsAdapter
from reimplement_cba.config import CBAConfig

config = CBAConfig.from_file("configs/vimd_phowhisper_medium.json")
metrics = ViMDMetricsAdapter.from_config(config)

# For curriculum scoring, supply the sample's region/province when available.
score = metrics.score_pair(reference, hypothesis, region="Quảng Nam")

# For checkpoint evaluation, keep the region list aligned with the transcripts.
report = metrics.evaluate_corpus(references, hypotheses, regions=provinces)
```

Model weights are downloaded lazily on the first non-empty pair. From the workspace root (the directory containing `pyproject.toml`), install the optional dependencies with:

```bash
pip install -e '.[semantic]'
```

When running the example from `Reimplement-CBA/`, set the package path as in the project README, for example `PYTHONPATH=../../../src python your_script.py`.

The result dictionary contains `WER` as a ratio, `semscore` and `SemScore` as 0–100 aliases, plus `NLI_mean`, `BERTScore_F1_mean`, and `PhoneticSimilarity_mean`. `score_pair` returns only the 0–100 SemScore expected by the curriculum adapter; use `score_components` to inspect the component scores.
