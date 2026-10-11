# MDV ASR fine-tuning and benchmark source

The active `mdv_asr_module` package under `src/` contains reusable ASR fine-tuning and benchmarking code. New notebook entrypoints are Python percent-cell scripts in `src/nb/`; older scripts remain in `nb/`. Datasets, checkpoints, outputs, and third-party repositories are external inputs, not source artifacts.

The package folders are intentionally limited to:

- `config/`, `core/`, `runtime/`: experiment settings, shared errors/artifacts, and local/Kaggle/DDP launch support.
- `data/`: dataset indexing, Arrow/Parquet/package readers, manifests, audio decode and preview used by training/benchmark.
- `models/`: ASR inference backends and checkpoint auditing. `models/adalora.py` is a compatibility import only.
- `training/methods/`: model-independent `FineTuningMethod`, `ModuleTargetSelector`, `AdaLoRAMethod`, and `PiSSAMethod`.
- `training/qwen.py`: Qwen-specific forward, collation, validation and training loop; `training/ada_lora.py` preserves the older trainer entrypoint.
- `evaluation/`, `workflows/`: prediction records, CER/WER/alignment/reports, and end-to-end train/benchmark flows.

## Install

For benchmark/data access:

```bash
python -m pip install -e ".[data]"
```

For Qwen fine-tuning, install the training extra in a compatible environment:

```bash
python -m pip install -e ".[data,train]"
```

The project intentionally keeps Torch and NumPy out of its base wheel dependency resolution so Kaggle can install the project wheel with `pip install --no-deps` and retain its preinstalled runtime stack. NeMo is optional.

## CLI

```bash
mdv-asr config validate --config configs/training/qwen_full_vimd_2epoch.json
mdv-asr train smoke --config configs/training/qwen_full_vimd_2epoch.json
mdv-asr benchmark run --config configs/benchmark/vimd_three_models.json
```

Set `HF_TOKEN` in the environment or create a Kaggle Secret named `HF_TOKEN`; never put access tokens in source/config files. Dataset download is opt-in in configs; prefer attaching the dataset through Kaggle Input.

## PiSSA and Kaggle

Use `src/nb/kaggle_finetune_qwen_pissa.py`: fill in the repository URL, attach
ViMD, enable T4 x2, then execute the cells. It installs the API versions in
`requirements-kaggle.txt`, clones/installs this source, builds an index once,
and runs smoke → train → evaluation through the existing worker. New notebooks
are saved as `.py` only. If Kaggle needs an `.ipynb`, convert a copy with
`jupytext --to ipynb src/nb/kaggle_finetune_qwen_pissa.py`.

Select `train.method = "pissa"` or `"adalora"` in config. The example is
`configs/training/qwen_full_vimd_pissa_2epoch.json`: rank 8, alpha 16, dropout 0,
`pissa_niter_4`, common initialization seed 24127159, q/v projections in both
audio encoder and decoder, 2 epochs. These are starting settings, not a claim
of improved CER/WER. The same method object can attach to a different model
with configurable Linear-layer suffixes or a regex; it does not own an ASR
processor or assume a Qwen architecture. Quantized/TP/FSDP training and a
Cohere training backend are not implemented here.

```python
from mdv_asr_module.config import PiSSASettings
from mdv_asr_module.training.methods import PiSSAMethod

method = PiSSAMethod(PiSSASettings(r=8, target_modules=("q_proj", "v_proj")))
model = method.attach(fresh_pretrained_model)
# Trainer: backward → successful optimizer step → method.after_optimizer_step(n)
#          → zero_grad. Do not call the method hook for a skipped AMP update.
method.save_initialization("run/pissa_initial")
# ... training ...
method.save_adapter("run/epoch_1_adapter")
```

PiSSA uses PEFT's initialization and conversion, not a custom SVD algorithm.
All base parameters are frozen after PEFT splits target weights into principal
factors and residuals. The shared Qwen loop still trains audio → transcript;
province/region are metadata, not inputs.

Artifacts have distinct purposes:

- `pissa_initial/`: immutable initial adapter factors for conversion.
- `training_checkpoints/epoch_XXXX.pt`: native resume state including initial
  factors, **targeted residual weights**, trained factors, optimizer, scheduler,
  scaler, progress, and per-rank RNG. Residuals make fp16/bf16 resume exact but
  increase checkpoint size. Resume on a fresh trainer with the same config,
  model revision, and DDP world size; only load trusted `.pt` files.
- `epoch_N_adapter/`: PiSSA converted to ordinary rank-**2r** LoRA for inference
  on the **original pretrained base**, via `PeftModel.from_pretrained` like
  AdaLoRA. Export rank 16 for training rank 8 is expected. Float16 residual
  rounding can introduce small numerical differences after conversion; final
  evaluation uses this exported adapter, not an assumed exact low-precision
  identity. Do not use it to resume the native rank-r training run.
- AdaLoRA exports retain full-shaped factors with masked singular values,
  avoiding empty-rank tensors/key-remapping warnings while preserving the
  adaptive allocation in native resume state and `training_method.json`.

Both methods share the optimizer-step lifecycle and checkpoint interface.
Existing AdaLoRA JSON configs continue to work without `train.method`.

## Semantic metrics (optional, after ASR inference)

Install with `python -m pip install -e ".[semantic]"`. This extra adds
`bert-score==0.3.13` and the metric encoder dependencies; ordinary WER/CER
imports do not load Torch, Transformers, or any pretrained model. The first
nonempty `score`/`score_batch` call loads metric weights from HF or cache.
On Kaggle, enable Internet for that initial download or pass a local model
snapshot directory. No ASR checkpoint or audio is needed to score saved texts.

The public metric API is grouped in `evaluation/metrics/`:

- `metric.py` defines the abstract `Metric` interface. `error_metrics.py`
  contains WER/CER and corpus aggregation; `similarity_metrics.py` contains
  both semantic metrics and their result types.
- `similarity_metrics.py`: `SemSimE5`, symmetric cosine similarity with
  `intfloat/multilingual-e5-base`, `query: ` on both texts, attention-masked
  mean pooling and L2 normalization. Returns `SemSimE5Result(similarity,
  distance)` where distance is `1 - similarity`, not independent evidence.
  This is **SemSim-E5**, not the original English SemScore configuration.
- The same `similarity_metrics.py` contains `BERTScoreMetric`, the official
  BERTScore implementation with `FacebookAI/xlm-roberta-large`, layer 17, fast tokenizer,
  `idf=False`, `rescale_with_baseline=False`. Returns
  `BERTScoreResult(precision, recall, f1)`. Layer 17 follows the upstream
  XLM-R-large setting; it has not been optimized on ViMD test data.

Both classes expose `score(reference, prediction)`,
`score_batch(references, predictions)`, `aggregate(scores)` and `metadata()`.
Keep one instance of each encoder across pretrained, small-AdaLoRA and
full-AdaLoRA predictions. Group by model before aggregating; use the **same
sample IDs and references**, and do not silently discard inference errors.
`PredictionRecord` objects from the benchmark can be used directly:

```python
from mdv_asr_module.evaluation.metrics import SemSimE5, BERTScoreMetric

e5 = SemSimE5(device="cuda", batch_size=16)
bert = BERTScoreMetric(device="cuda", batch_size=8)

# model_predictions: successful PredictionRecord rows for ONE ASR model.
if any(row.error for row in model_predictions):
    raise ValueError("Resolve inference errors before semantic evaluation")
references = [row.reference for row in model_predictions]
hypotheses = [row.prediction for row in model_predictions]
e5_scores = e5.score_batch(references, hypotheses)
bert_scores = bert.score_batch(references, hypotheses)
summary = {**e5.aggregate(e5_scores), **bert.aggregate(bert_scores)}
per_sample = [
    {**row.to_dict(), "SemSim_E5": a.similarity, "SemDist_E5": a.distance,
     "BERTScore_precision": b.precision, "BERTScore_recall": b.recall,
     "BERTScore_F1": b.f1}
    for row, a, b in zip(model_predictions, e5_scores, bert_scores)
]
# Save these alongside results, especially encoder revision/hash and versions.
metric_config = {"e5": e5.metadata(), "bert_score": bert.metadata()}
```

Summaries use an **unweighted sentence mean**, median and 5th percentile;
they are not corpus edit-error rates or percentages of semantic correctness.
Subset the scored rows to aggregate by region/province without re-encoding.
Preprocessing is NFC plus whitespace normalization only: accents, case,
punctuation, numbers, negation, dialect vocabulary and English are preserved.
Empty references and non-string inputs (including CSV NaN) raise; an empty
hypothesis is assigned E5 similarity 0/distance 1 and BERTScore P/R/F1 0 by
explicit policy and remains in the denominator. Empty aggregate metrics are
`None`, not zero. Convert genuinely empty CSV hypothesis cells to `""` when
reading; do not stringify NaN as `"nan"`.

The default token ceiling is 512 **including special tokens and E5 prefixes**,
or the tokenizer's lower limit. Overlong text raises instead of silently
truncating; decide a consistent long-utterance policy before benchmarking.
Use `revision=` with E5 or immutable local snapshots for either metric to pin
weights. Store `metadata()` **after** scoring; a local snapshot may not expose
an HF commit hash, so also record the snapshot provenance in your run config.
Semantic similarity can miss negation/name/number errors; keep WER/CER and
human review. The existing training/benchmark workflows do not automatically
run these optional metrics or change checkpoint selection.

Implementation references: [E5 model card](https://huggingface.co/intfloat/multilingual-e5-base),
[BERTScore](https://github.com/Tiiiger/bert_score).

## Benchmarks and tests

Before running `configs/benchmark/vimd_three_models.json`, set both adapter paths
to the local or Kaggle Input locations of the small- and full-ViMD adapters. Model
weights and datasets are deliberately not stored in this code repository. Use
`python -m pytest` to run the unit tests when the `dev` and `train` extras are
installed. Tests use tiny random models on CPU without downloading pretrained
weights or running ViMD training; Kaggle's two-GPU smoke is still required.
Semantic metric tests additionally require the `semantic` extra and use a
tiny locally generated encoder with the real BERTScore backend, not downloaded
E5/XLM-R weights. They verify implementation behavior, not quality on ViMD.
