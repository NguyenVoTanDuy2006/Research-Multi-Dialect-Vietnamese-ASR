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

## Optional semantic metrics

WER/CER stay in `evaluation/error_metrics.py`. Semantic metrics are separate in
`evaluation/similarity_metrics.py`; both groups implement the abstract
`evaluation/metric.py` API. Install the optional encoder dependencies with:

```bash
python -m pip install -e ".[semantic]"
```

`SemSimE5` uses `intfloat/multilingual-e5-base`, symmetric `query: ` prefixes,
attention-masked mean pooling and cosine similarity. `BERTScoreMetric` wraps
the official BERTScore implementation with `FacebookAI/xlm-roberta-large`,
layer 17, IDF and baseline rescaling disabled. Both lazily load and reuse one
model instance; they accept text pairs or aligned batches and summarize
sentence-level mean, median and 5th percentile. Their scores are similarity
indicators, not percentages of semantic correctness, and do not replace WER/CER
or manual checks for negation, names and numbers.

```python
from mdv_asr_module.evaluation.similarity_metrics import BERTScoreMetric, SemSimE5

e5 = SemSimE5(device="cuda", batch_size=16)
bert = BERTScoreMetric(device="cuda", batch_size=8)
e5_scores = e5.score_batch(references, hypotheses)
bert_scores = bert.score_batch(references, hypotheses)
summary = {**e5.aggregate(e5_scores), **bert.aggregate(bert_scores)}
metric_config = {"e5": e5.metadata(), "bert_score": bert.metadata()}
```

Input rows must have a non-empty string reference and string hypothesis. Empty
hypotheses remain included with similarity/distance `0/1` for E5 and
precision/recall/F1 `0` for BERTScore. Inputs exceeding the 512-token limit
(including prefixes/special tokens) raise instead of being silently truncated.
For repeatable comparisons, use an immutable local model snapshot for
BERTScore and pin `revision=` for E5. Run the metrics separately on each model's
paired references/sample IDs; the benchmark workflow does not run them
automatically.

## Benchmarks and tests

Before running `configs/benchmark/vimd_three_models.json`, set both adapter paths
to the local or Kaggle Input locations of the small- and full-ViMD adapters. Model
weights and datasets are deliberately not stored in this code repository. Use
`python -m pytest` to run the unit tests when the `dev` and `train` extras are
installed. Tests use tiny random models on CPU without downloading pretrained
weights or running ViMD training; Kaggle's two-GPU smoke is still required.
