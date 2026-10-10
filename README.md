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

## Benchmarks and tests

Before running `configs/benchmark/vimd_three_models.json`, set both adapter paths
to the local or Kaggle Input locations of the small- and full-ViMD adapters. Model
weights and datasets are deliberately not stored in this code repository. Use
`python -m pytest` to run the unit tests when the `dev` and `train` extras are
installed. Tests use tiny random models on CPU without downloading pretrained
weights or running ViMD training; Kaggle's two-GPU smoke is still required.
