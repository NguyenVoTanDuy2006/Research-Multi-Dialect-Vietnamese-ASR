# MDV ASR fine-tuning and benchmark source

The active `mdv_asr` package under `src/` contains only reusable code for ASR fine-tuning and benchmarking: dataset/audio access required by those workflows, model backends, AdaLoRA training, evaluation metrics, reports, and checkpoint auditing. Notebook entrypoints in `nb/` configure and call this package. Dataset sampling/packaging, lexical or other exploratory analysis, and old implementations are kept outside the active package (in `legacy/` where applicable).

The package folders are intentionally limited to:

- `config/`, `core/`, `runtime/`: experiment settings, shared errors/artifacts, and local/Kaggle/DDP launch support.
- `data/`: dataset indexing, Arrow/Parquet/package readers, manifests, audio decode and preview used by training/benchmark.
- `models/`, `training/`: ASR model backends, AdaLoRA configuration (`models/adalora.py`), Qwen+AdaLoRA trainer (`training/ada_lora.py`), and checkpoint audit.
- `evaluation/`, `workflows/`: prediction records, CER/WER/alignment/reports, and end-to-end train/benchmark flows.

## Install

For benchmark/data access:

```bash
python -m pip install -e ".[data]"
```

For Qwen/AdaLoRA training, install the training extra in a compatible environment:

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

## Notebook scripts

Python notebook scripts in `nb/` use Jupyter percent-cell markers. Benchmark/training entrypoints can be exported to `.ipynb` with:

```bash
python scripts/export_notebooks.py
```

The exporter skips existing `.ipynb` files by default. Datasets, checkpoints, outputs and third-party repositories are external runtime inputs; they are not bundled as source.

## Kaggle and benchmarks

Build a code-only Kaggle bundle (wheel, configs, checksums; no dataset, model, or token):

```bash
python scripts/build_kaggle_bundle.py --output outputs/kaggle_code_bundle
```

Before running `configs/benchmark/vimd_three_models.json`, set both adapter paths
to the local or Kaggle Input locations of the small- and full-ViMD adapters. Model
weights and datasets are deliberately not stored in this code repository. Use
`python -m pytest` to run the unit tests when the `dev` extra is installed.
