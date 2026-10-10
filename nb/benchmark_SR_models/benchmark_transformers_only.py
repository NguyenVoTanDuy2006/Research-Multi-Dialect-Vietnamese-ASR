# %% [markdown]
# # Transformers ASR backends benchmark
# Configure Qwen, ARK, Cohere and/or Granite in `configs/benchmark/sr_transformers.json`.

# %%
from pathlib import Path

try:
    from _bootstrap import initialize_project
except ImportError:
    from nb._bootstrap import initialize_project

ROOT = initialize_project()
from mdv_asr_module.config.loader import ConfigLoader
from mdv_asr_module.workflows.dataset import DatasetWorkflow
from mdv_asr_module.workflows.benchmark import ASRBenchmarkWorkflow

CONFIG_PATH = ROOT / "configs/benchmark/sr_transformers.json"
config = ConfigLoader().load(CONFIG_PATH)
dataset = DatasetWorkflow.from_config(config.dataset)
benchmark = ASRBenchmarkWorkflow(config, dataset)
try:
    results = benchmark.run_test()
    print(benchmark.report())
finally:
    benchmark.close()
    dataset.close()
