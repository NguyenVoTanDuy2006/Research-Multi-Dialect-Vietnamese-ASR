# %% [markdown]
# # NeMo ASR benchmark
# NeMo is imported lazily only when the configured NeMo backend is loaded.

# %%
from pathlib import Path

try:
    from _bootstrap import initialize_project
except ImportError:
    from nb._bootstrap import initialize_project

ROOT = initialize_project()
from mdv_asr.config.loader import ConfigLoader
from mdv_asr.workflows.dataset import DatasetWorkflow
from mdv_asr.workflows.benchmark import ASRBenchmarkWorkflow

CONFIG_PATH = ROOT / "configs/benchmark/sr_nemo.json"
config = ConfigLoader().load(CONFIG_PATH)
dataset = DatasetWorkflow.from_config(config.dataset)
benchmark = ASRBenchmarkWorkflow(config, dataset)
try:
    results = benchmark.run_test()
    print(benchmark.report())
finally:
    benchmark.close()
    dataset.close()
