from __future__ import annotations

import ast
from pathlib import Path

from mdv_asr_module.configs.loader import ConfigLoader


def test_benchmark_notebook_uses_base_first_and_all_four_adapter_configs():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "src/nb/benchmark_qwen.py"
    notebook = ast.parse(notebook_path.read_text(encoding="utf-8"))
    assignments = [
        node for node in ast.walk(notebook)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "MODEL_CONFIG_FILES"
                for target in node.targets)
    ]
    assert len(assignments) == 1

    namespace = {"Path": Path, "REPO_DIR": repo_root}
    cell = ast.Module(body=assignments, type_ignores=[])
    exec(compile(cell, str(notebook_path), "exec"), namespace)
    model_paths = namespace["MODEL_CONFIG_FILES"]

    config = ConfigLoader().load_benchmark_configs(
        repo_root / "src/mdv_asr_module/configs/datasets/vimd_full.json",
        model_paths,
        output_dir=repo_root / "outputs/test-only",
    )
    models = config.benchmark.models

    assert [model.key for model in models] == [
        "qwen_vimd_base",
        "qwen_vimd_adalora_small",
        "qwen_vimd_adalora_full",
        "qwen_vimd_pissa_small",
        "qwen_vimd_pissa_full",
    ]
    assert models[0].model_id == "Qwen/Qwen3-ASR-1.7B-hf"
    assert models[0].adapter_path is None
    assert all(model.adapter_path for model in models[1:])


def test_adapter_configs_use_the_attached_kaggle_model_paths():
    repo_root = Path(__file__).resolve().parents[2]
    configs_dir = repo_root / "src/mdv_asr_module/configs/models"
    expected_paths = {
        "qwen_vimd_adalora_small": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/adalora-small/V1",
        "qwen_vimd_adalora_full": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/adalora-full/V1",
        "qwen_vimd_pissa_small": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/pissa-small/V1",
        "qwen_vimd_pissa_full": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/pissa-full/V1",
    }

    loader = ConfigLoader()
    actual_paths = {
        loader.load(configs_dir / f"{key}.json").model.key:
        loader.load(configs_dir / f"{key}.json").model.adapter_path
        for key in expected_paths
    }

    assert actual_paths == expected_paths
