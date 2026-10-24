from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from mdv_asr_module.configs.loader import ConfigLoader


def test_benchmark_notebook_uses_base_first_and_all_four_adapter_configs():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "src/nb/benchmark.py"
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


def test_benchmark_notebook_enables_semantic_config_and_optional_bert_score():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "src/nb/benchmark.py"
    notebook = ast.parse(notebook_path.read_text(encoding="utf-8"))
    settings_node = next(
        node for node in notebook.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name)
                and target.id == "SEMANTIC_METRICS_SETTINGS" for target in node.targets)
    )
    settings_ns = {"Path": Path}
    exec(compile(ast.Module(body=[settings_node], type_ignores=[]), str(notebook_path), "exec"), settings_ns)
    loader = ConfigLoader()
    base_config = loader.load_benchmark_configs(
        repo_root / "src/mdv_asr_module/configs/datasets/vimd_full.json",
        [repo_root / "src/mdv_asr_module/configs/models/qwen_vimd_base.json"],
        output_dir=repo_root / "outputs/test-only",
    )
    config = loader.with_overrides(
        base_config, {"semantic_metrics": settings_ns["SEMANTIC_METRICS_SETTINGS"]}
    )
    assert config.semantic_metrics.enabled is True
    assert config.semantic_metrics.metrics == ("semsim_e5", "bertscore")
    assert config.semantic_metrics.device is None
    assert config.semantic_metrics.batch_size == 8
    assert config.semantic_metrics.max_length == 512
    assert config.semantic_metrics.long_text_policy == "chunk"

    function_node = next(
        node for node in notebook.body
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_bert_score_available"
    )
    calls = []
    specs = iter([None, object()])
    namespace = {
        "importlib": SimpleNamespace(
            util=SimpleNamespace(find_spec=lambda name: next(specs)),
            invalidate_caches=lambda: None,
        ),
        "subprocess": SimpleNamespace(run=lambda command, check: calls.append(command)
                                      or SimpleNamespace(returncode=0)),
        "sys": SimpleNamespace(executable="python"),
    }
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(notebook_path), "exec"), namespace)
    assert namespace["ensure_bert_score_available"](
        True, ("semsim_e5", "bertscore")
    ) is True
    assert calls == [["python", "-m", "pip", "install", "bert-score==0.3.13"]]


def test_benchmark_notebook_does_not_install_optional_dependency_when_disabled():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "src/nb/benchmark.py"
    notebook = ast.parse(notebook_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in notebook.body
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_bert_score_available"
    )
    calls = []
    namespace = {
        "importlib": SimpleNamespace(
            util=SimpleNamespace(find_spec=lambda name: calls.append(("find_spec", name))),
            invalidate_caches=lambda: None,
        ),
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
        "sys": SimpleNamespace(executable="python"),
    }
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(notebook_path), "exec"), namespace)
    assert namespace["ensure_bert_score_available"](False, ("bertscore",)) is False
    assert calls == []


def test_benchmark_notebook_reuses_installed_bert_score_and_reports_semantic_paths(
    tmp_path, capsys,
):
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "src/nb/benchmark.py"
    notebook = ast.parse(notebook_path.read_text(encoding="utf-8"))
    settings_node = next(
        node for node in notebook.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "ARTIFACT_NAMES"
                for target in node.targets)
    )
    installed = tmp_path / "semantic_metrics_overall.csv"
    installed.touch()
    settings_ns = {"Path": Path}
    exec(compile(ast.Module(body=[settings_node], type_ignores=[]), str(notebook_path), "exec"), settings_ns)
    function_nodes = [
        node for node in notebook.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"ensure_bert_score_available", "print_artifact_paths"}
    ]
    pip_calls = []
    namespace = {
        "Path": Path,
        "importlib": SimpleNamespace(util=SimpleNamespace(find_spec=lambda name: object())),
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: pip_calls.append(args)),
        "sys": SimpleNamespace(executable="python"),
    }
    exec(compile(ast.Module(body=function_nodes, type_ignores=[]), str(notebook_path), "exec"), namespace)
    assert namespace["ensure_bert_score_available"](True, ("semsim_e5", "bertscore")) is True
    assert pip_calls == []
    namespace["print_artifact_paths"](tmp_path, settings_ns["ARTIFACT_NAMES"])
    output = capsys.readouterr().out
    assert "semantic_metrics_overall.csv" in settings_ns["ARTIFACT_NAMES"]
    assert "semantic_metrics_overall.csv ->" in output


def test_e5_only_benchmark_does_not_probe_or_install_bert_score():
    repo_root = Path(__file__).resolve().parents[2]
    notebook_path = repo_root / "src/nb/benchmark.py"
    notebook = ast.parse(notebook_path.read_text(encoding="utf-8"))
    function_node = next(
        node for node in notebook.body
        if isinstance(node, ast.FunctionDef) and node.name == "ensure_bert_score_available"
    )
    calls = []
    namespace = {
        "importlib": SimpleNamespace(
            util=SimpleNamespace(find_spec=lambda name: calls.append(("find_spec", name))),
            invalidate_caches=lambda: None,
        ),
        "subprocess": SimpleNamespace(run=lambda *args, **kwargs: calls.append((args, kwargs))),
        "sys": SimpleNamespace(executable="python"),
    }
    exec(compile(ast.Module(body=[function_node], type_ignores=[]), str(notebook_path), "exec"), namespace)

    assert namespace["ensure_bert_score_available"](True, ("semsim_e5",)) is False
    assert calls == []


def test_adapter_configs_use_the_attached_kaggle_model_paths():
    repo_root = Path(__file__).resolve().parents[2]
    configs_dir = repo_root / "src/mdv_asr_module/configs/models"
    expected_paths = {
        "qwen_vimd_adalora_small": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/adalora-small/1",
        "qwen_vimd_adalora_full": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/adalora-full/1",
        "qwen_vimd_pissa_small": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/pissa-small/1",
        "qwen_vimd_pissa_full": "/kaggle/input/models/nvtd1796/qwen3-asr-vimd-adapters/pytorch/pissa-full/1",
    }

    loader = ConfigLoader()
    actual_paths = {
        loader.load(configs_dir / f"{key}.json").model.key:
        loader.load(configs_dir / f"{key}.json").model.adapter_path
        for key in expected_paths
    }

    assert actual_paths == expected_paths
