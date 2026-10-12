from __future__ import annotations

import json

from mdv_asr_module.configs.loader import ConfigLoader


def test_load_benchmark_configs_combines_one_dataset_and_multiple_models(tmp_path):
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps({
        "platform": "local",
        "dataset": {
            "kind": "arrow",
            "root": str(tmp_path / "dataset"),
            "dataset_id": "vimd_test",
            "cache_dir": str(tmp_path / "cache"),
            "allow_download": False,
        },
        "output_dir": str(tmp_path / "old-output"),
        "run_id": "dataset-config",
    }), encoding="utf-8")
    model_paths = []
    for key in ("model_a", "model_b"):
        path = tmp_path / f"{key}.json"
        path.write_text(json.dumps({
            "platform": "kaggle",
            "model": {
                "key": key,
                "backend": "qwen3_asr",
                "model_id": "Qwen/test-model",
                "adapter_path": f"/models/{key}",
            },
        }), encoding="utf-8")
        model_paths.append(path)

    output_dir = tmp_path / "benchmark-output"
    config = ConfigLoader().load_benchmark_configs(
        dataset_path,
        model_paths,
        output_dir=output_dir,
        split="test",
    )

    assert config.dataset.dataset_id == "vimd_test"
    assert config.platform == "local"
    assert config.decode.language == "Vietnamese"
    assert config.decode.max_new_tokens == 768
    assert [model.key for model in config.benchmark.models] == ["model_a", "model_b"]
    assert config.benchmark.split == "test"
    assert config.output_dir == str(output_dir)
    assert config.run_id == "vimd_test_benchmark"
