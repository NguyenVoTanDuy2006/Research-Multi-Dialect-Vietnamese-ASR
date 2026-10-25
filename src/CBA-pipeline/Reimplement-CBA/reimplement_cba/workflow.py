from __future__ import annotations

from reimplement_cba.adapters.vimd import ViMDAdapter
from reimplement_cba.config import CBAConfig


STAGE_ORDER = (
    "validate ViMD configuration and splits",
    "preprocess labelled audio",
    "initial inference with unadapted PhoWhisper",
    "apply initial transcript rule",
    "fine-tune AdaLoRA and select checkpoint on validation WER/SemScore",
    "score samples and apply adaptive curriculum rule for each round",
    "resume the previous adapter for the next round (configured rounds)",
    "merge/export and run final inference",
    "postprocess and report held-out WER/SemScore",
)


def format_plan(config: CBAConfig) -> str:
    lines = [
        f"Run: {config.run_id}",
        f"Dataset config: {config.dataset_config_path}",
        f"Baseline/base model: {config.values['model']['base_model']}",
        f"Curriculum: {config.values['curriculum']['max_rounds']} rounds, "
        f"domain={config.values['curriculum']['domain']}, "
        f"threshold={config.values['curriculum']['threshold']}",
        f"Checkpoint priority: {config.values['checkpoint_selection']['primary_metric']} "
        f"then {config.values['checkpoint_selection']['tie_break_metric']}",
        "Stages:",
    ]
    lines.extend(f"  {number}. {stage}" for number, stage in enumerate(STAGE_ORDER, start=1))
    return "\n".join(lines)


def validate_config(config: CBAConfig, *, check_data: bool = False) -> dict:
    adapter = ViMDAdapter(config)
    try:
        result = {"config": config.summary(), "dataset": adapter.validate_config()}
        if check_data:
            result["data"] = adapter.validate_splits()
        return result
    finally:
        adapter.close()


def run(config: CBAConfig) -> None:
    validate_config(config)
    raise NotImplementedError(
        "The scaffold is ready, but model loading, SemScore, and training stages are TODO. "
        "Use the 'plan' command to inspect the configured workflow."
    )
