from __future__ import annotations

import gc
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from reimplement_cba.config import CBAConfig
from reimplement_cba.contracts import CheckpointMetrics, SampleScore
from reimplement_cba.stages.adaptive_filter import apply_adaptive_rule
from reimplement_cba.stages.checkpoint_selection import select_best_checkpoint
from reimplement_cba.stages.initial_filter import InitialTranscriptFilter
from reimplement_cba.stages.prediction_io import write_predictions
from reimplement_cba.stages.preprocess import AudioPreprocessor


STAGE_ORDER = (
    "validate ViMD configuration and splits",
    "filter train records by reviewed IDs, transcript markers, and training duration",
    "run unadapted PhoWhisper on eligible train and validation samples",
    "train AdaLoRA and rank epoch/final checkpoints on validation SemScore then WER",
    "score train samples against the previous round and apply the adaptive rule",
    "repeat with the selected adapter and curriculum subset",
    "choose the best round checkpoint on validation and report final test scores",
)


class WorkflowRuntime:
    """Lazy constructors for the model and dataset dependencies used by a run."""

    def open_dataset(self, config: CBAConfig):
        from reimplement_cba.adapters.vimd import ViMDAdapter

        return ViMDAdapter(config)

    def create_metrics(self, config: CBAConfig):
        from reimplement_cba.adapters.metrics import ViMDMetricsAdapter

        return ViMDMetricsAdapter.from_config(config)

    def baseline_stage(self, config: CBAConfig, *, ct2_model_path: Path | None = None):
        from reimplement_cba.stages.initial_inference import InitialInferenceStage

        return InitialInferenceStage(config, ct2_model_path=ct2_model_path)

    def training_stage(self, config: CBAConfig):
        from reimplement_cba.stages.finetune import AdaLoRATrainingStage

        return AdaLoRATrainingStage(config)

    def adapted_stage(
        self,
        config: CBAConfig,
        adapter_path: str | Path,
        *,
        ct2_model_path: Path | None = None,
    ):
        from reimplement_cba.stages.inference import AdaptedInferenceStage

        return AdaptedInferenceStage(
            config, adapter_path, ct2_model_path=ct2_model_path
        )

    def prepare_ct2(
        self, config: CBAConfig, destination: Path, *, adapter_path: str | Path | None = None
    ) -> Path:
        from reimplement_cba.stages.ct2_conversion import prepare_ct2_checkpoint

        return prepare_ct2_checkpoint(config, destination, adapter_path=adapter_path)


def _atomic_write_json(payload: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2, default=str)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, destination)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _release_model_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _split_records(records: list[object]) -> dict[str, list[object]]:
    aliases = {"train": "train", "validation": "validation", "valid": "validation",
               "dev": "validation", "test": "test"}
    result: dict[str, list[object]] = {"train": [], "validation": [], "test": []}
    for record in records:
        split = aliases.get(str(getattr(record, "split", "")).casefold())
        if split is not None:
            result[split].append(record)
    for split, members in result.items():
        if not members:
            raise ValueError(f"ViMD split '{split}' contains no records")
        ids = [str(record.sample_id) for record in members]
        if len(ids) != len(set(ids)):
            raise ValueError(f"ViMD split '{split}' contains duplicate sample IDs")
    return result


def _initial_train_pool(
    config: CBAConfig, records: list[object]
) -> tuple[list[object], list[dict[str, Any]]]:
    initial_rule = config.values["initial_rule"]
    initial_filter = InitialTranscriptFilter(
        rejected_markers=initial_rule.get("reject_transcript_markers", []),
        rejected_sample_ids=config.english_exclusion_sample_ids,
    )
    filtered = initial_filter.apply(records)
    preprocessor = AudioPreprocessor(config)
    selected = preprocessor.select_training_records(filtered.kept)
    rejected = [
        asdict(preprocessor.rejected_record(record, initial_filter.rejection_reason(record)))
        for record in filtered.rejected
    ]
    rejected.extend(asdict(item) for item in selected.rejected)
    if not selected.kept:
        raise ValueError("No training records remain after initial filtering")
    return list(selected.kept), rejected


def _ordered_rows(records: list[object], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in by_id:
            raise ValueError(f"Duplicate prediction for sample {sample_id!r}")
        by_id[sample_id] = row
    expected = {str(record.sample_id) for record in records}
    if set(by_id) != expected:
        missing = sorted(expected - set(by_id))
        unexpected = sorted(set(by_id) - expected)
        raise ValueError(f"Prediction IDs do not match records: missing={missing}, unexpected={unexpected}")
    return [by_id[str(record.sample_id)] for record in records]


def _region(record: object) -> str | None:
    return getattr(record, "province_name", None) or getattr(record, "region", None)


def _evaluate(metrics: object, records: list[object], rows: list[dict[str, Any]]) -> dict:
    ordered = _ordered_rows(records, rows)
    return metrics.evaluate_corpus(
        [str(record.text) for record in records],
        [str(row["hypothesis"]) for row in ordered],
        regions=[_region(record) for record in records],
    )


def _sample_scores(metrics: object, records: list[object], rows: list[dict[str, Any]]) -> dict[str, float]:
    ordered = _ordered_rows(records, rows)
    components = metrics.score_batch(
        [str(record.text) for record in records],
        [str(row["hypothesis"]) for row in ordered],
        regions=[_region(record) for record in records],
    )
    if len(components) != len(records):
        raise ValueError("SemScore returned a different number of samples than the predictions")
    return {
        str(record.sample_id): float(component.semscore)
        for record, component in zip(records, components)
    }


def _adapter_checkpoints(final_adapter: str | Path) -> list[Path]:
    final_path = Path(final_adapter).expanduser().resolve()
    trainer_dir = final_path.parent / "trainer"
    candidates = sorted(trainer_dir.glob("checkpoint-*")) if trainer_dir.is_dir() else []
    candidates.append(final_path)
    result = []
    for candidate in candidates:
        if not candidate.is_dir() or not (candidate / "adapter_config.json").is_file():
            continue
        if not any((candidate / name).is_file() for name in (
            "adapter_model.safetensors", "adapter_model.bin"
        )):
            continue
        result.append(candidate)
    if final_path not in result:
        raise FileNotFoundError(f"Training did not save a usable final adapter: {final_path}")
    return result


def _ct2_artifact(
    config: CBAConfig,
    runtime: WorkflowRuntime,
    *,
    adapter_path: str | Path | None = None,
    round_index: int | None = None,
) -> Path | None:
    if config.values.get("inference", {}).get("backend", "transformers") == "transformers":
        return None
    if adapter_path is None:
        configured = config.values.get("inference", {}).get("ct2_model_path")
        if configured:
            return Path(config.resolve_model_reference(configured))
        destination = config.output_dir / "ct2" / "baseline"
    else:
        if round_index is None:
            raise ValueError("round_index is required for an adapted CT2 checkpoint")
        destination = config.output_dir / "ct2" / f"round-{round_index}" / Path(adapter_path).name
    if destination.exists():
        if not destination.is_dir():
            raise FileExistsError(f"CT2 artifact path is not a directory: {destination}")
        return destination
    return Path(runtime.prepare_ct2(config, destination, adapter_path=adapter_path))


def format_plan(config: CBAConfig) -> str:
    lines = [
        f"Run: {config.run_id}",
        f"Dataset config: {config.dataset_config_path}",
        f"Baseline/base model: {config.values['model']['base_model']}",
        "Inference backend: "
        f"{config.values.get('inference', {}).get('backend', 'transformers')}",
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
    from reimplement_cba.adapters.vimd import ViMDAdapter

    adapter = ViMDAdapter(config)
    try:
        result = {"config": config.summary(), "dataset": adapter.validate_config()}
        if check_data:
            result["data"] = adapter.validate_splits()
        return result
    finally:
        adapter.close()


def run(config: CBAConfig, *, runtime: WorkflowRuntime | None = None) -> dict[str, Any]:
    """Run baseline, AdaLoRA curriculum rounds, checkpoint selection, and final test."""
    config.validate()
    runtime = runtime or WorkflowRuntime()
    output_dir = config.output_dir
    summary_path = output_dir / "run-summary.json"
    if summary_path.exists() or (output_dir / "adapters" / "round-1").exists():
        raise FileExistsError(
            f"A CBA run already exists in {output_dir}; choose a fresh output_dir"
        )

    with runtime.open_dataset(config) as vimd:
        vimd.validate_config()
        vimd.validate_splits()
        splits = _split_records(list(vimd.open().index.records))
        train_pool, initial_rejected = _initial_train_pool(config, splits["train"])
        write_predictions(
            initial_rejected,
            output_dir / "preprocess" / "initial-train-rejected.jsonl",
        )
        validation = splits["validation"]
        test = splits["test"]
        summary: dict[str, Any] = {
            "status": "running", "config": config.summary(),
            "initial_train_count": len(train_pool),
            "initial_train_rejected_count": len(initial_rejected),
            "rounds": [],
        }
        _atomic_write_json(summary, summary_path)

        baseline_ct2 = _ct2_artifact(config, runtime)
        baseline_train_rows = runtime.baseline_stage(
            config, ct2_model_path=baseline_ct2
        ).run(
            train_pool, vimd.read_audio,
            output_path=output_dir / "predictions" / "baseline-train.jsonl",
        )
        baseline_validation_rows = runtime.baseline_stage(
            config, ct2_model_path=baseline_ct2
        ).run(
            validation, vimd.read_audio,
            output_path=output_dir / "predictions" / "baseline-validation.jsonl",
        )
        scorer = runtime.create_metrics(config)
        previous_scores = _sample_scores(scorer, train_pool, baseline_train_rows)
        summary["baseline_validation"] = _evaluate(
            scorer, validation, baseline_validation_rows
        )
        _atomic_write_json(summary, summary_path)
        del scorer
        _release_model_memory()

        selected_rounds: list[CheckpointMetrics] = []
        selected_round_numbers: dict[str, int] = {}
        previous_checkpoint: str | None = None
        for round_index in range(1, int(config.values["curriculum"]["max_rounds"]) + 1):
            resume_adapter = (
                previous_checkpoint
                if round_index > 1 and
                config.values["curriculum"]["round_initialization"] == "resume_previous_adapter"
                else None
            )
            final_adapter = runtime.training_stage(config).run(
                train_pool, validation,
                audio_reader=vimd.read_audio,
                round_index=round_index,
                resume_adapter=resume_adapter,
            )
            candidate_rows: dict[str, list[dict[str, Any]]] = {}
            for checkpoint in _adapter_checkpoints(final_adapter):
                ct2_path = _ct2_artifact(
                    config, runtime, adapter_path=checkpoint, round_index=round_index
                )
                candidate_rows[str(checkpoint)] = runtime.adapted_stage(
                    config, checkpoint, ct2_model_path=ct2_path
                ).run(
                    validation, vimd.read_audio,
                    output_path=output_dir / "predictions" / f"round-{round_index}"
                    / f"{checkpoint.name}-validation.jsonl",
                )

            scorer = runtime.create_metrics(config)
            candidate_metrics: list[CheckpointMetrics] = []
            candidate_reports: dict[str, dict] = {}
            for checkpoint, rows in candidate_rows.items():
                report = _evaluate(scorer, validation, rows)
                candidate_reports[checkpoint] = report
                candidate_metrics.append(CheckpointMetrics(
                    checkpoint=checkpoint,
                    wer=float(report["WER"]),
                    semscore=float(report["semscore"]),
                ))

            best = select_best_checkpoint(candidate_metrics)
            del scorer, candidate_rows
            _release_model_memory()
            previous_checkpoint = best.checkpoint
            selected_rounds.append(best)
            selected_round_numbers[best.checkpoint] = round_index
            selected_ct2 = _ct2_artifact(
                config, runtime, adapter_path=best.checkpoint, round_index=round_index
            )
            train_rows = runtime.adapted_stage(
                config, best.checkpoint, ct2_model_path=selected_ct2
            ).run(
                train_pool, vimd.read_audio,
                output_path=output_dir / "predictions" / f"round-{round_index}"
                / "selected-train.jsonl",
            )
            scorer = runtime.create_metrics(config)
            current_scores = _sample_scores(scorer, train_pool, train_rows)
            sample_scores = [
                SampleScore(str(record.sample_id), previous_scores[str(record.sample_id)],
                            current_scores[str(record.sample_id)])
                for record in train_pool
            ]
            kept, rejected = apply_adaptive_rule(
                sample_scores,
                domain=str(config.values["curriculum"]["domain"]),
                threshold=float(config.values["curriculum"]["threshold"]),
            )
            decisions_by_id = {item.sample_id: item for item in kept + rejected}
            write_predictions(
                [asdict(decisions_by_id[str(record.sample_id)]) for record in train_pool],
                output_dir / "curriculum" / f"round-{round_index}-decisions.jsonl",
            )
            summary["rounds"].append({
                "round_index": round_index,
                "train_count": len(train_pool),
                "resume_adapter": resume_adapter,
                "candidate_metrics": [asdict(item) for item in candidate_metrics],
                "selected_checkpoint": best.checkpoint,
                "selected_validation": candidate_reports[best.checkpoint],
                "curriculum": {
                    "kept_count": len(kept), "rejected_count": len(rejected),
                    "decisions_path": str(
                        output_dir / "curriculum" / f"round-{round_index}-decisions.jsonl"
                    ),
                },
            })
            _atomic_write_json(summary, summary_path)
            del scorer
            _release_model_memory()
            if not kept and round_index < int(config.values["curriculum"]["max_rounds"]):
                summary["stop_reason"] = f"no_samples_after_round_{round_index}"
                break
            keep_ids = {item.sample_id for item in kept}
            train_pool = [record for record in train_pool if str(record.sample_id) in keep_ids]
            previous_scores = current_scores

        final_best = select_best_checkpoint(selected_rounds)
        final_round = selected_round_numbers[final_best.checkpoint]
        baseline_test_rows = runtime.baseline_stage(
            config, ct2_model_path=baseline_ct2
        ).run(
            test, vimd.read_audio,
            output_path=output_dir / "predictions" / "baseline-test.jsonl",
        )
        final_ct2 = _ct2_artifact(
            config, runtime, adapter_path=final_best.checkpoint, round_index=final_round
        )
        final_test_rows = runtime.adapted_stage(
            config, final_best.checkpoint, ct2_model_path=final_ct2
        ).run(
            test, vimd.read_audio,
            output_path=output_dir / "predictions" / "final-test.jsonl",
        )
        scorer = runtime.create_metrics(config)
        summary["final"] = {
            "checkpoint": final_best.checkpoint,
            "validation": next(
                item["selected_validation"] for item in summary["rounds"]
                if item["selected_checkpoint"] == final_best.checkpoint
            ),
            "test": _evaluate(scorer, test, final_test_rows),
            "baseline_test": _evaluate(scorer, test, baseline_test_rows),
        }
        del scorer
        _release_model_memory()
        summary["status"] = "complete"
        _atomic_write_json(summary, summary_path)
        return summary
