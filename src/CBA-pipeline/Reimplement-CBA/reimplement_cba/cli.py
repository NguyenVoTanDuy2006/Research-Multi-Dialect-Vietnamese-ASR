from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reimplement_cba.config import CBAConfig


DEFAULT_CONFIG = Path("configs/vimd_phowhisper_small.json")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="reimplement-cba")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate run and ViMD configs")
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))
    validate.add_argument(
        "--check-data", action="store_true",
        help="open the configured source and check train/validation/test splits",
    )

    plan = commands.add_parser("plan", help="print the configured CBA stage order")
    plan.add_argument("--config", default=str(DEFAULT_CONFIG))

    execute = commands.add_parser("run", help="run the full CBA curriculum workflow")
    execute.add_argument("--config", default=str(DEFAULT_CONFIG))

    convert = commands.add_parser(
        "prepare-ct2", help="merge an optional adapter and convert a Whisper checkpoint"
    )
    convert.add_argument("--config", default=str(DEFAULT_CONFIG))
    convert.add_argument("--adapter-path", help="optional PEFT/AdaLoRA adapter directory")
    convert.add_argument("--output", required=True, help="new CTranslate2 output directory")

    infer = commands.add_parser("infer", help="run baseline or adapted inference on a ViMD split")
    infer.add_argument("--config", default=str(DEFAULT_CONFIG))
    infer.add_argument("--split", required=True, choices=("train", "validation", "test"))
    infer.add_argument("--adapter-path", help="optional fine-tuned PEFT adapter directory")
    infer.add_argument("--ct2-model-path", help="prepared CT2 artifact for a CT2 backend")
    infer.add_argument("--output", help="optional prediction JSONL destination")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = CBAConfig.from_file(args.config)
        if args.command == "validate":
            from reimplement_cba.workflow import validate_config

            print(json.dumps(validate_config(config, check_data=args.check_data), indent=2,
                             ensure_ascii=False))
        elif args.command == "plan":
            from reimplement_cba.workflow import format_plan

            print(format_plan(config))
        elif args.command == "run":
            from reimplement_cba.workflow import run

            summary = run(config)
            print(json.dumps({
                "run_id": config.run_id,
                "status": summary["status"],
                "rounds_completed": len(summary["rounds"]),
                "summary_path": str(config.output_dir / "run-summary.json"),
                "final": summary["final"],
            }, indent=2, ensure_ascii=False))
        elif args.command == "prepare-ct2":
            from reimplement_cba.stages.ct2_conversion import prepare_ct2_checkpoint

            output_path = Path(args.output).expanduser()
            if not output_path.is_absolute():
                output_path = config.path.parent / output_path
            result = prepare_ct2_checkpoint(
                config,
                output_path,
                adapter_path=args.adapter_path,
            )
            print(json.dumps({"ct2_model_path": str(result)}, indent=2))
        elif args.command == "infer":
            from reimplement_cba.adapters.vimd import ViMDAdapter
            from reimplement_cba.stages.inference import AdaptedInferenceStage
            from reimplement_cba.stages.initial_inference import InitialInferenceStage

            adapter_path = args.adapter_path
            ct2_model_path = args.ct2_model_path
            if adapter_path:
                adapter_path = config.resolve_model_reference(adapter_path)
            if ct2_model_path:
                ct2_model_path = config.resolve_model_reference(ct2_model_path)
            accepted_splits = {
                "train": {"train"},
                "validation": {"validation", "valid", "dev"},
                "test": {"test"},
            }[args.split]
            with ViMDAdapter(config) as vimd:
                records = [
                    record for record in vimd.open().index.records
                    if record.split.casefold() in accepted_splits
                ]
                if not records:
                    raise ValueError(f"ViMD split '{args.split}' contains no records")
                stage = (
                    AdaptedInferenceStage(
                        config,
                        adapter_path,
                        ct2_model_path=ct2_model_path,
                    )
                    if adapter_path
                    else InitialInferenceStage(config, ct2_model_path=ct2_model_path)
                )
                rows = stage.run(records, vimd.read_audio, output_path=args.output)
            print(json.dumps({"split": args.split, "prediction_count": len(rows)}, indent=2))
        return 0
    except (FileNotFoundError, FileExistsError, KeyError, ValueError,
            NotImplementedError, RuntimeError) as error:
        print(f"reimplement-cba: {error}", file=sys.stderr)
        return 2
