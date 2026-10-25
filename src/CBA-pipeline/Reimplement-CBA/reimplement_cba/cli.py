from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from reimplement_cba.config import CBAConfig
from reimplement_cba.workflow import format_plan, run, validate_config


DEFAULT_CONFIG = Path("configs/vimd_phowhisper_medium.json")


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

    execute = commands.add_parser("run", help="run the workflow scaffold")
    execute.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = CBAConfig.from_file(args.config)
        if args.command == "validate":
            print(json.dumps(validate_config(config, check_data=args.check_data), indent=2,
                             ensure_ascii=False))
        elif args.command == "plan":
            print(format_plan(config))
        elif args.command == "run":
            run(config)
        return 0
    except (FileNotFoundError, KeyError, ValueError, NotImplementedError) as error:
        print(f"reimplement-cba: {error}", file=sys.stderr)
        return 2
