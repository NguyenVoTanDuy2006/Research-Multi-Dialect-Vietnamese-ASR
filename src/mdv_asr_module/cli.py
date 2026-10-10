from __future__ import annotations

import argparse
import json
from pathlib import Path

from mdv_asr_module.config.loader import ConfigLoader


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mdv-asr", description="Vietnamese ASR fine-tuning and benchmarking")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="Validate or resolve an experiment config")
    config.add_argument("action", choices=("validate", "resolve"))
    config.add_argument("--config", required=True)
    config.add_argument("--output")

    train = commands.add_parser("train", help="Launch ASR training worker")
    train.add_argument("action", choices=("smoke", "run", "evaluate"))
    train.add_argument("--config", required=True)
    train.add_argument("--manifest")
    train.add_argument("--resume-from")
    train.add_argument("--workers", type=int)

    benchmark = commands.add_parser("benchmark", help="Run paired ASR benchmark")
    benchmark.add_argument("action", choices=("run",))
    benchmark.add_argument("--config", required=True)

    audit = commands.add_parser("audit", help="Inspect and optionally load-check an AdaLoRA adapter")
    audit.add_argument("--adapter", required=True)
    audit.add_argument("--config", help="Config with the base Qwen model for loaded tensor checks")
    audit.add_argument("--forward", action="store_true")

    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "config":
        config = ConfigLoader().load(args.config)
        if args.action == "resolve":
            destination = args.output or str(Path(args.config).with_suffix(".resolved.json"))
            ConfigLoader().save_resolved(config, destination)
            print(destination)
        else:
            print(json.dumps({"valid": True, "run_id": config.run_id}, ensure_ascii=False))
        return 0
    if args.command == "train":
        from mdv_asr_module.runtime.launcher import WorkerLauncher

        config = ConfigLoader().load(args.config)
        config.validate()
        if config.train is None:
            raise ValueError("Config is missing train section")
        config_path = ConfigLoader().save_resolved(config, Path(config.output_dir) / "resolved_config.json")
        launcher = WorkerLauncher()
        extra = []
        if args.resume_from:
            extra.extend(["--resume-from", args.resume_from])
        launch = {"task": "asr", "config_path": config_path,
                  "manifest_path": args.manifest,
                  "nproc_per_node": args.workers or config.train.world_size,
                  "extra_args": extra}
        method = {"smoke": launcher.smoke, "run": launcher.train,
                  "evaluate": launcher.evaluate}[args.action]
        method(**launch)
        return 0
    if args.command == "benchmark":
        config = ConfigLoader().load(args.config)
        from mdv_asr_module.workflows.dataset import DatasetWorkflow
        workflow = DatasetWorkflow(config.dataset)
        try:
            workflow.scan()
            from mdv_asr_module.workflows.benchmark import ASRBenchmarkWorkflow

            results = ASRBenchmarkWorkflow(config, workflow).run_test()
            print(json.dumps({"models": len(results), "output_dir": config.output_dir}, ensure_ascii=False))
        finally:
            workflow.close()
        return 0
    if args.command == "audit":
        return _audit(args)
    return 2


def _audit(args) -> int:
    from mdv_asr_module.models.audit import AdapterAuditor

    auditor = AdapterAuditor()
    report = auditor.audit(args.adapter)
    if args.config:
        config = ConfigLoader().load(args.config)
        if config.model is None:
            raise ValueError("Config needs a model section for load-time adapter comparison")
        from mdv_asr_module.models.qwen import QwenASRBackend, QwenModelLoader
        from mdv_asr_module.runtime.environment import RuntimeEnvironment

        loader = QwenModelLoader(RuntimeEnvironment.get_hf_token())
        bundle = loader.load_base(config.model)
        try:
            loader.load_adapter(bundle, args.adapter)
            report["loaded_state"] = auditor.compare_loaded_state(args.adapter, bundle.model)
            if args.forward:
                report["forward_check"] = "requires an explicit audio batch; load-time comparison passed"
        finally:
            loader.release(bundle)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
