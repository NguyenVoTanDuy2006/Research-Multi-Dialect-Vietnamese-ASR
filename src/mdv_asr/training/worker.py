from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from mdv_asr.config.loader import ConfigLoader
from mdv_asr.core.artifacts import hash_payload
from mdv_asr.data.manifest import ManifestRepository
from mdv_asr.data.sources import DatasetSourceFactory
from mdv_asr.runtime.distributed import DistributedContext


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ViMD ASR worker")
    parser.add_argument("--task", choices=("asr", "benchmark"), default="asr")
    parser.add_argument("--mode", choices=("smoke", "train", "evaluate"), required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--resume-from")
    return parser.parse_args(argv)


def _load_index(args, config, source):
    if not args.manifest:
        return source.build_index()
    repository = ManifestRepository()
    return repository.import_legacy(args.manifest, config.dataset)


def _hf_token():
    from mdv_asr.runtime.environment import RuntimeEnvironment

    return RuntimeEnvironment.get_hf_token()


def main(argv=None):
    args = parse_args(argv)
    config = ConfigLoader().load(args.config)
    source = DatasetSourceFactory.create(config.dataset)
    context = DistributedContext.from_environment()
    try:
        context.initialize()
        index = _load_index(args, config, source)
        if args.task == "asr" and args.mode in {"smoke", "train"}:
            if config.model is None or config.train is None:
                raise ValueError("ASR training requires model and train config sections")
            from .ada_lora import QwenAdaLoraTrainer

            trainer = QwenAdaLoraTrainer(
                records_by_split={name: list(index.for_split(name)) for name in ("train", "valid", "test")},
                dataset_config=config.dataset, model_spec=config.model,
                train_config=config.train, decode_config=config.decode,
                output_dir=config.output_dir, context=context,
                token=_hf_token(),
            )
            try:
                report = trainer.smoke() if args.mode == "smoke" else trainer.fit(args.resume_from)
                if context.is_main:
                    print(json.dumps(report, ensure_ascii=False, default=str), flush=True)
            finally:
                trainer.close()
            return 0
        if args.mode == "evaluate":
            benchmark_config = config.benchmark
            if benchmark_config is None and args.task == "asr" and config.model is not None:
                best_path = Path(config.output_dir) / "best_checkpoint.json"
                if not best_path.is_file():
                    raise FileNotFoundError("Training evaluation needs best_checkpoint.json")
                best = json.loads(best_path.read_text(encoding="utf-8"))
                selected_model = replace(config.model, adapter_path=best["adapter_path"])
                from mdv_asr.config.types import BenchmarkConfig

                benchmark_config = BenchmarkConfig(models=(selected_model,), split="test")
            if benchmark_config is None:
                raise ValueError("Evaluation requires a benchmark section or trained ASR model")
            from mdv_asr.evaluation.benchmark import BenchmarkRunner
            from mdv_asr.evaluation.evaluator import ASREvaluator
            from mdv_asr.evaluation.reports import BenchmarkReporter

            records = list(index.for_split(benchmark_config.split))
            runner = BenchmarkRunner(benchmark_config, source, ASREvaluator(context=context),
                                     decode=config.decode)
            results = runner.run(records)
            if context.is_main:
                reporter = BenchmarkReporter(config.output_dir)
                reporter.write_all(results, baseline_key=benchmark_config.models[0].key)
            runner.validate_comparison()
            return 0
        raise ValueError(f"Unsupported task/mode combination: {args.task}/{args.mode}")
    finally:
        source.close()
        context.close()


if __name__ == "__main__":
    raise SystemExit(main())

