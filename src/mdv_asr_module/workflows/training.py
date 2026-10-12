from __future__ import annotations

from pathlib import Path

from mdv_asr_module.runtime.launcher import WorkerLauncher


class ASRTrainingWorkflow:
    def __init__(self, config, manifest_path: str | Path,
                 launcher: WorkerLauncher | None = None):
        self.config = config
        self.manifest_path = Path(manifest_path)
        self.launcher = launcher or WorkerLauncher()

    def prepare(self):
        self.config.validate()
        if self.config.model is None or self.config.train is None:
            raise ValueError("Training config requires model and train sections")
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        return {"config": self.config.to_dict(), "manifest": str(self.manifest_path)}

    def smoke(self):
        self.prepare()
        return self.launcher.smoke(
            task="asr", config_path=self._write_config(), manifest_path=self.manifest_path,
            nproc_per_node=self.config.train.world_size,
        )

    def train(self, resume_from: str | Path | None = None):
        self.prepare()
        args = ["--resume-from", str(resume_from)] if resume_from else ()
        return self.launcher.train(
            task="asr", config_path=self._write_config(), manifest_path=self.manifest_path,
            nproc_per_node=self.config.train.world_size, extra_args=args,
        )

    def evaluate_best(self):
        self.config.validate()
        return self.launcher.evaluate(
            task="asr", config_path=self._write_config(), manifest_path=self.manifest_path,
            nproc_per_node=self.config.train.world_size if self.config.train else 1,
        )

    def history(self):
        import pandas as pd

        path = Path(self.config.output_dir) / "train_history.csv"
        if not path.is_file():
            raise FileNotFoundError(path)
        return pd.read_csv(path)

    def best_adapter(self):
        from mdv_asr_module.training.checkpoints import CheckpointManager

        return CheckpointManager(self.config.output_dir).get_best_adapter()

    def _write_config(self):
        from mdv_asr_module.configs.loader import ConfigLoader

        path = Path(self.config.output_dir) / "resolved_config.json"
        return ConfigLoader().save_resolved(self.config, path)

