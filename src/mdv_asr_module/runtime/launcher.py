from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .environment import RuntimeEnvironment


class WorkerLauncher:
    def __init__(self, module: str = "mdv_asr_module.training.worker",
                 environment: RuntimeEnvironment | None = None,
                 torchrun: str = "torchrun"):
        self.module = module
        self.environment = environment or RuntimeEnvironment()
        self.torchrun = torchrun

    def build_command(self, task: str, mode: str, *, config_path: str | Path,
                      manifest_path: str | Path | None = None,
                      nproc_per_node: int = 1, extra_args: Sequence[str] = ()) -> list[str]:
        if nproc_per_node < 1:
            raise ValueError("nproc_per_node must be positive")
        command = [self.torchrun, "--standalone", f"--nproc_per_node={nproc_per_node}",
                   "--module", self.module, "--task", task, "--mode", mode,
                   "--config", str(config_path)]
        if manifest_path is not None:
            command.extend(["--manifest", str(manifest_path)])
        command.extend(extra_args)
        return command

    def launch(self, task: str, mode: str, *, config_path: str | Path,
               manifest_path: str | Path | None = None, nproc_per_node: int = 1,
               extra_args: Sequence[str] = (), check: bool = True) -> subprocess.CompletedProcess:
        command = self.build_command(task, mode, config_path=config_path,
                                     manifest_path=manifest_path,
                                     nproc_per_node=nproc_per_node, extra_args=extra_args)
        result = subprocess.run(command, env=self.environment.subprocess_env(), text=True)
        if check and result.returncode:
            raise RuntimeError(f"Worker exited with status {result.returncode}: {command}")
        return result

    def smoke(self, **kwargs):
        return self.launch(kwargs.pop("task", "asr"), "smoke", **kwargs)

    def train(self, **kwargs):
        return self.launch(kwargs.pop("task", "asr"), "train", **kwargs)

    def evaluate(self, **kwargs):
        return self.launch(kwargs.pop("task", "asr"), "evaluate", **kwargs)

