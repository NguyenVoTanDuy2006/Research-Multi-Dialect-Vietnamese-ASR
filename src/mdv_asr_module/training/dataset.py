from __future__ import annotations

import os
from typing import Iterable

from torch.utils.data import DataLoader, Dataset, DistributedSampler

from mdv_asr_module.configs.types import DatasetConfig, TrainConfig
from mdv_asr_module.data.audio import AudioReader
from mdv_asr_module.data.records import SampleRecord
from mdv_asr_module.data.sources import DatasetSourceFactory


class ASRAudioDataset(Dataset):
    def __init__(self, records: Iterable[SampleRecord], source_config: DatasetConfig,
                 reader: AudioReader | None = None):
        self.records = tuple(records)
        self.source_config = source_config
        self.reader = reader or AudioReader()
        self._source = None
        self._source_pid = None

    def __len__(self) -> int:
        return len(self.records)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_source"] = None
        state["_source_pid"] = None
        return state

    def _get_source(self):
        pid = os.getpid()
        if self._source is None or self._source_pid != pid:
            self._source = DatasetSourceFactory.create(self.source_config)
            self._source_pid = pid
        return self._source

    def __getitem__(self, index):
        record = self.records[index]
        audio = self.reader.read(record, self._get_source())
        return {"record": record, "audio": audio}

    def close(self):
        if self._source is not None:
            self._source.close()
            self._source = None


class ASRDataLoaderFactory:
    def __init__(self, config: TrainConfig, context=None):
        self.config = config
        self.context = context

    def make_sampler(self, dataset, shuffle: bool):
        if self.context is not None and self.context.is_distributed:
            return DistributedSampler(dataset, num_replicas=self.context.world_size,
                                     rank=self.context.rank, shuffle=shuffle,
                                     seed=self.config.seed, drop_last=False)
        return None

    def worker_init_fn(self, worker_id: int) -> None:
        import random
        import numpy as np
        import torch

        seed = torch.initial_seed() % (2**32)
        random.seed(seed)
        np.random.seed(seed)

    def create(self, records, source_config: DatasetConfig, split: str,
               collate_fn, *, shuffle: bool = False):
        dataset = ASRAudioDataset(records, source_config)
        sampler = self.make_sampler(dataset, shuffle)
        return DataLoader(
            dataset, batch_size=self.config.per_device_batch_size,
            sampler=sampler, shuffle=shuffle and sampler is None,
            num_workers=self.config.num_workers, pin_memory=False,
            collate_fn=collate_fn, worker_init_fn=self.worker_init_fn,
            persistent_workers=self.config.num_workers > 0,
        ), sampler

    @staticmethod
    def steps_per_epoch(loader, accumulation_steps: int) -> int:
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be positive")
        return (len(loader) + accumulation_steps - 1) // accumulation_steps

