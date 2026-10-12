from __future__ import annotations

from pathlib import Path

from mdv_asr_module.configs.types import DatasetConfig
from mdv_asr_module.data.audio import AudioPreview, AudioReader
from mdv_asr_module.data.manifest import ManifestRepository
from mdv_asr_module.data.sources import DatasetSourceFactory


class DatasetWorkflow:
    def __init__(self, config: DatasetConfig, source=None, index=None):
        config.validate()
        self.config = config
        self.source = source or DatasetSourceFactory.create(config)
        self.index = index
        self.reader = AudioReader()
        self.preview = AudioPreview(self.reader)
        self.repository = ManifestRepository()

    @classmethod
    def from_config(cls, config: DatasetConfig):
        return cls(config)

    def _cache_path(self) -> Path | None:
        if not self.config.cache_dir:
            return None
        return Path(self.config.cache_dir) / f"{self.config.dataset_id.replace('/', '_')}_manifest.json"

    def scan(self, use_cache: bool = True, force: bool = False):
        cache = self._cache_path()
        fingerprint = self.source.fingerprint()
        if cache and use_cache and not force and self.repository.is_reusable(cache, fingerprint):
            self.index = self.repository.load(cache)
        else:
            self.index = self.source.build_index()
            if cache:
                self.repository.save(self.index, cache)
        return self.index

    def find_by_filename(self, filename: str, split: str | None = None):
        if self.index is None:
            self.scan()
        return self.index.find_by_filename(filename, split=split)

    def play_by_filename(self, filename: str, split: str | None = None,
                         autoplay: bool = False):
        if self.index is None:
            self.scan()
        return self.preview.play_by_filename(self.index, self.source, filename, split, autoplay)

    def close(self):
        self.source.close()

