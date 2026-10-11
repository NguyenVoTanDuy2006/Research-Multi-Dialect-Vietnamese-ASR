from __future__ import annotations

import hashlib
import io
from abc import ABC, abstractmethod
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from mdv_asr_module.config.types import DatasetConfig
from mdv_asr_module.core.errors import DatasetIntegrityError

from .index import DatasetIndex
from .records import AudioLocator, SampleRecord


class DatasetSource(ABC):
    """Common lazy-read contract for Arrow and Parquet datasets."""

    def __init__(self, dataset_id: str):
        self.dataset_id = dataset_id
        self._index: DatasetIndex | None = None

    @abstractmethod
    def build_index(self) -> DatasetIndex:
        raise NotImplementedError

    @abstractmethod
    def read_bytes(self, locator: AudioLocator) -> bytes:
        raise NotImplementedError

    @abstractmethod
    def fingerprint(self) -> str:
        raise NotImplementedError

    def close(self) -> None:
        self._index = None

    @staticmethod
    def _split_for(path: Path) -> str | None:
        parts = [part.casefold() for part in path.parts]
        name = path.name.casefold()
        for value in ("train", "valid", "validation", "test"):
            if value in parts or name.startswith(value + "-") or f"-{value}-" in name:
                return "valid" if value == "validation" else value
        return None

    @staticmethod
    def _audio_bytes(audio_info: dict[str, Any], *relative_roots: Path) -> bytes:
        payload = audio_info.get("bytes")
        if payload is not None:
            return bytes(payload)
        raw_path = audio_info.get("path")
        if not raw_path:
            raise DatasetIntegrityError("Audio metadata has neither bytes nor path")
        path = _resolve_audio_path(raw_path, relative_roots)
        if not path.is_file():
            raise FileNotFoundError(f"Audio payload does not exist: {path}")
        return path.read_bytes()

    @staticmethod
    def _audio_metadata(audio_info: dict[str, Any], *relative_roots: Path) -> tuple[float, str | None]:
        import soundfile as sf

        payload = audio_info.get("bytes")
        if payload is not None:
            info = sf.info(io.BytesIO(payload))
            audio_path = None
        else:
            raw_path = audio_info.get("path")
            if not raw_path:
                raise DatasetIntegrityError("Audio metadata has neither bytes nor path")
            path = _resolve_audio_path(raw_path, relative_roots)
            if not path.is_file():
                raise FileNotFoundError(f"Audio payload does not exist: {path}")
            info = sf.info(str(path))
            audio_path = str(path.resolve())
        if info.frames <= 0 or info.samplerate <= 0:
            raise DatasetIntegrityError("Audio has invalid frames or sample rate")
        return float(info.duration), audio_path

    @classmethod
    def _record(cls, row: dict[str, Any], *, dataset_id: str, split: str,
                sample_id: str, locator: AudioLocator, duration: float,
                fallback_name: str) -> SampleRecord:
        audio_path = row.get("filename") or row.get("audio_path") or fallback_name
        filename = PurePosixPath(str(audio_path).replace("\\", "/")).name
        speaker = row.get("speakerID", row.get("speaker_id"))
        raw_gender = row.get("gender", row.get("resolved_speaker_gender"))
        gender_map = {"0": "female", "female": "female", "f": "female", "nữ": "female", "nu": "female",
                      "1": "male", "male": "male", "m": "male", "nam": "male"}
        gender = gender_map.get(str(raw_gender).strip().casefold(), "unknown")
        record = SampleRecord(
            dataset_id=str(row.get("dataset_id") or dataset_id),
            sample_id=str(row.get("sample_id") or sample_id),
            source_sample_id=(str(row["source_sample_id"]) if row.get("source_sample_id") else None),
            split=str(row.get("split") or split), filename=filename,
            text=str(row.get("text") or ""), duration_seconds=float(row.get("duration_seconds") or duration),
            audio_locator=locator, region=_optional_string(row.get("region")),
            province_name=_optional_string(row.get("province_name")),
            province_code=row.get("province_code"),
            speaker_id=_optional_string(speaker), gender=gender,
            source_gender=_optional_string(raw_gender),
            metadata={key: value for key, value in row.items()
                      if key not in {"audio", "bytes", "text", "filename", "audio_path"}},
        )
        record.validate()
        return record


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in {"", "nan", "none", "<na>"} else text


def _resolve_audio_path(raw_path: Any, relative_roots: Iterable[Path]) -> Path:
    path = Path(str(raw_path)).expanduser()
    if path.is_absolute():
        return path
    candidates = [root / path for root in relative_roots]
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


class _ShardDatasetSource(DatasetSource):
    def __init__(self, root: str | Path, dataset_id: str):
        super().__init__(dataset_id)
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self._datasets: dict[str, Any] = {}
        self._paths = self._find_files()

    @abstractmethod
    def _find_files(self) -> dict[str, list[Path]]:
        raise NotImplementedError

    @abstractmethod
    def _load_shard(self, path: Path):
        raise NotImplementedError

    def _get_shard(self, path: Path):
        key = str(path)
        if key not in self._datasets:
            self._datasets[key] = self._load_shard(path)
        return self._datasets[key]

    def build_index(self) -> DatasetIndex:
        if self._index is not None:
            return self._index
        records: list[SampleRecord] = []
        for split, paths in self._paths.items():
            split_offset = 0
            for path in paths:
                dataset = self._get_shard(path)
                for local_index in range(len(dataset)):
                    row = dataset[local_index]
                    audio_info = row.get("audio")
                    if not isinstance(audio_info, dict):
                        raise DatasetIntegrityError(f"Missing audio object in {path}:{local_index}")
                    duration, resolved_path = self._audio_metadata(audio_info, path.parent, self.root)
                    locator = AudioLocator(self.locator_kind, str(path), row_index=local_index)
                    if resolved_path:
                        row = dict(row)
                        row["audio_path"] = resolved_path
                    records.append(self._record(
                        row, dataset_id=self.dataset_id, split=split,
                        sample_id=f"{split}_{split_offset + local_index:07d}",
                        locator=locator, duration=duration,
                        fallback_name=f"{split}_{split_offset + local_index:07d}.wav",
                    ))
                split_offset += len(dataset)
        if not records:
            raise DatasetIntegrityError(f"No dataset rows found under {self.root}")
        self._index = DatasetIndex(records, source_fingerprint=self.fingerprint())
        return self._index

    @property
    @abstractmethod
    def locator_kind(self) -> str:
        raise NotImplementedError

    def read_bytes(self, locator: AudioLocator) -> bytes:
        locator.validate()
        dataset = self._get_shard(Path(locator.path))
        row = dataset[int(locator.row_index)]
        return self._audio_bytes(row["audio"], Path(locator.path).parent, self.root)

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for split, paths in sorted(self._paths.items()):
            digest.update(split.encode())
            for path in paths:
                stat = path.stat()
                digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return digest.hexdigest()

    def close(self) -> None:
        self._datasets.clear()
        super().close()


class ArrowDatasetSource(_ShardDatasetSource):
    @property
    def locator_kind(self) -> str:
        return "arrow"

    def _find_files(self) -> dict[str, list[Path]]:
        files = sorted(self.root.rglob("*.arrow"))
        return _group_split_files(files)

    def _load_shard(self, path: Path):
        from datasets import Audio, Dataset

        return Dataset.from_file(str(path)).cast_column("audio", Audio(decode=False))


class ParquetDatasetSource(_ShardDatasetSource):
    @property
    def locator_kind(self) -> str:
        return "parquet"

    def _find_files(self) -> dict[str, list[Path]]:
        files = sorted(self.root.rglob("*.parquet"))
        return _group_split_files(files)

    def _load_shard(self, path: Path):
        from datasets import Audio, load_dataset

        dataset = load_dataset("parquet", data_files={"train": [str(path)]}, split="train")
        return dataset.cast_column("audio", Audio(decode=False))


def _group_split_files(files: Iterable[Path]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {"train": [], "valid": [], "test": []}
    for path in files:
        split = DatasetSource._split_for(path)
        if split in result:
            result[split].append(path.resolve())
    result = {split: paths for split, paths in result.items() if paths}
    if not result:
        raise FileNotFoundError("No recognized train/valid/test Arrow or Parquet shards found")
    return result


class DatasetSourceFactory:
    @classmethod
    def create(cls, config: DatasetConfig) -> DatasetSource:
        config.validate()
        root = cls.resolve_root(config)
        kind = config.kind
        if kind == "auto":
            if list(root.rglob("*.arrow")):
                kind = "arrow"
            elif list(root.rglob("*.parquet")):
                kind = "parquet"
            else:
                raise FileNotFoundError(
                    f"No Arrow or Parquet shards found under configured dataset root: {root}"
                )
        source_type = {
            "arrow": ArrowDatasetSource,
            "parquet": ParquetDatasetSource,
        }[kind]
        return source_type(root, config.dataset_id)

    @staticmethod
    def resolve_root(config: DatasetConfig) -> Path:
        if config.root:
            root = Path(config.root).expanduser().resolve()
            if not root.is_dir():
                raise FileNotFoundError(f"Configured dataset root does not exist: {root}")
            return root
        candidates = [
            Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0"),
            Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset"),
        ]
        # Prefer the versioned KaggleHub directory. Its parent also exists and
        # recursively sees the same shards, so treating both as ambiguous breaks
        # the common Kaggle mount layout.
        for candidate in candidates:
            if candidate.is_dir() and (
                next(candidate.rglob("*.arrow"), None) is not None
                or next(candidate.rglob("*.parquet"), None) is not None
            ):
                return candidate.resolve()
        if config.allow_download and config.handle:
            import kagglehub

            return Path(kagglehub.dataset_download(config.handle)).resolve()
        raise FileNotFoundError("Set dataset root, attach it to Kaggle, or explicitly enable download")

