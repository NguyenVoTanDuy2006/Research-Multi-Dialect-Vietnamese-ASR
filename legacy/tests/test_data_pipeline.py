from __future__ import annotations

import io
import wave

import numpy as np

from mdv_asr.config.types import DatasetConfig, SamplingConfig
from mdv_asr.data.audio import AudioReader
from mdv_asr.data.index import DatasetIndex
from mdv_asr.data.manifest import ManifestRepository
from mdv_asr.data.packaging import DatasetPackager
from mdv_asr.data.records import AudioLocator, SampleRecord, SelectionResult
from mdv_asr.data.sources import DatasetSource, DatasetSourceFactory, _resolve_audio_path
from mdv_asr.data.sampling import StratifiedSampler, SpeakerGenderResolver


def make_wav(sample_rate=22050, channels=2, seconds=0.2):
    frames = int(sample_rate * seconds)
    t = np.arange(frames, dtype=np.float32) / sample_rate
    signal = np.sin(2 * np.pi * 440 * t)
    values = np.stack([signal] * channels, axis=1)
    pcm = (values * 32000).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm.tobytes())
    return buffer.getvalue()


class MemorySource(DatasetSource):
    def __init__(self, records, payload):
        super().__init__("unit")
        self.records = records
        self.payload = payload

    def build_index(self):
        return DatasetIndex(self.records, source_fingerprint="unit-fingerprint")

    def read_bytes(self, locator):
        return self.payload

    def fingerprint(self):
        return "unit-fingerprint"


def record(sample_id="s1", *, duration=0.2, split="train", speaker="sp1", gender="female"):
    return SampleRecord("unit", sample_id, split, f"{sample_id}.wav", "xin chào",
                        duration, AudioLocator("file", f"{sample_id}.wav"),
                        province_name="A", speaker_id=speaker, gender=gender)


def test_audio_reader_downmixes_and_resamples():
    payload = make_wav()
    source = MemorySource([], payload)
    audio = AudioReader().read(record(), source)
    assert audio.sample_rate == 16000
    assert audio.waveform.ndim == 1
    assert audio.waveform.dtype == np.float32
    assert abs(audio.duration_seconds - 0.2) < 0.002


def test_package_then_reload_and_read_audio(tmp_path):
    payload = make_wav(16000, 1)
    rows = [record()]
    source = MemorySource(rows, payload)
    selected = SelectionResult(rows, seed=3, requested_seconds={"train": 0.2})
    output = DatasetPackager().package(selected, source, tmp_path / "package")
    package_source = DatasetSourceFactory.create(DatasetConfig(
        kind="package", root=str(output), dataset_id="unit"))
    index = package_source.build_index()
    row = index.for_split("train")[0]
    restored_audio = AudioReader().read(row, package_source)
    assert restored_audio.sample_rate == 16000
    assert abs(restored_audio.duration_seconds - 0.2) < 0.001
    package_source.close()


def test_manifest_round_trip_uses_audio_locator(tmp_path):
    index = DatasetIndex([record()], source_fingerprint="fingerprint")
    path = ManifestRepository().save(index, tmp_path / "manifest.json")
    restored = ManifestRepository().load(path)
    assert restored.get("s1").audio_locator.kind == "file"
    assert restored.source_fingerprint == "fingerprint"


def test_gender_conflicts_become_unknown_and_sampling_is_reproducible():
    rows = [record("a", duration=2, speaker="shared", gender="female"),
            record("b", duration=2, speaker="shared", gender="male"),
            record("c", duration=2, speaker="m1", gender="male"),
            record("d", duration=2, speaker="f1", gender="female")]
    resolved = SpeakerGenderResolver().resolve(rows)
    assert resolved[0].gender == resolved[1].gender == "unknown"
    config = SamplingConfig(seed=9, target_seconds={"train": 4}, balance_train_gender=True)
    sampler = StratifiedSampler()
    first = sampler.sample(DatasetIndex(rows), config)
    second = sampler.sample(DatasetIndex(rows), config)
    assert [row.sample_id for row in first.records] == [row.sample_id for row in second.records]


def test_audio_path_resolution_checks_shard_then_dataset_root(tmp_path):
    shard_dir = tmp_path / "cache" / "shards"
    dataset_root = tmp_path / "dataset"
    payload = dataset_root / "audio" / "clip.wav"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(make_wav(16000, 1))
    shard_dir.mkdir(parents=True)
    assert _resolve_audio_path("audio/clip.wav", (shard_dir, dataset_root)) == payload
