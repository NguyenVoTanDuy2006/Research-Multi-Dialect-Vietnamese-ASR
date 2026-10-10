from __future__ import annotations

import io
from pathlib import Path
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .records import AudioSample, SampleRecord
from .sources import DatasetSource


class AudioReader:
    def __init__(self, target_sample_rate: int = 16000):
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        self.target_sample_rate = int(target_sample_rate)

    def inspect(self, record: SampleRecord, source: DatasetSource) -> dict[str, float | int]:
        payload = source.read_bytes(record.audio_locator)
        info = sf.info(io.BytesIO(payload))
        return {"duration_seconds": float(info.duration), "sample_rate": int(info.samplerate),
                "channels": int(info.channels), "frames": int(info.frames)}

    def read(self, record: SampleRecord, source: DatasetSource) -> AudioSample:
        payload = source.read_bytes(record.audio_locator)
        waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
        waveform = waveform.mean(axis=1, dtype=np.float32) if waveform.shape[1] > 1 else waveform[:, 0]
        if sample_rate != self.target_sample_rate:
            divisor = gcd(int(sample_rate), self.target_sample_rate)
            waveform = resample_poly(
                waveform, self.target_sample_rate // divisor, int(sample_rate) // divisor
            ).astype(np.float32)
        result = AudioSample(np.ascontiguousarray(waveform, dtype=np.float32),
                             self.target_sample_rate, record.sample_id)
        result.validate()
        return result

    def validate_duration(self, record: SampleRecord, audio: AudioSample,
                           tolerance_seconds: float = 0.15) -> None:
        delta = abs(record.duration_seconds - audio.duration_seconds)
        if delta > tolerance_seconds:
            raise ValueError(f"Audio duration mismatch for {record.sample_id}: {delta:.3f}s")

    def export(self, record: SampleRecord, source: DatasetSource,
               destination: str | Path) -> Path:
        audio = self.read(record, source)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio.waveform, audio.sample_rate, format="WAV", subtype="PCM_16")
        return destination


class AudioPreview:
    def __init__(self, reader: AudioReader | None = None):
        self.reader = reader or AudioReader()

    def play(self, record: SampleRecord, source: DatasetSource, autoplay: bool = False) -> None:
        from IPython.display import Audio as NotebookAudio, display

        audio = self.reader.read(record, source)
        print(f"{record.filename} | split={record.split} | {audio.duration_seconds:.2f}s")
        display(NotebookAudio(data=audio.waveform, rate=audio.sample_rate, autoplay=autoplay))

    def play_by_filename(self, index, source: DatasetSource, filename: str,
                          split: str | None = None, autoplay: bool = False) -> SampleRecord:
        record = index.find_by_filename(filename, split=split)
        self.play(record, source, autoplay=autoplay)
        return record

