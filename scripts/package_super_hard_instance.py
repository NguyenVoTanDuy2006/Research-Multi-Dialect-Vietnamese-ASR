#!/usr/bin/env python3
"""Extract one ASR test instance from demo/super hard.mp4.

The output uses the same record fields and ZIP member layout as the ViMD
AdaLoRA sample package, but is kept separate because this video is not ViMD.
Requires ffmpeg on PATH; Python dependencies are standard-library only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import wave
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "demo" / "super hard.mp4"
DEFAULT_OUTPUT = ROOT / "outputs" / "super_hard_instance"
SAMPLE_ID = "test_super_hard_000000"
WAV_NAME = "super_hard.wav"
ARCHIVE_MEMBER = f"test/audio/{SAMPLE_ID}__{WAV_NAME}"

# Keep the user-supplied reference verbatim, including spelling and spacing.
TRANSCRIPT = (
    "Hết cải thìa thì gieo cải ngọt. Mà cắt cải ngọt thì gieo cải thìa "
    "với tàng ô với ngò thôi chứ hỗm giờ làm miết cho kịp. Rau xanh là có "
    "mấy món đó thôi. Còn bắp su á thì hỗm cắt trước rồi,mà  bữa nay mưa "
    "có chỗ  chòm hư, rồi có chòm lên trễ."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def package(video: Path, output_dir: Path) -> dict[str, object]:
    video = video.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Không tìm thấy video: {video}")
    if output_dir.exists():
        raise FileExistsError(f"Đầu ra đã tồn tại, không ghi đè: {output_dir}")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("Cần cài ffmpeg và thêm vào PATH.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".super_hard_build_", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        audio_dir = staging / "audio"
        audio_dir.mkdir()
        wav_path = audio_dir / WAV_NAME

        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                "-i", str(video), "-map", "0:a:0", "-vn", "-ac", "1",
                "-ar", "16000", "-c:a", "pcm_s16le", str(wav_path),
            ],
            check=True,
        )
        with wave.open(str(wav_path), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frames = stream.getnframes()
        if (channels, sample_rate, sample_width) != (1, 16000, 2) or frames <= 0:
            raise ValueError("WAV xuất ra không phải mono 16 kHz PCM16 hợp lệ.")

        duration_seconds = frames / sample_rate
        audio_bytes = wav_path.read_bytes()
        record = {
            "sample_id": SAMPLE_ID,
            "split": "test",
            "row_index": None,
            "archive": "test.zip",
            "audio_path": ARCHIVE_MEMBER,
            "filename": WAV_NAME,
            "region": None,
            "province_code": None,
            "province_name": None,
            "speakerID": None,
            "gender": None,
            "gender_code": None,
            "sample_rate": sample_rate,
            "channels": channels,
            "frames": frames,
            "format": "WAV",
            "subtype": "PCM_16",
            "duration_seconds": round(duration_seconds, 6),
            "source_bytes": len(audio_bytes),
            "text": TRANSCRIPT,
        }
        with zipfile.ZipFile(staging / "test.zip", "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr(ARCHIVE_MEMBER, audio_bytes)

        manifest = {
            "schema_version": 1,
            "source_dataset": "external_video",
            "source_video": str(video),
            "source_video_sha256": sha256(video),
            "source_license": None,
            "actual_by_split": {
                "test": {
                    "clips": 1,
                    "duration_seconds": round(duration_seconds, 6),
                    "duration_hours": round(duration_seconds / 3600, 6),
                    "source_bytes": len(audio_bytes),
                    "speakers": 0,
                    "provinces": 0,
                }
            },
            "records": [record],
        }
        write_json(staging / "sample_metadata.json", manifest)
        with (staging / "sample_metadata.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(record))
            writer.writeheader()
            writer.writerow(record)
        (staging / "README.md").write_text(
            "# Super hard — one ASR test instance\n\n"
            "`audio/super_hard.wav` is the extracted mono 16 kHz PCM16 audio. "
            "`test.zip` contains the same WAV at the `audio_path` recorded in "
            "`sample_metadata.json`/`.csv`. The supplied transcript is copied "
            "verbatim, without spelling or whitespace normalization.\n\n"
            "This video is an external source, **not a ViMD record**. Province, "
            "region, speaker, gender, and license are unknown; verify rights "
            "before redistribution. Keep this package separate from the ViMD "
            "train/valid/test set unless you intentionally merge it.\n",
            encoding="utf-8",
        )

        with zipfile.ZipFile(staging / "test.zip") as archive:
            if archive.testzip() is not None or archive.namelist() != [ARCHIVE_MEMBER]:
                raise AssertionError("ZIP không hợp lệ hoặc chứa sai số lượng audio.")
            if archive.read(ARCHIVE_MEMBER) != audio_bytes:
                raise AssertionError("Audio trong ZIP khác file WAV đã xuất.")

        staging.rename(output_dir)
        return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = package(args.video, args.output_dir)
    print(f"Đã tạo: {args.output_dir.resolve()}")
    print(f"Sample: {record['sample_id']} | {record['duration_seconds']:.3f}s | "
          f"{record['sample_rate']} Hz | {record['channels']} kênh")
    print(f"ZIP member: {record['audio_path']}")


if __name__ == "__main__":
    main()
