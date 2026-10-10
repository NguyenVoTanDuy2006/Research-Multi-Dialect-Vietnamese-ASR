# %% [markdown]
# # Trích audio theo tên file từ ViMD
#
# Notebook-script này dùng các module `DatasetSource`, `DatasetIndex` và
# `AudioReader` của project. Audio đầu ra được chuẩn hóa thành WAV mono 16 kHz,
# phù hợp để nghe/đưa vào ASR; transcript và metadata được ghi kèm thành JSON/CSV.
#
# Chọn `ENVIRONMENT` ở cell cấu hình. Kaggle mặc định dò dataset đã attach vào
# Input; local dùng đường dẫn dataset bạn khai báo. Có thể sửa `FILENAMES`/`SPLIT`.

# %% [code]
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from IPython.display import Audio as NotebookAudio, display


def find_project_root() -> Path:
    candidates = [Path.cwd(), *Path.cwd().parents]
    script_file = globals().get("__file__")
    if script_file:
        script_dir = Path(script_file).resolve().parent
        candidates.extend([script_dir, *script_dir.parents])
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "nb").is_dir():
            return candidate.resolve()
    raise FileNotFoundError(
        "Không tìm thấy thư mục gốc project. Hãy chạy script trong repo đã clone."
    )


PROJECT_ROOT = find_project_root()
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from mdv_asr_module.config.types import DatasetConfig
from mdv_asr_module.data.audio import AudioReader
from mdv_asr_module.data.sources import DatasetSourceFactory
from mdv_asr_module.runtime.environment import RuntimeEnvironment

print("Project:", PROJECT_ROOT)
print("Source modules:", SRC_ROOT)

# %% [code]
# Chọn một trong hai: "local" hoặc "kaggle". Path selection nằm trong
# RuntimeEnvironment để các notebook/workflow khác cũng dùng lại được.
ENVIRONMENT = "local"

# Local: thay bằng thư mục chứa các shard Arrow/Parquet của ViMD.
LOCAL_DATASET_ROOT: Path | None = Path(
    "/home/fernir/.cache/huggingface/datasets/nguyendv02___vi_md_dataset/default/0.0.0/3a5b30157034e7eadd5c75fae1a820c6f9383398"
)
# Kaggle: để None để tự dò /kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0.
KAGGLE_DATASET_ROOT: Path | None = None  # Có thể đặt đường dẫn Input cụ thể.

DATASET_HANDLE = "thanhlamdev/vimd-dataset"
DATASET_ID = "vimd"
ALLOW_KAGGLEHUB_DOWNLOAD = False  # Chỉ bật nếu muốn tải dataset khi không attach Input.

runtime = RuntimeEnvironment(platform=ENVIRONMENT, project_root=PROJECT_ROOT)
KAGGLE_DATASET_CANDIDATES = [
    Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0"),
    Path("/kaggle/input/datasets/thanhlamdev/vimd-dataset"),
]
ALLOW_DOWNLOAD = ENVIRONMENT == "kaggle" and ALLOW_KAGGLEHUB_DOWNLOAD
if ALLOW_DOWNLOAD and KAGGLE_DATASET_ROOT is None:
    # DatasetSourceFactory/KaggleHub will resolve the root if Input is absent.
    DATASET_ROOT = None
else:
    DATASET_ROOT = runtime.resolve_dataset_root(
        local_root=LOCAL_DATASET_ROOT,
        kaggle_root=KAGGLE_DATASET_ROOT,
        kaggle_candidates=KAGGLE_DATASET_CANDIDATES,
    )

OUTPUT_DIR = runtime.output_path(
    "extracted_audio",
    local_root=PROJECT_ROOT / "outputs",
    kaggle_root=Path("/kaggle/working"),
    create=True,
)

# Sửa danh sách tên file cần trích. Tên có hoặc không có prefix đều được thử khớp.
FILENAMES = ["30_0309.wav", "59_0313.wav", "37_0338.wav"]
SPLIT: str | None = "test"  # Đặt None để tìm mọi split; trùng tên sẽ báo ambiguous.

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

dataset_config = DatasetConfig(
    kind="auto",
    root=str(DATASET_ROOT) if DATASET_ROOT else None,
    handle=DATASET_HANDLE,
    dataset_id=DATASET_ID,
    cache_dir=None,
    allow_download=ALLOW_DOWNLOAD,
)
dataset_config.validate()
print("Dataset config:", dataset_config)
print("Environment:", ENVIRONMENT)
print("Runtime:", runtime.describe())
print("Output:", OUTPUT_DIR)

# %% [markdown]
# ## Tạo index và tìm mẫu
#
# `build_index()` quét metadata các shard để lập chỉ mục tên file và locator.
# Nó không export toàn bộ audio. Lần quét đầu có thể mất vài phút; sau đó phần
# trích xuất chỉ đọc bytes của những mẫu được chọn.

# %% [code]
source = DatasetSourceFactory.create(dataset_config)
index = source.build_index()
print("Indexed records:", len(index))
selected_records = index.find_many(FILENAMES, split=SPLIT)
print("Matched records:", len(selected_records))
for record in selected_records:
    print(
        f"{record.filename} | id={record.sample_id} | split={record.split} | "
        f"{record.duration_seconds:.2f}s | {record.province_name or 'unknown province'}"
    )

# %% [markdown]
# ## Export WAV và transcript manifest
#
# Tên đầu ra có sample ID để tránh va chạm giữa các split. Transcript được giữ
# nguyên như trong dataset; file audio được đổi thành mono PCM-16 WAV ở 16 kHz.

# %% [code]
reader = AudioReader(target_sample_rate=16000)
manifest_rows = []

for record in selected_records:
    output_name = f"{record.split}_{record.sample_id}_{Path(record.filename).name}"
    audio_path = OUTPUT_DIR / output_name
    reader.export(record, source, audio_path)
    manifest_rows.append({
        "dataset_id": record.dataset_id,
        "sample_id": record.sample_id,
        "split": record.split,
        "filename": record.filename,
        "exported_audio": str(audio_path.resolve()),
        "duration_seconds": record.duration_seconds,
        "transcript": record.text,
        "region": record.region,
        "province_name": record.province_name,
        "speaker_id": record.speaker_id,
    })
    print("Exported:", audio_path)

json_path = OUTPUT_DIR / "extracted_manifest.json"
json_path.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
csv_path = OUTPUT_DIR / "extracted_manifest.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(manifest_rows[0]) if manifest_rows else [
        "dataset_id", "sample_id", "split", "filename", "exported_audio",
        "duration_seconds", "transcript", "region", "province_name", "speaker_id",
    ])
    writer.writeheader()
    writer.writerows(manifest_rows)

print(f"\nĐã trích {len(manifest_rows)} audio.")
print("JSON:", json_path)
print("CSV:", csv_path)

# %% [markdown]
# ## Nghe thử audio đã trích
#
# Chọn chỉ số trong `PREVIEW_INDEX` để phát một mẫu và đối chiếu transcript.

# %% [code]
PREVIEW_INDEX = 0
if manifest_rows:
    preview_row = manifest_rows[PREVIEW_INDEX]
    preview_path = Path(preview_row["exported_audio"])
    print("File:", preview_path.name)
    print("Transcript:", preview_row["transcript"])
    display(NotebookAudio(filename=str(preview_path), autoplay=False))
else:
    print("Không có mẫu để preview.")

# %% [code]
source.close()
print("Dataset source closed.")
