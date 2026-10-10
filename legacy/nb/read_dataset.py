# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Đọc dataset ViMD cho bài toán ASR
#
# Notebook này đọc dataset `nguyendv02/ViMD_Dataset`, kiểm tra các split và xem transcript/audio. Audio được giữ ở chế độ `decode=False` trong phần kiểm tra metadata; muốn lấy waveform cần cài thêm PyTorch + TorchCodec tương thích.

# %% [markdown]
# ## Context & Methods
#
# - Dataset: ViMD, gồm tiếng nói đa vùng miền tiếng Việt.
# - Output kiểm tra chính: `text` và metadata; audio dùng `decode=False` để không tự giải mã waveform.
# - Notebook không sửa dữ liệu gốc và không tải lại nếu Hugging Face đã có cache.

# %% [code]
from pathlib import Path
from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

DATASET_ID = "nguyendv02/ViMD_Dataset"
CACHE_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"
cache_infos = sorted(
    (CACHE_ROOT / "nguyendv02___vi_md_dataset").rglob("dataset_info.json"),
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)

if cache_infos:
    # Đọc Arrow cache trực tiếp, không cần tải lại hoặc tạo cache lock mới.
    CACHE_DIR = cache_infos[0].parent
    dataset = DatasetDict({
        split: concatenate_datasets([
            Dataset.from_file(str(path))
            for path in sorted(CACHE_DIR.glob(f"vi_md_dataset-{split}-*.arrow"))
        ])
        for split in ("train", "valid", "test")
    })
    print(f"Loaded processed Arrow cache: {CACHE_DIR}")
else:
    # Máy chưa có cache sẽ tải từ Hugging Face và lưu vào cache mặc định.
    dataset = load_dataset(DATASET_ID, download_mode="reuse_dataset_if_exists")

dataset

# %% [markdown]
# ## Data

# %% [code]
split_summary = {
    split: {
        "rows": len(part),
        "columns": part.column_names,
        "features": str(part.features),
    }
    for split, part in dataset.items()
}
split_summary

# %% [code]
# Không decode audio: chỉ xem bytes/path và metadata văn bản.
dataset_no_decode = {
    split: part.cast_column("audio", Audio(decode=False))
    for split, part in dataset.items()
}
metadata_columns = [
    "region", "province_code", "province_name",
    "filename", "text", "speakerID", "gender"
]
preview = dataset_no_decode["train"].select_columns(metadata_columns).to_pandas().head(10)
preview

# %% [code]
sample = dataset_no_decode["train"][0]
audio_info = sample["audio"]
{
    "filename": sample["filename"],
    "text": sample["text"],
    "audio_path": audio_info.get("path"),
    "audio_bytes": len(audio_info.get("bytes", b"")),
}

# %% [code]
from IPython.display import Audio as NotebookAudio, display

# Phát mẫu đầu tiên trong notebook; không cần decode bằng torchcodec.
display(NotebookAudio(data=audio_info["bytes"], autoplay=False))

# %% [markdown]
# ## Results
#
# Các kiểm tra chi tiết hơn về null, speaker leakage, định dạng WAV và tổng thời lượng đã được thực hiện trong [audit report](../docs/vimd_dataset_audit.md).

# %% [code]
assert set(dataset.keys()) == {"train", "valid", "test"}
assert all(len(part) > 0 for part in dataset.values())
assert all("text" in part.column_names and "audio" in part.column_names for part in dataset.values())
print("Dataset loaded successfully")

# %% [markdown]
# ## Takeaways
#
# Sau khi chạy notebook, có thể dùng `dataset_no_decode` để kiểm tra transcript/metadata. Khi huấn luyện ASR, cần bổ sung backend giải mã audio và chuẩn hóa waveform về mono 16 kHz trong pipeline preprocessing.
