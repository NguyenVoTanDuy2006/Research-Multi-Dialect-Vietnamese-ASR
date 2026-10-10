# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Tổng hợp thí nghiệm ViMD — dữ liệu, DI, SR và đối chiếu bài báo
# Nguồn mặc định: `outputs/vimd_local_experiment`. Chỉnh **cell cấu hình** để dùng thư mục hoặc file khác, rồi Run All. Không tải model, không train và không cần GPU. Notebook mặc định chỉ **in số liệu**, không render bảng/biểu đồ trong output; các CSV/PNG vẫn được lưu trong thư mục `analysis`.
#
# Notebook đọc metadata sample, metadata/cache dataset gốc và kết quả theo từng audio. Báo cáo được xuất vào thư mục `analysis`.
#
# Nguồn đối chiếu: [Nguyen et al., EMNLP 2024](https://aclanthology.org/2024.emnlp-main.426/), PDF local `docs/2024.emnlp-main.426.pdf`: bảng 5 (trang in 7480), bảng 6 (7482), bảng 7 (7483), §5.4 và Appendix C–D (7493). Các số baseline bên dưới được chép từ **bảng**, ưu tiên bảng nếu phần diễn giải có số khác.
#
# ## Cách đọc kết quả
# - DI: F1-macro theo bộ nhãn cố định của task; accuracy là chỉ số riêng. Báo cả số lớp thực sự có trong sample.
# - SR: WER/CER corpus = tổng số edit / tổng độ dài reference, không lấy trung bình WER các câu. WER tách token theo khoảng trắng; CER bỏ khoảng trắng. Chuẩn hóa NFC, chữ thường và bỏ dấu câu.
# - Sample nhỏ so với bài báo; chênh lệch điểm chỉ mang tính mô tả, không chứng minh mô hình tốt/kém hơn.
# - SR chia theo vùng từ một checkpoint toàn quốc là **đánh giá theo vùng**, không có nghĩa đã fine-tune riêng từng vùng.
# - Không dùng aggregate có sẵn để trộn với dòng sample; lỗi inference được thống kê riêng, không coi là dự đoán rỗng thành công.
#

# %% [code]
# Nếu thiếu thư viện, chạy một lần trong kernel của bạn:
# %pip install pandas numpy matplotlib pyarrow soundfile nbformat nbclient rapidfuzz
from pathlib import Path
import json
import re
import unicodedata
import hashlib
import struct
import itertools
from datetime import datetime, timezone
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from IPython.display import Audio as NotebookAudio, display, Markdown
from rapidfuzz.distance import Levenshtein

pd.set_option("display.max_columns", 24)
pd.set_option("display.max_colwidth", 75)
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": False})
COLORS = ["#2563a6", "#d18c24", "#8b668b"]
SPLITS = ["train", "valid", "test"]
REGIONS = ["North", "Central", "South"]
SHOW_TABLES = False
SHOW_PLOTS = False


# %% [markdown]
# ## 1. Cấu hình đầu vào
# Ưu tiên metadata CSV/JSON bạn chỉ định. Nếu chưa có metadata dataset gốc, notebook tự tìm Arrow cache local ViMD; có thể đặt `DATASET_ROOT` thành thư mục dataset đã upload lên Kaggle. Chỉ đọc header audio để lấy thời lượng, không decode waveform.
#
# `DI_MODEL_OVERRIDES` chỉ điền khi đã xác nhận backbone. Dòng kết quả ghi model=`DI` không đủ để suy ra backbone. Nếu có `di_checkpoints/<task>/di_config.json`, notebook tự đọc. `SR_STAGE` mặc định direct inference theo notebook thực nghiệm hiện tại.
#

# %% [code]
PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents] if (p / "docs").is_dir() and (p / "nb").is_dir()),
    Path.cwd()
)
EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "vimd_local_experiment"
SAMPLE_METADATA_FILE = EXPERIMENT_DIR / "vimd_sample_metadata.json"  # .json hoặc .csv
RESULTS_FILE = EXPERIMENT_DIR / "vimd_results.csv"
FULL_METADATA_FILE = None  # CSV/JSON: split, filename, region, province_name, duration_seconds
DATASET_ROOT = None        # Arrow/Parquet gốc; None = tự tìm HF cache local
REPORT_DIR = EXPERIMENT_DIR / "analysis"

# Điền đúng backbone nếu di_config.json không đi kèm kết quả, ví dụ:
# DI_MODEL_OVERRIDES = {"DI_VN_3": "nguyenvulebinh/wav2vec2-base-vi"}
DI_MODEL_OVERRIDES = {}
# Dùng khi CSV ghi model=DI nhưng có nhiều checkpoint/backbone.
# Key có thể là đường dẫn checkpoint đầy đủ hoặc chỉ tên thư mục checkpoint.
DI_CHECKPOINT_MODEL_OVERRIDES = {}
SR_STAGE = "direct"        # "direct", "finetuned", "unknown"
RESULTS_SPLIT_DEFAULT = "test"  # dùng cho dòng error thiếu split; được ghi vào báo cáo
EVALUATION_SPLIT = "test"
NORMALIZE_PUNCTUATION = True
CER_REMOVE_SPACES = True
MAX_EXAMPLES = 10
# Audio phát thử trong notebook (để False nếu chỉ muốn phân tích số liệu).
PLAY_AUDIO = False
AUDIO_FILENAME = None  # ví dụ: "81_0302.wav"
AUDIO_SPLIT = None      # None hoặc "train"/"valid"/"test"
AUDIO_CENTRAL_ONLY = False
FAIL_ON_DUPLICATE_RESULTS = True

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = REPORT_DIR / "figures"
FIGURE_DIR.mkdir(exist_ok=True)
print("Experiment:", EXPERIMENT_DIR)
print("Report:", REPORT_DIR)


# %% [markdown]
# ## 2. Đọc dữ liệu và kiểm tra nguồn
# CSV và JSON metadata là hai cách biểu diễn của cùng tập sample; chỉ đọc **một** file được chọn. Thời lượng bị thiếu/không hợp lệ được giữ là NA, không đổi thành 0. Không đọc `di_cache/*preprocessed*` làm dataset gốc vì đó là dữ liệu đã biến đổi.
#

# %% [code]
issues, source_files, exported_tables = [], [], {}
def notice(message):
    issues.append(str(message))
    print("Lưu ý:", message)

def record_source(path):
    path = Path(path).resolve()
    source_files.append({"path": str(path), "size_bytes": path.stat().st_size,
                         "mtime_ns": path.stat().st_mtime_ns})
    return path

def read_metadata(path):
    path = record_source(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, keep_default_na=False)
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frame = pd.DataFrame(payload["records"] if isinstance(payload, dict) else payload)
    return frame

def clean_metadata(frame, name):
    frame = frame.copy()
    aliases = {"set": "split", "length": "duration_seconds", "duration": "duration_seconds",
               "province": "province_name", "original_dataset_index": "row_index"}
    for old, new in aliases.items():
        if old in frame and new not in frame:
            frame = frame.rename(columns={old: new})
    for col in ["split", "filename", "region", "province_name", "speakerID"]:
        if col not in frame:
            frame[col] = pd.NA
        frame[col] = frame[col].replace("", pd.NA).astype("string")
    frame["split"] = frame["split"].replace({"validation": "valid"})
    if "duration_seconds" not in frame:
        frame["duration_seconds"] = np.nan
    frame["duration_seconds"] = pd.to_numeric(frame["duration_seconds"], errors="coerce")
    bad = ~np.isfinite(frame["duration_seconds"]) | (frame["duration_seconds"] <= 0)
    if bad.any():
        notice(f"{name}: {bad.sum()} dòng thiếu/thời lượng không hợp lệ.")
        frame.loc[bad, "duration_seconds"] = np.nan
    for col in ["split", "region", "province_name"]:
        if frame[col].isna().any():
            notice(f"{name}: {frame[col].isna().sum()} dòng thiếu {col}.")
    keys = frame[["split", "filename"]].dropna()
    if keys.duplicated().any():
        raise ValueError(f"{name}: trùng (split, filename); cần xác định nguồn trước khi tổng hợp.")
    frame["dataset"] = name
    return frame

sample = clean_metadata(read_metadata(SAMPLE_METADATA_FILE), "sample")
print("Sample metadata:", len(sample), "dòng")

def split_from_path(path):
    tokens = re.split(r"[/_.-]+", str(path).lower())
    return next((s for s in SPLITS if s in tokens), "valid" if "validation" in tokens else None)

def discover_dataset_root():
    roots = list((Path.home() / ".cache/huggingface/datasets/nguyendv02___vi_md_dataset").rglob("dataset_info.json"))
    roots = sorted(roots, key=lambda p: p.stat().st_mtime_ns, reverse=True)
    return roots[0].parent if roots else None

def header_info(buffer):
    # RIFF/WAV fast path: read fmt/data chunk headers, never scan waveform.
    view = memoryview(buffer)
    if len(view) >= 12 and bytes(view[:4]) == b'RIFF' and bytes(view[8:12]) == b'WAVE':
        position, fmt, data_size = 12, None, None
        while position + 8 <= len(view):
            tag = bytes(view[position:position+4])
            size = struct.unpack_from('<I', view, position+4)[0]
            start = position + 8
            if start + size > len(view):
                raise ValueError('Truncated RIFF chunk')
            if tag == b'fmt ' and size >= 16:
                encoding, channels, rate, byte_rate, align, bits = struct.unpack_from('<HHIIHH', view, start)
                fmt = (encoding, channels, rate, align)
            elif tag == b'data':
                data_size = size
            if fmt and data_size is not None:
                encoding, channels, rate, align = fmt
                if encoding in (1, 3, 65534) and align > 0 and rate > 0:
                    return dict(duration_seconds=(data_size // align)/rate, sample_rate=rate, channels=channels)
                break
            position = start + size + (size % 2)
    # Non-PCM/container fallback; still no waveform decoding.
    with pa.BufferReader(buffer) as stream:
        info = sf.info(stream)
    return dict(duration_seconds=info.frames/info.samplerate, sample_rate=info.samplerate, channels=info.channels)

def metadata_from_batches(batches, split, source):
    rows = []
    for batch in batches:
        names = batch.schema.names
        keep = [x for x in ["filename", "region", "province_code", "province_name", "speakerID", "gender",
                            "duration_seconds", "length"] if x in names]
        records = batch.select(keep).to_pylist()
        audio_col = batch.column(names.index("audio")) if "audio" in names else None
        if audio_col is not None:
            binary = audio_col.field("bytes")
            buffers = binary.buffers()
            offset_type = np.int64 if pa.types.is_large_binary(binary.type) else np.int32
            offsets = np.frombuffer(buffers[1], dtype=offset_type)
            valid_bytes = binary.is_valid().to_numpy(zero_copy_only=False)
        for i, row in enumerate(records):
            row["split"] = split
            row["source_shard"] = str(source)
            duration = row.get("duration_seconds", row.get("length"))
            if duration is None and audio_col is not None:
                try:
                    if valid_bytes[i]:
                        j = binary.offset + i
                        begin, end = int(offsets[j]), int(offsets[j+1])
                        meta = header_info(buffers[2].slice(begin, end-begin))
                    else:
                        audio_path = Path(audio_col.field("path")[i].as_py())
                        if not audio_path.is_absolute():
                            audio_path = source.parent / audio_path
                        info = sf.info(audio_path)
                        meta = dict(duration_seconds=info.frames/info.samplerate, sample_rate=info.samplerate, channels=info.channels)
                    duration = meta["duration_seconds"]
                    row.update(meta)
                except Exception as exc:
                    row["metadata_error"] = repr(exc)
            row["duration_seconds"] = duration
            rows.append(row)
    return rows

full = pd.DataFrame()
if FULL_METADATA_FILE is not None:
    full = clean_metadata(read_metadata(FULL_METADATA_FILE), "full_dataset")
else:
    root = Path(DATASET_ROOT) if DATASET_ROOT else discover_dataset_root()
    if root is None:
        notice("Chưa có nguồn dataset gốc: điền FULL_METADATA_FILE hoặc DATASET_ROOT để so sánh theo tỉnh/vùng.")
    else:
        candidates = sorted(root.rglob("vi_md_dataset-*.arrow"))
        if not candidates:
            candidates = sorted(root.rglob("*.parquet"))
        candidates = [p for p in candidates if split_from_path(p) and "di_cache" not in p.parts]
        rows = []
        for number, path in enumerate(candidates, 1):
            path = record_source(path)
            split = split_from_path(path)
            if path.suffix == ".arrow":
                with pa.memory_map(str(path), "r") as source:
                    rows.extend(metadata_from_batches(pa.ipc.open_stream(source), split, path))
            else:
                rows.extend(metadata_from_batches(pq.ParquetFile(path).iter_batches(batch_size=64), split, path))
            if number % 20 == 0 or number == len(candidates):
                print(f"Header metadata: {number}/{len(candidates)} shards, {len(rows)} audio")
        if rows:
            full = clean_metadata(pd.DataFrame(rows), "full_dataset")
        else:
            notice(f"Không tìm thấy shard ViMD có split trong {root}.")

if not full.empty:
    # Add original row positions in the same sorted-shard order, without reading audio again.
    full["row_index"] = full.groupby("split", sort=False).cumcount()
    lookup = full[["split", "filename"]]
    membership = sample.merge(lookup.assign(in_original=True), on=["split", "filename"], how="left", validate="one_to_one")
    if membership["in_original"].isna().any():
        notice(f"{membership['in_original'].isna().sum()} sample không khớp dataset gốc theo split+filename.")
    for frame, name in [(sample, "sample"), (full, "full_dataset")]:
        sets = {s: set(frame.loc[frame["split"].eq(s), "speakerID"].dropna()) for s in SPLITS}
        for a,b in itertools.combinations(SPLITS, 2):
            overlap = sets[a] & sets[b]
            if overlap:
                notice(f"{name}: {len(overlap)} speakerID giao nhau {a}/{b}; cần kiểm tra tính độc lập người nói.")

data = pd.concat([full, sample], ignore_index=True)
if SHOW_TABLES:
    display(data.groupby(["dataset", "split"], dropna=False).size().rename("n_audio").to_frame())
else:
    counts = data.groupby(["dataset", "split"], dropna=False).size()
    print('Đã đọc dữ liệu:', '; '.join(f'{dataset}/{split}={int(n)} audio' for (dataset, split), n in counts.items()))

def play_audio_by_filename(filename, split=None, central_only=False):
    """Tìm và phát một file WAV theo tên gốc, kể cả tên sample có prefix."""
    target = Path(str(filename)).name.casefold()
    frames = []
    for frame in (sample, full):
        if frame is None or frame.empty or "filename" not in frame.columns:
            continue
        mask = frame["filename"].map(lambda value: (
            pd.notna(value) and (Path(str(value)).name.casefold() == target or
                                 Path(str(value)).name.casefold().endswith("_" + target))))
        subset = frame.loc[mask].copy()
        if split is not None and "split" in subset:
            subset = subset[subset["split"].eq(split)]
        if central_only and "region" in subset:
            subset = subset[subset["region"].eq("Central")]
        if not subset.empty:
            frames.append(subset)
    metadata = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    paths = []
    if not metadata.empty and "audio_path" in metadata:
        for value in metadata["audio_path"].dropna().tolist():
            path = Path(str(value))
            if path.is_file():
                paths.append(path)
    # Sample WAV thường được lưu với dạng 00001_test_<tên-gốc>.wav.
    search_dirs = [EXPERIMENT_DIR / "sample_30min_audio", EXPERIMENT_DIR / "sample_4h_audio", EXPERIMENT_DIR]
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.wav"):
            name = path.name.casefold()
            if name == target or name.endswith("_" + target):
                paths.append(path)
    unique_paths = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
    if not unique_paths:
        hint = " (lọc miền Trung)" if central_only else ""
        raise FileNotFoundError(f"Không tìm thấy audio '{filename}'{hint} trong metadata/sample_30min_audio.")
    path = unique_paths[0]
    print(f"Phát: {path}")
    if not metadata.empty:
        cols = [c for c in ["split", "filename", "region", "province_name", "duration_seconds"] if c in metadata]
        if cols:
            print(metadata[cols].drop_duplicates().to_string(index=False))
    display(NotebookAudio(filename=str(path), autoplay=False))
    return path

if PLAY_AUDIO:
    if not AUDIO_FILENAME:
        raise ValueError("Đặt AUDIO_FILENAME, ví dụ '81_0302.wav', trước khi phát audio.")
    play_audio_by_filename(AUDIO_FILENAME, split=AUDIO_SPLIT, central_only=AUDIO_CENTRAL_ONLY)


# %% [markdown]
# ## 3. Số audio, thời lượng và phân bổ
# Mỗi hàng metadata tương ứng một utterance gốc. Không cộng số đoạn DI sau chia nhỏ vào số audio. Tổng giờ ở nhóm thiếu duration chỉ là **tổng phần đo được**; xem `n_duration_missing` trước khi sử dụng.
#

# %% [code]
def save_table(name, frame):
    frame.to_csv(REPORT_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    exported_tables[name] = len(frame)
    return frame

def summarize(frame, keys):
    out = frame.groupby(keys, dropna=False, observed=True).agg(
        n_audio=("filename", "size"),
        duration_seconds=("duration_seconds", lambda s: s.sum(min_count=1)),
        n_duration_known=("duration_seconds", "count"),
        n_provinces=("province_name", "nunique"),
        n_speakers=("speakerID", "nunique"),
    ).reset_index()
    out["n_duration_missing"] = out["n_audio"] - out["n_duration_known"]
    out["duration_hours"] = out["duration_seconds"] / 3600
    return out

split_summary = save_table("dataset_by_split", summarize(data, ["dataset", "split"]))
region_summary = save_table("dataset_by_region", summarize(data, ["dataset", "split", "region"]))
province_summary = save_table("dataset_by_province", summarize(data, ["dataset", "split", "region", "province_name"]))
total_summary = save_table("dataset_totals", summarize(data, ["dataset"]))
if SHOW_TABLES:
    display(total_summary)
    display(split_summary)
    display(region_summary)
    display(province_summary.head(15))
else:
    print('Dataset totals:')
    for row in total_summary.to_dict('records'):
        print(f"  {row['dataset']}: n={int(row['n_audio'])}, hours={row['duration_hours']:.3f}")
    print('Dataset split totals:')
    for row in split_summary.to_dict('records'):
        print(f"  {row['dataset']}/{row['split']}: n={int(row['n_audio'])}, hours={row['duration_hours']:.3f}")
print("CSV phân bổ theo tỉnh:", REPORT_DIR / "dataset_by_province.csv")

def finish_plot(name):
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

for measure, label in [("n_audio", "Số audio"), ("duration_hours", "Thời lượng (giờ)")]:
    names = list(data["dataset"].unique())
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 4), squeeze=False)
    for ax, name in zip(axes[0], names):
        table = region_summary[region_summary["dataset"].eq(name)].pivot(index="split", columns="region", values=measure)
        table = table.reindex(SPLITS)
        table.plot.bar(ax=ax, color=COLORS, rot=0)
        ax.set(title=f"{name}: {label} theo split/vùng", ylabel=label, xlabel="Split")
        ax.set_ylim(bottom=0)
        ax.legend(title="Vùng")
    finish_plot(f"dataset_{measure}")


# %% [markdown]
# ## 4. Sample so với dataset gốc
# Tỷ lệ lấy mẫu = số lượng/thời lượng sample chia số tương ứng của dataset gốc. Chênh lệch cơ cấu dùng **điểm phần trăm**, so sánh trong cùng split. Tỉnh/vùng không có mẫu được giữ lại với số lượng 0 khi đã có đầy đủ metadata của split.
#

# %% [code]
def compare_distribution(keys, name):
    if full.empty:
        return pd.DataFrame()
    original = summarize(full, keys)[keys + ["n_audio", "duration_hours", "n_duration_missing"]]
    sampled = summarize(sample, keys)[keys + ["n_audio", "duration_hours", "n_duration_missing"]]
    joined = original.merge(sampled, on=keys, how="outer", suffixes=("_full", "_sample"), indicator=True)
    absent = joined["_merge"].eq("left_only") & joined["split"].isin(sample["split"].dropna())
    joined.loc[absent, ["n_audio_sample", "duration_hours_sample", "n_duration_missing_sample"]] = 0
    for measure in ["n_audio", "duration_hours"]:
        joined[f"{measure}_retained_pct"] = 100 * joined[f"{measure}_sample"] / joined[f"{measure}_full"]
        for tag in ["full", "sample"]:
            denom = joined.groupby("split")[f"{measure}_{tag}"].transform(lambda x: x.sum(min_count=1))
            joined[f"{measure}_{tag}_share_pct"] = 100 * joined[f"{measure}_{tag}"] / denom
        joined[f"{measure}_share_delta_pp"] = joined[f"{measure}_sample_share_pct"] - joined[f"{measure}_full_share_pct"]
    return save_table(name, joined)

split_comparison = compare_distribution(["split"], "sample_vs_full_split")
region_comparison = compare_distribution(["split", "region"], "sample_vs_full_region")
province_comparison = compare_distribution(["split", "region", "province_name"], "sample_vs_full_province")
if SHOW_TABLES:
    display(split_comparison)
    display(region_comparison)
if not province_comparison.empty:
    for split in SPLITS:
        sub = province_comparison[province_comparison["split"].eq(split)].sort_values(["region", "province_name"])
        fig, axes = plt.subplots(1, 2, figsize=(15, max(6, len(sub) * .22)))
        labels = sub["province_name"].fillna("Unknown")
        y = np.arange(len(sub))
        for ax, measure, title in zip(axes, ["n_audio", "duration_hours"], ["Cơ cấu số audio (%)", "Cơ cấu thời lượng (%)"]):
            ax.barh(y - .18, sub[f"{measure}_full_share_pct"], .36, label="Dataset gốc", color=COLORS[0])
            ax.barh(y + .18, sub[f"{measure}_sample_share_pct"], .36, label="Sample", color=COLORS[1])
            ax.set_yticks(y, labels)
            ax.invert_yaxis()
            ax.set(title=f"{split} — {title}", xlabel="% trong split")
            ax.set_xlim(left=0)
            ax.legend()
        finish_plot(f"province_distribution_{split}")


# %% [markdown]
# ## 5. Đọc kết quả và xác định task/mô hình
# Các dòng `aggregate` chỉ dùng để kiểm tra cách tính cũ. Không cộng lại với kết quả từng audio. Mọi model lỗi vẫn xuất hiện ở bảng coverage; prediction rỗng nhưng không có exception được tính là lỗi nhận dạng, không bị loại.
#
# Nếu kết quả chứa nhiều lần chạy cho cùng task/model/checkpoint/audio, notebook dừng để bạn chọn đúng run thay vì tự ý chọn điểm tốt nhất.
#

# %% [code]
results = pd.read_csv(record_source(RESULTS_FILE), keep_default_na=False)
for column in ["record_type", "task", "model", "checkpoint", "split", "filename", "region",
               "province_name", "reference", "prediction", "error"]:
    if column not in results:
        results[column] = ""
saved_aggregates = results[results["record_type"].eq("aggregate")].copy()
attempts = results[results["record_type"].isin(["sample", "error"])].copy()
attempts["split"] = attempts["split"].replace({"": RESULTS_SPLIT_DEFAULT, "validation": "valid"})
attempts = attempts[attempts["split"].eq(EVALUATION_SPLIT)].copy()
attempts["success"] = attempts["record_type"].eq("sample") & attempts["error"].eq("")
if "stage" not in attempts:
    attempts["stage"] = ""
sr_mask = attempts["task"].str.startswith("SR")
attempts.loc[sr_mask, "stage"] = attempts.loc[sr_mask, "stage"].replace("", SR_STAGE)
attempts.loc[~sr_mask, "stage"] = attempts.loc[~sr_mask, "stage"].replace("", "finetuned")

aliases = {"whisperbase": "whisper-base", "PhoWhisperbase": "PhoWhisper-base",
           "phowhisper-base": "PhoWhisper-base", "base": "whisper-base"}
def canonical_model(value):
    value = str(value).rstrip("/").split("/")[-1]
    return aliases.get(value, value)

def resolve_di(row):
    if not row["task"].startswith("DI"):
        return canonical_model(row["model"])
    if row["task"] in DI_MODEL_OVERRIDES:
        return canonical_model(DI_MODEL_OVERRIDES[row["task"]])
    checkpoint = str(row.get("checkpoint", "") or "").rstrip("/")
    checkpoint_name = Path(checkpoint).name
    for key in [checkpoint, checkpoint_name]:
        if key and key in DI_CHECKPOINT_MODEL_OVERRIDES:
            return canonical_model(DI_CHECKPOINT_MODEL_OVERRIDES[key])
    for path in [EXPERIMENT_DIR / "di_checkpoints" / row["task"] / "di_config.json",
                 Path(row["checkpoint"]) / "di_config.json"]:
        if path.is_file():
            config = json.loads(record_source(path).read_text())
            if config.get("backbone"):
                return canonical_model(config["backbone"])
    if row["model"] != "DI":
        return canonical_model(row["model"])
    return "UNKNOWN_DI_BACKBONE"

# Resolve each group once, rather than opening checkpoint metadata for every audio.
mapping = {}
for key, group in attempts.groupby(["task", "model", "checkpoint"], dropna=False):
    mapping[key] = resolve_di(group.iloc[0])
attempts["model_resolved"] = [mapping[(r.task, r.model, r.checkpoint)] for r in attempts.itertuples()]
if attempts["model_resolved"].eq("UNKNOWN_DI_BACKBONE").any():
    notice("Không có di_config.json và model DI chỉ ghi 'DI': tính F1 bình thường; chưa gán baseline cùng backbone. Điền DI_MODEL_OVERRIDES sau khi xác nhận.")

# Enrich missing region/province from metadata, with a many-to-one validated join.
lookup = (full if not full.empty else sample)[["split", "filename", "region", "province_name"]]
attempts = attempts.merge(lookup, on=["split", "filename"], how="left", suffixes=("", "_meta"), validate="many_to_one")
for col in ["region", "province_name"]:
    attempts[col] = attempts[col].replace("", pd.NA).fillna(attempts[f"{col}_meta"])
keys = ["task", "model_resolved", "checkpoint", "stage", "split", "filename"]
dupes = attempts["filename"].ne("") & attempts.duplicated(keys, keep=False)
if dupes.any():
    save_table("duplicate_results", attempts.loc[dupes])
    if FAIL_ON_DUPLICATE_RESULTS:
        raise ValueError("Trùng kết quả: xem duplicate_results.csv và chọn file của một run.")
    notice("Có kết quả trùng: chưa loại tự động; metric sẽ đếm cả các dòng này.")
success = attempts[attempts["success"]].copy()
coverage = attempts.groupby(["task", "model_resolved", "checkpoint", "stage"], dropna=False).agg(
    n_rows=("success", "size"), n_success=("success", "sum"), n_files=("filename", lambda x: x[x.ne("")].nunique())
).reset_index()
coverage["n_errors"] = coverage["n_rows"] - coverage["n_success"]
coverage["success_pct"] = 100 * coverage["n_success"] / coverage["n_rows"]
save_table("result_coverage", coverage)
if SHOW_TABLES:
    display(coverage)
else:
    print('Inference coverage:')
    for row in coverage.to_dict('records'):
        print(f"  {row['task']} | {row['model_resolved']} | success={int(row['n_success'])}/{int(row['n_rows'])} ({row['success_pct']:.2f}%)")
save_table("inference_errors", attempts[~attempts["success"]])
if (~attempts["success"]).any():
    notice("WER/F1 chỉ tính trên inference thành công; xem n_errors vì coverage khác nhau có thể gây thiên lệch.")


# %% [markdown]
# ## 6. Dialect Identification — 5 task của bài báo
# DI_VN_3: vùng; DI_North/Central/South: tỉnh trong vùng; DI_VN_63: toàn bộ tỉnh. F1-macro tính với bộ nhãn 3/25/19/19/63 nếu metadata gốc đầy đủ. Lớp không có reference và prediction nhận F1=0 (`zero_division=0`); kèm thêm F1 trên lớp có reference để thấy tác động của sample nhỏ. Không coi F1 là accuracy. Mỗi tổ hợp `task + model/backbone + checkpoint` được tính riêng, nên nhiều mô hình DI sẽ không bị gộp.
#

# %% [code]
DI_TASKS = {"DI_VN_3": (None, 3), "DI_North": ("North", 25),
            "DI_Central": ("Central", 19), "DI_South": ("South", 19), "DI_VN_63": (None, 63)}
di_rows, class_rows = [], []
label_source = full if not full.empty else sample
for (task, model, checkpoint, stage), group in success[success["task"].str.startswith("DI")].groupby(
    ["task", "model_resolved", "checkpoint", "stage"], dropna=False
):
    if task not in DI_TASKS:
        notice(f"Task DI chưa có định nghĩa: {task}")
        continue
    region, n_expected = DI_TASKS[task]
    labels = REGIONS if task == "DI_VN_3" else sorted(label_source.loc[
        label_source["region"].eq(region) if region else np.ones(len(label_source), dtype=bool), "province_name"
    ].dropna().unique())
    y_true, y_pred = group["reference"].astype(str), group["prediction"].astype(str)
    unknown = (set(y_true) | set(y_pred)) - set(labels)
    if unknown:
        notice(f"{task}: nhãn ngoài bộ nhãn metadata: {sorted(unknown)}; kiểm tra name/id mapping.")
    scores, supported = [], []
    for label in labels:
        tp = int(((y_true == label) & (y_pred == label)).sum())
        fp = int(((y_true != label) & (y_pred == label)).sum())
        fn = int(((y_true == label) & (y_pred != label)).sum())
        support = int((y_true == label).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.
        scores.append(f1)
        if support:
            supported.append(f1)
        class_rows.append(dict(task=task, model=model, checkpoint=checkpoint, stage=stage, label=label, support=support,
                               precision=tp/(tp+fp) if tp+fp else 0., recall=tp/(tp+fn) if tp+fn else 0., f1=f1))
    di_rows.append(dict(task=task, model=model, checkpoint=checkpoint, stage=stage, n_samples=len(group),
                        n_labels=len(labels), n_expected_labels=n_expected, n_labels_observed=y_true.nunique(),
                        label_universe_complete=len(labels)==n_expected and not unknown,
                        f1_macro=float(np.mean(scores)) if scores else np.nan,
                        f1_observed_classes=float(np.mean(supported)) if supported else np.nan,
                        accuracy=float((y_true==y_pred).mean())))
    # Row-normalized confusion, including unused labels to preserve task coverage.
    matrix_labels = list(labels) + sorted(unknown)
    cm = pd.crosstab(y_true, y_pred).reindex(index=matrix_labels, columns=matrix_labels, fill_value=0)
    cm.index.name = "reference"
    tag = re.sub(r"[^A-Za-z0-9_-]", "_", task + "_" + model + "_" + hashlib.sha256(checkpoint.encode()).hexdigest()[:6])
    save_table(f"confusion_counts_{tag}", cm.reset_index())
    rates = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0)
    fig, ax = plt.subplots(figsize=(max(5, len(matrix_labels)*.25), max(4, len(matrix_labels)*.25)))
    im = ax.imshow(np.ma.masked_invalid(rates.to_numpy()), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(matrix_labels)), matrix_labels, rotation=90)
    ax.set_yticks(range(len(matrix_labels)), matrix_labels)
    ax.set(title=f"{task} — {model}\nHàng trống: không có reference trong sample",
           xlabel="Prediction", ylabel="Reference")
    fig.colorbar(im, ax=ax, label="Tỷ lệ theo reference")
    finish_plot(f"confusion_{tag}")

di_summary = save_table("di_task_metrics", pd.DataFrame(di_rows))
di_class = save_table("di_class_metrics", pd.DataFrame(class_rows))
print('\nDI F1-macro theo task (0-1, format paper):')
if di_summary.empty:
    print('  Không có kết quả DI thành công.')
else:
    for row in di_summary.sort_values(['model', 'stage', 'task', 'checkpoint']).to_dict('records'):
        checkpoint_name = Path(str(row['checkpoint']).rstrip('/')).name
        print(f"  {row['model']} | checkpoint={checkpoint_name} | {row['stage']} | {row['task']}: F1-macro={row['f1_macro']:.4f}, n={int(row['n_samples'])}, labels={int(row['n_labels_observed'])}/{int(row['n_expected_labels'])}")
if SHOW_TABLES:
    display(di_summary)
task_status = pd.DataFrame({"task": list(DI_TASKS)})
task_status["has_results"] = task_status["task"].isin(di_summary["task"] if not di_summary.empty else [])
save_table("di_task_coverage", task_status)
if SHOW_TABLES:
    display(task_status)



# %% [markdown]
# ## 7. Speech Recognition — WER/CER toàn quốc, vùng và tỉnh
# Tính lại từ reference/prediction, giữ lại WER từng câu để xem outlier. CER là bổ sung của thực nghiệm này; bảng 7 bài báo không công bố CER nên không tạo baseline CER giả.
#
# Không có mẫu cho một task/model ≠ điểm 0. Với reference rỗng, edit vẫn được cộng vào numerator corpus; nếu toàn nhóm không có token reference thì rate là NA. Coverage và số câu phải được xem cùng metric.
#

# %% [code]
def normalized_text(value):
    text = unicodedata.normalize("NFC", str(value)).lower()
    if NORMALIZE_PUNCTUATION:
        text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())

sr = success[success["task"].str.startswith("SR")].copy()
sr_records = []
for row in sr.to_dict("records"):
    ref, pred = normalized_text(row["reference"]), normalized_text(row["prediction"])
    ref_chars = ref.replace(" ", "") if CER_REMOVE_SPACES else ref
    pred_chars = pred.replace(" ", "") if CER_REMOVE_SPACES else pred
    row.update(word_edits=Levenshtein.distance(ref.split(), pred.split()), n_ref_words=len(ref.split()),
               char_edits=Levenshtein.distance(ref_chars, pred_chars), n_ref_chars=len(ref_chars),
               empty_prediction=not bool(pred))
    row["wer_recomputed"] = row["word_edits"] / row["n_ref_words"] if row["n_ref_words"] else np.nan
    row["cer_recomputed"] = row["char_edits"] / row["n_ref_chars"] if row["n_ref_chars"] else np.nan
    sr_records.append(row)
sr = pd.DataFrame(sr_records)
sr_metrics, sr_provinces = [], []
def sr_measure(group):
    nwords, nchars = group["n_ref_words"].sum(), group["n_ref_chars"].sum()
    times = pd.to_numeric(group.get("inference_seconds", pd.Series(index=group.index, dtype=float)), errors="coerce")
    return dict(n_samples=len(group), word_edits=int(group["word_edits"].sum()), n_ref_words=int(nwords),
                char_edits=int(group["char_edits"].sum()), n_ref_chars=int(nchars),
                wer=group["word_edits"].sum()/nwords if nwords else np.nan,
                cer=group["char_edits"].sum()/nchars if nchars else np.nan,
                wer_sentence_mean=group["wer_recomputed"].mean(), cer_sentence_mean=group["cer_recomputed"].mean(),
                n_empty_prediction=int(group["empty_prediction"].sum()),
                inference_seconds_total=times.sum(min_count=1), inference_seconds_mean=times.mean())

if not sr.empty:
    save_table("sr_sample_metrics", sr)
    for (model, checkpoint, stage, source_task), group in sr.groupby(["model_resolved", "checkpoint", "stage", "task"]):
        task_groups = [("SR_VN_63", group)] if source_task in ["SR", "SR_VN_63"] else [(source_task, group)]
        if source_task in ["SR", "SR_VN_63"]:
            task_groups += [("SR_" + region, group[group["region"].eq(region)]) for region in REGIONS]
        for task, subset in task_groups:
            if subset.empty:
                continue
            scope = "regional_slice" if task != source_task and task != "SR_VN_63" else "reported_task"
            sr_metrics.append(dict(task=task, source_task=source_task, model=model, checkpoint=checkpoint, stage=stage,
                                   scope=scope, **sr_measure(subset)))
        for (region, province), subset in group.groupby(["region", "province_name"], dropna=False):
            sr_provinces.append(dict(model=model, checkpoint=checkpoint, stage=stage, source_task=source_task,
                                      region=region, province_name=province, **sr_measure(subset)))
    save_table("sr_error_examples", sr.sort_values("wer_recomputed", ascending=False).groupby("model_resolved").head(MAX_EXAMPLES)[
        ["model_resolved", "filename", "region", "province_name", "reference", "prediction", "wer_recomputed", "cer_recomputed"]])

sr_summary = save_table("sr_task_metrics", pd.DataFrame(sr_metrics))
sr_province_summary = save_table("sr_province_metrics", pd.DataFrame(sr_provinces))
print('\nSR WER/CER theo task (0-1; WER là chỉ số paper công bố):')
if sr_summary.empty:
    print('  Không có kết quả SR thành công.')
else:
    for row in sr_summary.sort_values(['model', 'stage', 'task']).to_dict('records'):
        print(f"  {row['model']} | {row['stage']} | {row['task']}: WER={row['wer']:.4f}, CER={row['cer']:.4f}, n={int(row['n_samples'])}")

if SHOW_TABLES:
    display(sr_summary)
if not sr_summary.empty:
    for metric in ["wer", "cer"]:
        # Separate checkpoints/stages so distinct runs are not silently averaged.
        plot_data = sr_summary.copy()
        plot_data["series"] = plot_data["model"] + " | " + plot_data["stage"] + " | " + plot_data["checkpoint"].map(
            lambda s: hashlib.sha256(str(s).encode()).hexdigest()[:6])
        plot_data["percent"] = 100 * plot_data[metric]
        if not plot_data.duplicated(["task", "series"]).any():
            table = plot_data.pivot(index="series", columns="task", values="percent")
            ax = table.plot.barh(figsize=(12, max(4, len(table)*.65)), color=COLORS + ["#64748b"])
            ax.set(title=f"SR — {metric.upper()} theo task/vùng", xlabel=f"{metric.upper()} (%)", ylabel="")
            ax.set_xlim(left=0)
            finish_plot(f"sr_{metric}")
else:
    notice("Không có dòng SR thành công để tính WER/CER.")


# %% [markdown]
# ## 8. Baseline bài báo và đối chiếu
# Bảng 5: thời lượng giờ đã làm tròn trong bài báo. Bảng 6: F1-macro; bảng 7: WER, cả direct và fine-tuned. Giá trị gốc là tỷ lệ 0–1; `delta_pp = 100 × (thực nghiệm − paper)`. F1 tăng là tốt; WER giảm là tốt.
#
# Đối chiếu chỉ ghép **đúng model + task + stage**. Nếu thiếu backbone DI thì để NA, không gán wav2vec2-base-vi chỉ từ cấu hình notebook. Paper đánh giá full test, thực nghiệm hiện tại đánh giá sample và huấn luyện ít giờ hơn. Không có kiểm định ý nghĩa thống kê ở đây.
#

# %% [code]
PAPER_URL = "https://aclanthology.org/2024.emnlp-main.426/"
paper_dataset = pd.DataFrame([
    ("train",15023,81.43), ("valid",1900,10.26), ("test",2026,10.87)
], columns=["split","paper_n_audio","paper_duration_hours"])
if not full.empty:
    paper_dataset_check = summarize(full, ["split"]).merge(paper_dataset, on="split", how="outer")
    paper_dataset_check["count_delta"] = paper_dataset_check["n_audio"] - paper_dataset_check["paper_n_audio"]
    paper_dataset_check["hours_delta"] = paper_dataset_check["duration_hours"] - paper_dataset_check["paper_duration_hours"]
    save_table("full_dataset_vs_paper_table5", paper_dataset_check)
    if SHOW_TABLES:
        display(paper_dataset_check)
    else:
        print('Dataset vs paper:')
        for row in paper_dataset_check.to_dict('records'):
            print(f"  {row['split']}: n={row.get('n_audio', float('nan')):.0f} vs {row['paper_n_audio']:.0f}, hours={row.get('duration_hours', float('nan')):.3f} vs {row['paper_duration_hours']:.2f}")

di_baselines = {
 "wav2vec2-base-vi": [.9102,.4322,.5863,.3560,.3522],
 "wav2vec2-large-vi": [.9147,.4229,.5981,.3528,.3570],
 "wav2vec2-xls-r-300m": [.8901,.3216,.3282,.2787,.3728],
 "wav2vec2-large-xlsr-53": [.8736,.2830,.1990,.2440,.3255],
 "whisper-base": [.8559,.4336,.5854,.3383,.3976],
 "PhoWhisper-base": [.8697,.4470,.6251,.3257,.4107],
}
sr_direct = {
 "wav2vec2-base-vietnamese": [.2032,.2728,.2248,.2307],
 "wav2vec2-base-vietnamese-160h": [.2750,.3812,.3093,.3174],
 "wav2vec2-base-vietnamese-250h": [.1498,.2097,.1724,.1747],
 "wav2vec2-base-vi-vlsp2020": [.1364,.1926,.1481,.1568],
 "whisper-base": [.2637,.3991,.2946,.3138],
 "PhoWhisper-base": [.1496,.2415,.1787,.1861],
}
sr_finetuned = {
 "wav2vec2-base-vietnamese": [.1464,.2027,.1772,.1580],
 "wav2vec2-base-vietnamese-160h": [.1670,.2456,.2051,.1749],
 "wav2vec2-base-vietnamese-250h": [.1229,.1715,.1526,.1356],
 "wav2vec2-base-vi-vlsp2020": [.1217,.1719,.1508,.1224],
 "whisper-base": [.2005,.2789,.2089,.1993],
 "PhoWhisper-base": [.1320,.1826,.1354,.1630],
}
paper_rows = []
for model, values in di_baselines.items():
    for task, value in zip(DI_TASKS, values):
        paper_rows.append(dict(task=task, model=model, stage="finetuned", metric="f1_macro",
                               paper_value=value, source_table=6, source_url=PAPER_URL))
for stage, models in [("direct", sr_direct), ("finetuned", sr_finetuned)]:
    for model, values in models.items():
        for task, value in zip(["SR_North","SR_Central","SR_South","SR_VN_63"], values):
            paper_rows.append(dict(task=task, model=model, stage=stage, metric="wer",
                                   paper_value=value, source_table=7, source_url=PAPER_URL))
paper = save_table("paper_baselines", pd.DataFrame(paper_rows))
if SHOW_TABLES:
    display(paper.pivot_table(index=["metric","stage","model"], columns="task", values="paper_value"))
else:
    print('Paper baseline đã lưu tại:', REPORT_DIR / 'paper_baselines.csv')

local_metrics = []
if not di_summary.empty:
    for row in di_summary.to_dict("records"):
        local_metrics.append(dict(row, metric="f1_macro", local_value=row["f1_macro"],
                                  protocol_note="Sample test; DI finetune trên sample; cần xác minh chunking <=10s."))
if not sr_summary.empty:
    for row in sr_summary.to_dict("records"):
        eligible = not (row["stage"] == "finetuned" and row["scope"] == "regional_slice")
        local_metrics.append(dict(row, metric="wer", local_value=row["wer"],
                                  comparison_eligible=eligible,
                                  protocol_note="Sample test; cùng chuẩn hóa text chưa được kiểm chứng hoàn toàn với paper."
                                  if eligible else "Slice của checkpoint toàn quốc, không phải model fine-tune riêng vùng."))
comparison = pd.DataFrame()
if local_metrics:
    comparison = pd.DataFrame(local_metrics).merge(paper, on=["task","model","stage","metric"], how="left", validate="many_to_one")
    comparison["comparison_eligible"] = comparison["comparison_eligible"].fillna(True) if "comparison_eligible" in comparison else True
    for i, row in comparison.iterrows():
        if row["metric"] == "f1_macro" and not row.get("label_universe_complete", False):
            comparison.loc[i,"comparison_eligible"] = False
    comparison["matched_baseline"] = comparison["paper_value"].notna()
    comparison["delta_pp"] = np.where(comparison["comparison_eligible"], 100*(comparison["local_value"]-comparison["paper_value"]), np.nan)
    comparison["local_pct"] = comparison["local_value"]*100
    comparison["paper_pct"] = comparison["paper_value"]*100
    save_table("comparison_with_paper", comparison)
    if SHOW_TABLES:
        display(comparison[["task","model","stage","metric","n_samples","local_pct","paper_pct","delta_pp",
                            "matched_baseline","comparison_eligible","protocol_note"]])
    else:
        print('\nSo sánh số liệu với paper (đơn vị %):')
        for row in comparison.to_dict('records'):
            local = row['local_pct']
            paper_value = row['paper_pct']
            delta = row['delta_pp']
            print(f"  {row['model']} | {row['stage']} | {row['task']} | {row['metric']}: local={local:.2f}, paper={paper_value:.2f}, delta_pp={delta:.2f} | eligible={row['comparison_eligible']}")
    measured_keys = comparison[["task","model","stage","metric"]].drop_duplicates()
    missing = paper.merge(measured_keys.assign(has_result=True), on=["task","model","stage","metric"], how="left")
    save_table("paper_tasks_without_local_result", missing[missing["has_result"].isna()].drop(columns="has_result"))
    for metric in ["f1_macro", "wer"]:
        sub = comparison[comparison["metric"].eq(metric) & comparison["paper_value"].notna() & comparison["comparison_eligible"]]
        if sub.empty:
            continue
        y = np.arange(len(sub))
        fig, ax = plt.subplots(figsize=(12, max(4, len(sub)*.45)))
        ax.barh(y-.18, sub["paper_pct"], .36, label="Paper — full test", color=COLORS[0])
        ax.barh(y+.18, sub["local_pct"], .36, label="Thực nghiệm — sample test", color=COLORS[1])
        ax.set_yticks(y, sub["task"]+" | "+sub["model"]+" | "+sub["stage"])
        ax.set(xlabel=f"{metric} (%)", title=f"Đối chiếu mô tả — {metric}; khác quy mô dữ liệu")
        ax.set_xlim(left=0)
        ax.invert_yaxis()
        ax.legend()
        finish_plot(f"paper_comparison_{metric}")


# %% [markdown]
# ## 9. Tổng kết và xuất kết quả
# Báo cáo gồm CSV chi tiết, PNG biểu đồ, `analysis_manifest.json` ghi cấu hình/nguồn và `summary.md`. Kết luận được sinh từ dữ liệu vừa đọc; không suy diễn task chưa chạy. Để so sánh backbone DI, cần checkpoint config hoặc xác nhận thủ công trong cell cấu hình.
#

# %% [code]
lines = ["# Tổng hợp thực nghiệm ViMD", ""]
for row in total_summary.to_dict("records"):
    lines.append(f"- {row['dataset']}: {row['n_audio']:,} audio, {row['duration_hours']:.3f} giờ đo được; "
                 f"thiếu duration {row['n_duration_missing']}.")
if not di_summary.empty:
    for row in di_summary.to_dict("records"):
        lines.append(f"- {row['task']} ({row['model']}): F1-macro {100*row['f1_macro']:.2f}%, "
                     f"accuracy {100*row['accuracy']:.2f}%; {row['n_samples']} mẫu, "
                     f"{row['n_labels_observed']}/{row['n_expected_labels']} lớp xuất hiện.")
if not sr_summary.empty:
    for row in sr_summary[sr_summary["task"].eq("SR_VN_63")].to_dict("records"):
        lines.append(f"- SR toàn quốc ({row['model']}, {row['stage']}): WER {100*row['wer']:.2f}%, "
                     f"CER {100*row['cer']:.2f}%; {row['n_samples']} mẫu.")
lines += ["", "## Giới hạn đối chiếu",
          "- Paper dùng full dataset; đây là kết quả sample, không phải tái lập nguyên trạng.",
          "- DI model='DI' cần di_config.json hoặc DI_MODEL_OVERRIDES để xác định backbone.",
          "- Paper dùng audio DI chia đoạn <=10s. CSV kết quả không đủ xác minh quy trình train/eval.",
          "- Chênh lệch WER còn phụ thuộc chuẩn hóa, decoder LM/greedy và lựa chọn checkpoint.",
          "- Bảng 7 không có CER. Sample nhỏ không đại diện chắc chắn cho từng tỉnh.",
          "- Các lỗi inference bị loại khỏi metric nhưng được giữ trong bảng coverage/errors."]
lines += [f"- {x}" for x in dict.fromkeys(issues)]
summary_text = "\n".join(lines)
(REPORT_DIR / "summary.md").write_text(summary_text, encoding="utf-8")
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "experiment_dir": str(EXPERIMENT_DIR), "evaluation_split": EVALUATION_SPLIT,
    "sr_stage_default": SR_STAGE, "di_model_overrides": DI_MODEL_OVERRIDES,
    "normalize_punctuation": NORMALIZE_PUNCTUATION, "cer_remove_spaces": CER_REMOVE_SPACES,
    "sources": list({s["path"]: s for s in source_files}.values()),
    "paper_url": PAPER_URL, "paper_tables": [5,6,7],
    "software_versions": {p: importlib.metadata.version(p) for p in ["pandas","numpy","matplotlib","pyarrow","soundfile","rapidfuzz"]},
    "tables": exported_tables, "issues": list(dict.fromkeys(issues)),
    "metric_definitions": {"wer": "sum(word_edits)/sum(reference_words)",
                           "cer": "sum(char_edits)/sum(reference_chars)",
                           "di": "macro F1 over fixed task labels; zero_division=0"},
}
(REPORT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(summary_text)
print("Đã xuất báo cáo:", REPORT_DIR)
