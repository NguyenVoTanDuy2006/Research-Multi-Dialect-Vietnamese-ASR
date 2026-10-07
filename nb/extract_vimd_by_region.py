# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Trích xuất dữ liệu ViMD theo vùng miền
#
# Notebook lọc các bản ghi theo vùng được chọn (`North`, `Central` hoặc `South`) từ ViMD trong từng split `train`, `valid`, `test`. Chỉnh `TARGET_REGION` ở cell cấu hình để đổi vùng. Mặc định chỉ xuất metadata; có thể bật `EXPORT_AUDIO` để lưu WAV gốc.
#
# Nguồn dữ liệu chọn trong cell cấu hình: `local_cache`, `kaggle` hoặc `hub`. Notebook không tự tải toàn bộ dataset nếu nguồn local chưa được cấu hình.
#

# %% [markdown]
# ## Phạm vi và đầu ra
#
# Đầu ra được đặt trong thư mục theo vùng đã chọn, gồm metadata CSV/JSON, thống kê theo split và tỉnh, cùng dataset Hugging Face chỉ chứa metadata (nếu bật). Audio gốc chỉ được xuất khi `EXPORT_AUDIO=True`.
#
# `duration_seconds` được tính từ header audio khi `COMPUTE_DURATION=True`; việc này có thể mất thời gian vì phải đọc audio của các bản ghi được chọn.
#

# %% [code]
# Cấu hình — chỉnh cell này trước khi chạy.
from pathlib import Path
import io
import json
import os
import re
import struct
import unicodedata
import wave

import pandas as pd
from IPython.display import Audio as NotebookAudio, display

DATASET_SOURCE = 'local_cache'  # 'local_cache', 'kaggle' hoặc 'hub'
DATASET_ID = 'nguyendv02/ViMD_Dataset'
# Local Arrow cache. Để None để tự tìm ~/.cache/huggingface/datasets/...
LOCAL_CACHE_DIR = (
    "/home/fernir/.cache/huggingface/datasets/"
    "nguyendv02___vi_md_dataset/default/0.0.0/"
    "3a5b30157034e7eadd5c75fae1a820c6f9383398"
)
# Kaggle: thư mục chứa các file .arrow/.parquet đã upload.
DATASET_ROOT = '/kaggle/input/datasets/thanhlamdev/vimd-dataset/0.0.0'
SPLITS = ('train', 'valid', 'test')

# Chọn một trong ba vùng: 'North', 'Central', 'South'.
TARGET_REGION = 'South'
REGION_ALIASES = {
    'North': {'north', 'bac', 'miền bắc', 'mien bac'},
    'Central': {'central', 'trung', 'miền trung', 'mien trung'},
    'South': {'south', 'nam', 'miền nam', 'mien nam'},
}
if TARGET_REGION not in REGION_ALIASES:
    raise ValueError(f"TARGET_REGION phải là một trong {list(REGION_ALIASES)}")
TARGET_REGION_LABEL = {'North': 'miền Bắc', 'Central': 'miền Trung', 'South': 'miền Nam'}[TARGET_REGION]
TARGET_REGION_SLUG = TARGET_REGION.lower()
REGION_VALUES = {value.lower() for value in REGION_ALIASES[TARGET_REGION]}

COMPUTE_DURATION = True
EXPORT_AUDIO = False
EXPORT_FILTERED_METADATA_DATASET = True
MAX_AUDIO_EXPORT = None  # None = xuất tất cả audio thuộc vùng đã chọn
HF_TOKEN = os.environ.get('HF_TOKEN', '') or os.environ.get('HUGGINGFACE_TOKEN', '')

if Path('/kaggle/working').exists():
    OUTPUT_DIR = Path('/kaggle/working') / f'vimd_{TARGET_REGION_SLUG}'
else:
    OUTPUT_DIR = Path.cwd() / 'outputs' / f'vimd_{TARGET_REGION_SLUG}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUTPUT_DIR = OUTPUT_DIR / 'audio'

print('TARGET_REGION:', TARGET_REGION, f'({TARGET_REGION_LABEL})')
print('DATASET_SOURCE:', DATASET_SOURCE)
print('LOCAL_CACHE_DIR:', LOCAL_CACHE_DIR)
print('DATASET_ROOT:', DATASET_ROOT)
print('OUTPUT_DIR:', OUTPUT_DIR)

# %% [markdown]
# ### Cài dependency (chỉ chạy nếu môi trường chưa có)
#
# Nếu Kaggle/local chưa có các package dưới đây, bỏ comment và chạy cell này trước cell import dữ liệu. Sau khi cài package nhị phân, nên restart kernel/session rồi chạy lại từ đầu.

# %% [code]
# %pip install -q datasets pandas pyarrow soundfile "transformers>=5.13.0" accelerate
print('Bỏ comment lệnh %pip ở trên nếu thiếu dataset/audio hoặc muốn chạy Qwen3-ASR.')

# %% [code]
from datasets import Audio, Dataset, load_dataset

META_COLUMNS = [
    'region', 'province_code', 'province_name', 'filename',
    'text', 'speakerID', 'gender', 'audio',
]

def _normalise(value):
    value = unicodedata.normalize('NFC', str(value or '')).strip().lower()
    return ' '.join(value.split())

def _is_target_region(value):
    return _normalise(value) in REGION_VALUES

def _find_local_cache():
    if LOCAL_CACHE_DIR:
        candidate = Path(LOCAL_CACHE_DIR).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f'LOCAL_CACHE_DIR không tồn tại: {candidate}')
    roots = [Path.home() / '.cache' / 'huggingface' / 'datasets', Path('/kaggle/working')]
    candidates = []
    for root in roots:
        if root.exists():
            candidates.extend(root.rglob('dataset_info.json'))
    candidates = [p.parent for p in candidates if list(p.glob('vi_md_dataset-*.arrow'))]
    if not candidates:
        raise FileNotFoundError(
            'Không tìm thấy Arrow cache. Đặt VIMD_CACHE_DIR hoặc đổi DATASET_SOURCE sang kaggle/hub.'
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)

def _split_from_path(path):
    name = path.name.lower()
    for split in SPLITS:
        if name.startswith(f'vi_md_dataset-{split}-') or f'-{split}-' in name:
            return split
    return None

def _iter_local_arrow(root):
    root = Path(root)
    for split in SPLITS:
        files = sorted(root.glob(f'vi_md_dataset-{split}-*.arrow'))
        if not files:
            raise FileNotFoundError(f'Không có shard {split} trong {root}')
        offset = 0
        for path in files:
            part = Dataset.from_file(str(path)).cast_column('audio', Audio(decode=False))
            yield split, path, part, offset
            offset += len(part)

def _iter_kaggle_files(root):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f'DATASET_ROOT không tồn tại: {root}')
    arrow_groups = {split: sorted(p for p in root.rglob('*.arrow') if _split_from_path(p) == split) for split in SPLITS}
    if any(arrow_groups.values()):
        for split in SPLITS:
            offset = 0
            for path in arrow_groups[split]:
                part = Dataset.from_file(str(path)).cast_column('audio', Audio(decode=False))
                yield split, path, part, offset
                offset += len(part)
        return
    parquet_groups = {split: sorted(p for p in root.rglob('*.parquet') if _split_from_path(p) == split) for split in SPLITS}
    if not any(parquet_groups.values()):
        raise FileNotFoundError(f'Không tìm thấy file .arrow hoặc .parquet trong {root}')
    for split in SPLITS:
        if not parquet_groups[split]:
            continue
        part = load_dataset('parquet', data_files={split: [str(p) for p in parquet_groups[split]]}, split=split)
        yield split, root, part.cast_column('audio', Audio(decode=False)), 0

def iter_dataset_parts():
    if DATASET_SOURCE == 'local_cache':
        yield from _iter_local_arrow(_find_local_cache())
    elif DATASET_SOURCE == 'kaggle':
        yield from _iter_kaggle_files(DATASET_ROOT)
    elif DATASET_SOURCE == 'hub':
        hub_dataset = load_dataset(DATASET_ID, token=HF_TOKEN or None)
        for split in SPLITS:
            if split in hub_dataset:
                yield split, None, hub_dataset[split].cast_column('audio', Audio(decode=False)), 0
    else:
        raise ValueError(f'DATASET_SOURCE không hợp lệ: {DATASET_SOURCE}')

parts = list(iter_dataset_parts())
print('Số phần dữ liệu:', len(parts))
print('Splits:', sorted({split for split, _, _, _ in parts}))

# %% [code]
# Tra cứu/phát audio trực tiếp từ dataset nguồn (không phụ thuộc cell lọc vùng).
def _audio_bytes(audio_info):
    if isinstance(audio_info, dict) and audio_info.get('bytes') is not None:
        return audio_info['bytes']
    path = audio_info.get('path') if isinstance(audio_info, dict) else None
    if path and Path(path).exists():
        return Path(path).read_bytes()
    raise FileNotFoundError(f'Không đọc được audio: {audio_info}')

_AUDIO_FILENAME_INDEX = None

def _build_audio_filename_index():
    """Index metadata của toàn bộ parts, không giải mã audio và không cần region_df."""
    index = []
    metadata_columns = ('filename', 'region', 'province_name', 'province_code')
    for split_name, source_path, part, split_offset in parts:
        columns = [column for column in metadata_columns if column in part.column_names]
        if 'filename' not in columns:
            continue
        metadata = part.select_columns(columns)
        for local_index, row in enumerate(metadata):
            global_index = int(split_offset + local_index)
            filename_value = row.get('filename') or f'row_{global_index}.wav'
            index.append({
                'split': split_name,
                'split_row_index': global_index,
                'shard_row_index': local_index,
                'source_shard': str(source_path) if source_path else '',
                'filename': str(filename_value),
                'region': row.get('region'),
                'province_name': row.get('province_name'),
                'province_code': row.get('province_code'),
                '_part': part,
            })
    return index

def _region_alias_match(value, requested_region):
    if requested_region is None:
        return True
    requested = _normalise(requested_region)
    for label, aliases in REGION_ALIASES.items():
        normalized_aliases = {_normalise(label), *(_normalise(alias) for alias in aliases)}
        if requested in normalized_aliases:
            return _normalise(value) in normalized_aliases
    return _normalise(value) == requested

def find_audio_record_by_filename(filename, split=None, region=None):
    """Tìm tên audio trên toàn bộ dataset; split/region chỉ là bộ lọc tùy chọn."""
    global _AUDIO_FILENAME_INDEX
    if _AUDIO_FILENAME_INDEX is None:
        _AUDIO_FILENAME_INDEX = _build_audio_filename_index()

    target = Path(str(filename)).name.casefold()
    candidates = []
    for record in _AUDIO_FILENAME_INDEX:
        if split is not None and record['split'] != split:
            continue
        if not _region_alias_match(record.get('region'), region):
            continue
        stored = Path(record['filename']).name.casefold()
        if stored == target:
            priority = 0
        elif stored.endswith('_' + target) or target.endswith('_' + stored):
            priority = 1
        else:
            continue
        candidates.append((priority, record))

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy audio tên '{filename}'"
            + (f" trong split={split}" if split else "")
            + (f" thuộc region={region}" if region else "") + "."
        )
    best_priority = min(priority for priority, _ in candidates)
    matches = [record for priority, record in candidates if priority == best_priority]
    if len(matches) > 1:
        choices = [f"{r['split']} | {r.get('region')} | {r.get('province_name')} | row={r['split_row_index']}"
                   for r in matches[:10]]
        raise ValueError(
            f"Tên '{filename}' khớp với nhiều audio. Hãy truyền split= hoặc region= để chọn:\n"
            + "\n".join(choices)
        )
    return matches[0]

def play_audio_by_filename(filename, split=None, region=None, autoplay=False):
    """Phát audio trong dataset nguồn; không cần chạy bước lọc hay export audio."""
    record = find_audio_record_by_filename(filename, split=split, region=region)
    part = record['_part']
    audio_bytes = _audio_bytes(part[int(record['shard_row_index'])]['audio'])
    print(f"Phát: {record['filename']} | split={record['split']} | "
          f"region={record.get('region')} | province={record.get('province_name')}")
    display(NotebookAudio(data=audio_bytes, autoplay=autoplay))
    return audio_bytes

# Ví dụ: play_audio_by_filename('81_0302.wav', split='test')


# %% [markdown]
# ## Qwen3-ASR — dự đoán audio theo tên trên toàn dataset
#
# Thứ tự chạy độc lập cho chức năng này: cell cấu hình → cài dependencies nếu cần → nạp dataset nguồn tạo `parts` → cell tra cứu audio → cell load Qwen → cell dự đoán. Không cần chạy cell lọc vùng/tỉnh hoặc tạo `region_df`.
#
# Đặt `QWEN_PREDICT_FILENAME`, ví dụ `81_0302.wav`, ở cell dự đoán. Có thể đặt split để xử lý tên trùng. Token Hugging Face đọc từ biến môi trường `HF_TOKEN` hoặc `HUGGINGFACE_TOKEN`.
#

# %% [code]
import os
import tempfile
import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

QWEN_MODEL_ID = "Qwen/Qwen3-ASR-1.7B-hf"
QWEN_LANGUAGE = "Vietnamese"
QWEN_MAX_NEW_TOKENS = 256
HF_TOKEN = os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGINGFACE_TOKEN", "")
QWEN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
QWEN_DTYPE = (
    torch.bfloat16 if QWEN_DEVICE == "cuda" and torch.cuda.is_bf16_supported()
    else torch.float16 if QWEN_DEVICE == "cuda"
    else torch.float32
)

qwen_processor = AutoProcessor.from_pretrained(
    QWEN_MODEL_ID,
    token=HF_TOKEN or None,
)
qwen_load_kwargs = {"token": HF_TOKEN or None, "dtype": QWEN_DTYPE}
if QWEN_DEVICE == "cuda":
    qwen_load_kwargs["device_map"] = "auto"
qwen_model = AutoModelForMultimodalLM.from_pretrained(
    QWEN_MODEL_ID,
    **qwen_load_kwargs,
)
if QWEN_DEVICE == "cpu":
    qwen_model = qwen_model.to(QWEN_DEVICE)
qwen_model.eval()
print("Loaded:", QWEN_MODEL_ID)
print("Device:", qwen_model.device, "dtype:", qwen_model.dtype)


# %% [code]
def _get_audio_and_reference_by_filename(filename, split=None, region=None):
    record = find_audio_record_by_filename(filename, split=split, region=region)
    part = record["_part"]
    local_index = int(record["shard_row_index"])
    audio_bytes = _audio_bytes(part[local_index]["audio"])
    if "text" not in part.column_names:
        raise KeyError("Dataset nguồn không có cột 'text' để lấy transcript tham chiếu.")
    # Select only text so this lookup does not decode/load the audio column again.
    reference = part.select_columns(["text"])[local_index]["text"]
    return audio_bytes, record, reference

def qwen_predict_by_filename(filename, split=None, region=None, language=QWEN_LANGUAGE):
    """ASR theo tên file; tra trên dataset nguồn, độc lập với region_df/extraction."""
    audio_bytes, record, reference = _get_audio_and_reference_by_filename(
        filename, split=split, region=region
    )
    # Qwen processor nhận đường dẫn audio; temporary audio nằm ở thư mục writable.
    with tempfile.NamedTemporaryFile(suffix=".wav") as temporary_audio:
        temporary_audio.write(audio_bytes)
        temporary_audio.flush()
        inputs = qwen_processor.apply_transcription_request(
            audio=temporary_audio.name,
            language=language,
        )
        inputs = inputs.to(qwen_model.device, qwen_model.dtype)
        with torch.inference_mode():
            output_ids = qwen_model.generate(
                **inputs,
                max_new_tokens=QWEN_MAX_NEW_TOKENS,
                do_sample=False,
            )
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        prediction = qwen_processor.decode(
            generated_ids,
            return_format="transcription_only",
        )
        prediction = prediction[0] if isinstance(prediction, (list, tuple)) else str(prediction)

    print("filename:", record["filename"])
    print("split:", record["split"], "region:", record["region"], "province:", record["province_name"])
    print("reference (đọc trực tiếp từ dataset):", reference or "")
    print("prediction:", prediction)
    return prediction, reference

# Nhập tên file rồi chạy cell này, ví dụ:
QWEN_PREDICT_FILENAME = None  # "81_0302.wav"
QWEN_PREDICT_SPLIT = "test"
QWEN_PREDICT_REGION = None  # None = mọi miền; dùng để phân giải nếu tên bị trùng
if QWEN_PREDICT_FILENAME:
    qwen_prediction, qwen_reference = qwen_predict_by_filename(
        QWEN_PREDICT_FILENAME,
        split=QWEN_PREDICT_SPLIT,
        region=QWEN_PREDICT_REGION,
    )


# %% [markdown]
# ## Lọc vùng được chọn và ghi metadata
#
# Cell lọc này chỉ tạo bảng metadata/audio của vùng đã chọn. Tra cứu, phát audio và dự đoán Qwen không phụ thuộc vào việc chạy cell này.
#
# `split_row_index` là chỉ số trong split gốc; `shard_row_index` là chỉ số trong shard nguồn.
#

# %% [code]
def _audio_duration(audio_info):
    data = _audio_bytes(audio_info)
    try:
        with wave.open(io.BytesIO(data), 'rb') as wav_file:
            return wav_file.getnframes() / wav_file.getframerate()
    except (wave.Error, EOFError, ZeroDivisionError):
        import soundfile as sf
        return float(sf.info(io.BytesIO(data)).duration)

def _safe_filename(name, fallback):
    name = Path(str(name or fallback)).name
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    return name or fallback

records = []
exported_audio_count = 0
for split, source_path, part, split_offset in parts:
    available = [column for column in META_COLUMNS if column in part.column_names]
    if 'region' not in available or 'audio' not in available:
        raise KeyError(f'{split} thiếu region/audio; columns={part.column_names}')
    metadata_part = part.select_columns([column for column in available if column != 'audio'])
    for local_index, row in enumerate(metadata_part):
        if not _is_target_region(row.get('region')):
            continue
        original_index = int(split_offset + local_index)
        audio_info = None
        if COMPUTE_DURATION or EXPORT_AUDIO:
            audio_info = part[local_index]['audio']
        record = {
            'split': split,
            'split_row_index': original_index,
            'shard_row_index': int(local_index),
            'source_shard': str(source_path) if source_path else '',
            'region': row.get('region'),
            'province_code': row.get('province_code'),
            'province_name': row.get('province_name'),
            'filename': row.get('filename') or f'row_{original_index}.wav',
            'speakerID': row.get('speakerID'),
            'gender': row.get('gender'),
            'text': row.get('text'),
            'duration_seconds': None,
            'audio_file': '',
        }
        if COMPUTE_DURATION:
            record['duration_seconds'] = _audio_duration(audio_info)
        if EXPORT_AUDIO and (MAX_AUDIO_EXPORT is None or exported_audio_count < MAX_AUDIO_EXPORT):
            split_dir = AUDIO_OUTPUT_DIR / split
            split_dir.mkdir(parents=True, exist_ok=True)
            output_name = f'{split}_{original_index:06d}_{_safe_filename(record["filename"], f"row_{original_index}.wav")}'
            output_path = split_dir / output_name
            if not output_path.exists():
                output_path.write_bytes(_audio_bytes(audio_info))
            record['audio_file'] = str(output_path.relative_to(OUTPUT_DIR))
            exported_audio_count += 1
        records.append(record)

if not records:
    raise ValueError('Không tìm thấy bản ghi thuộc vùng đã chọn. Kiểm tra region và DATASET_SOURCE.')

region_df = pd.DataFrame(records)
region_df['duration_seconds'] = pd.to_numeric(region_df['duration_seconds'], errors='coerce')
region_df.to_csv(OUTPUT_DIR / f'{TARGET_REGION_SLUG}_metadata.csv', index=False, encoding='utf-8-sig')
region_df.to_json(OUTPUT_DIR / f'{TARGET_REGION_SLUG}_metadata.json', orient='records', force_ascii=False, indent=2)
if EXPORT_FILTERED_METADATA_DATASET:
    Dataset.from_pandas(region_df, preserve_index=False).save_to_disk(str(OUTPUT_DIR / f'{TARGET_REGION_SLUG}_metadata_dataset'))

print('Số bản ghi thuộc vùng đã chọn:', len(region_df))
print('Tổng thời lượng (giờ):', round(region_df['duration_seconds'].sum() / 3600, 3))
print('Audio đã xuất:', exported_audio_count)
print('Output:', OUTPUT_DIR)


# %% [code]
# # Tổng hợp số lượng/thời lượng theo split và tỉnh.
# summary_split = (
#     region_df.groupby('split', dropna=False)
#     .agg(n_audio=('filename', 'size'), duration_seconds=('duration_seconds', 'sum'), n_speakers=('speakerID', 'nunique'))
#     .reset_index()
# )
# summary_split['duration_minutes'] = summary_split['duration_seconds'] / 60
# summary_split['duration_hours'] = summary_split['duration_seconds'] / 3600
# summary_split.to_csv(OUTPUT_DIR / f'{TARGET_REGION_SLUG}_summary_by_split.csv', index=False, encoding='utf-8-sig')

# summary_province = (
#     region_df.groupby(['province_code', 'province_name'], dropna=False)
#     .agg(n_audio=('filename', 'size'), duration_seconds=('duration_seconds', 'sum'), n_speakers=('speakerID', 'nunique'))
#     .reset_index()
#     .sort_values(['n_audio', 'province_name'], ascending=[False, True])
# )
# summary_province['duration_minutes'] = summary_province['duration_seconds'] / 60
# summary_province['duration_hours'] = summary_province['duration_seconds'] / 3600
# summary_province.to_csv(OUTPUT_DIR / f'{TARGET_REGION_SLUG}_summary_by_province.csv', index=False, encoding='utf-8-sig')

# display(summary_split.round(3))
# display(summary_province.round(3))

# %% [code]
# Biểu đồ kiểm tra phân bổ theo tỉnh.
import matplotlib.pyplot as plt

plot_df = summary_province.sort_values('n_audio', ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(15, max(5, len(plot_df) * 0.25)), constrained_layout=True)
axes[0].barh(plot_df['province_name'].astype(str), plot_df['n_audio'])
axes[0].set_title(f'Số audio {TARGET_REGION_LABEL} theo tỉnh')
axes[0].set_xlabel('Số audio')
axes[0].set_ylabel('Tỉnh')
axes[1].barh(plot_df['province_name'].astype(str), plot_df['duration_hours'])
axes[1].set_title(f'Thời lượng {TARGET_REGION_LABEL} theo tỉnh')
axes[1].set_xlabel('Giờ')
axes[1].set_ylabel('Tỉnh')
plt.show()

# %% [code]
# Kiểm tra cuối và manifest để tái lập.
assert len(region_df) > 0
assert region_df['region'].map(_is_target_region).all()
manifest = {
    'dataset_source': DATASET_SOURCE,
    'dataset_id': DATASET_ID,
    'dataset_root': DATASET_ROOT,
    'splits': list(SPLITS),
    'target_region': TARGET_REGION,
    'region_values': sorted(REGION_VALUES),
    'n_audio': int(len(region_df)),
    'duration_seconds': float(region_df['duration_seconds'].sum()),
    'duration_hours': float(region_df['duration_seconds'].sum() / 3600),
    'export_audio': EXPORT_AUDIO,
    'output_dir': str(OUTPUT_DIR),
}
(OUTPUT_DIR / f'{TARGET_REGION_SLUG}_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False, indent=2))
print(f'Đã hoàn tất trích xuất dữ liệu {TARGET_REGION_LABEL}.')
