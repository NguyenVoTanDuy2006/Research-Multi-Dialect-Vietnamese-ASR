# %% [markdown]
# # Tính metrics ngữ nghĩa và lỗi từ prediction wide
#
# Đọc prediction CSV dạng wide (một hàng mỗi sample, một cột cho mỗi model),
# tính WER/CER cùng SemSim-E5/BERTScore, rồi xuất kết quả theo mẫu và tổng hợp.
# Chạy tuần tự từ trên xuống. SemSim-E5 và BERTScore cần tải model Hugging Face
# ở lần chạy đầu; có thể đổi `DEVICE` thành `"cpu"` nếu không dùng được GPU.
#
# Kaggle: notebook clone repo `source_code`; attach dataset chứa
# `predictions_wide.csv` hoặc điền đường dẫn trong cell cấu hình. Kết quả mặc định
# ghi vào `/kaggle/working/outputs/metrics_predictions`.

# %% [markdown]
# ## Clone source và cài thư viện

# %%
import os
import shutil
import subprocess
import sys
from pathlib import Path


def is_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src/mdv_asr_module").is_dir()


def find_project_root(*, script_path: str | Path | None = None,
                      cwd: str | Path | None = None,
                      configured_root: str | Path | None = None,
                      kaggle_working_dir: str | Path | None = None) -> Path:
    """Find the repo from this script, current directory, or Kaggle clone location."""
    starts = []
    if configured_root:
        configured = Path(configured_root).expanduser().resolve()
        if not is_project_root(configured):
            raise FileNotFoundError(f"MDV_ASR_REPO_DIR is not a source checkout: {configured}")
        return configured
    if script_path:
        script = Path(script_path).expanduser().resolve()
        starts.append(script if script.is_dir() else script.parent)
    if cwd:
        current = Path(cwd).expanduser().resolve()
        starts.extend((current, current / "mdv_asr_source"))
    if kaggle_working_dir:
        working = Path(kaggle_working_dir).expanduser().resolve()
        starts.extend((working / "mdv_asr_source", working))

    seen = set()
    for start in starts:
        for candidate in (start, *start.parents):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if is_project_root(candidate):
                return candidate
    raise FileNotFoundError(
        "Không tìm thấy repo source. Clone repo vào `/kaggle/working/mdv_asr_source`, "
        "chạy script từ repo, hoặc đặt biến MDV_ASR_REPO_DIR."
    )


IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or (
    Path("/kaggle/input").is_dir() and Path("/kaggle/working").is_dir()
)
REPO_URL = "https://github.com/NguyenVoTanDuy2006/Research-Multi-Dialect-Vietnamese-ASR.git"
REPO_BRANCH = "source_code"
KAGGLE_WORKING_DIR = Path("/kaggle/working")

if IS_KAGGLE:
    REPO_DIR = KAGGLE_WORKING_DIR / "mdv_asr_source"
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
        print("Đã xóa source clone cũ:", REPO_DIR)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH,
         REPO_URL, str(REPO_DIR)],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-r",
         str(REPO_DIR / "requirements-kaggle.txt")],
        check=True,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(REPO_DIR)],
        check=True,
    )
else:
    REPO_DIR = find_project_root(
        script_path=globals().get("__file__"),
        cwd=Path.cwd(),
        configured_root=os.environ.get("MDV_ASR_REPO_DIR"),
    )

PROJECT_ROOT = REPO_DIR
SOURCE_DIR = REPO_DIR / "src"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
print("Source:", REPO_DIR)

# %% [markdown]
# ## Nạp thư viện

# %%
import gc
import importlib.util
import json
import time
from datetime import datetime, timezone

import pandas as pd
import torch

from mdv_asr_module.core.artifacts import hash_file
from mdv_asr_module.evaluation.error_metrics import CERMetric, WERMetric
from mdv_asr_module.evaluation.similarity_metrics import BERTScoreMetric, SemSimE5

# %% [markdown]
# ## Cấu hình đầu vào và metrics
#
# Đặt `PREDICTION_PATH_OVERRIDE` thành đường dẫn CSV trong Kaggle Input nếu cần.
# Để None, notebook dùng file mặc định trong repo hoặc tự tìm đúng một file trong
# `/kaggle/input`. Các tùy chọn metrics và output cũng có thể chỉnh tại đây.

# %%
PREDICTION_PATH_OVERRIDE = None  # Ví dụ: "/kaggle/input/my-predictions/predictions_wide.csv"
OUTPUT_DIR_OVERRIDE = None  # Mặc định: /kaggle/working/outputs/metrics_predictions trên Kaggle

# Tên các cột model trong prediction CSV dạng wide.
MODEL_COLUMNS = {
    "base": "base",
    "ada-lora_small": "ada-lora_small",
    "ada-lora_full": "ada-lora_full",
    "pissa_small": "pissa_small",
    "pissa_full": "pissa_full",
}

# Giá trị None tự chọn CUDA nếu có, nếu không thì CPU.
DEVICE = None
SEMANTIC_BATCH_SIZE = 8
E5_MAX_LENGTH = 512
BERTSCORE_MAX_LENGTH = 512
LONG_TEXT_POLICY = "chunk"  # Giữ cả transcript dài, không cắt mất phần cuối.
INSTALL_DEPENDENCIES = IS_KAGGLE

# %% [markdown]
# ## Resolve đường dẫn và cài dependency còn thiếu

# %%


def resolve_prediction_path(project_root: str | Path, *,
                            configured_path: str | Path | None = None,
                            kaggle_input_root: str | Path | None = None) -> Path:
    """Use an explicit path, the repo's default prediction file, or one Kaggle input."""
    project_root = Path(project_root).expanduser().resolve()
    if configured_path:
        candidate = Path(configured_path).expanduser()
        if not candidate.is_absolute():
            candidate = project_root / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Configured prediction CSV does not exist: {candidate}")
        return candidate.resolve()

    repo_prediction = (
        project_root / "outputs/benchmark_models/qwen_vs_finetuning/predictions_wide.csv"
    )
    if repo_prediction.is_file():
        return repo_prediction.resolve()

    if kaggle_input_root:
        input_root = Path(kaggle_input_root).expanduser()
        if input_root.is_dir():
            matches = sorted(path.resolve() for path in input_root.rglob("predictions_wide.csv")
                             if path.is_file())
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError(
                    "Found multiple predictions_wide.csv files under Kaggle Input: "
                    f"{matches}. Set PREDICTION_PATH_OVERRIDE to select one."
                )
    raise FileNotFoundError(
        "Không tìm thấy predictions_wide.csv trong repo hoặc Kaggle Input. "
        "Attach file này làm Kaggle Input, đặt PREDICTION_PATH_OVERRIDE, hoặc "
        "lưu file tại outputs/benchmark_models/qwen_vs_finetuning/ trong repo."
    )


def resolve_metrics_output_dir(prediction_path: str | Path, *, is_kaggle: bool,
                               configured_dir: str | Path | None = None,
                               project_root: str | Path | None = None,
                               kaggle_working_dir: str | Path | None = None) -> Path:
    """Keep Kaggle artifacts on writable storage; keep local output beside input."""
    prediction_path = Path(prediction_path).expanduser().resolve()
    if configured_dir:
        candidate = Path(configured_dir).expanduser()
        if not candidate.is_absolute():
            base = (Path(kaggle_working_dir).expanduser() if is_kaggle
                    else Path(project_root or prediction_path.parent).expanduser())
            candidate = base / candidate
    elif is_kaggle:
        base = Path(kaggle_working_dir or "/kaggle/working").expanduser()
        candidate = base / "outputs/metrics_predictions"
    else:
        candidate = prediction_path.parent / "metrics"
    return candidate.resolve()


PREDICTION_PATH = resolve_prediction_path(
    PROJECT_ROOT,
    configured_path=PREDICTION_PATH_OVERRIDE,
    kaggle_input_root=Path("/kaggle/input") if IS_KAGGLE else None,
)
OUTPUT_DIR = resolve_metrics_output_dir(
    PREDICTION_PATH,
    is_kaggle=IS_KAGGLE,
    configured_dir=OUTPUT_DIR_OVERRIDE,
    project_root=PROJECT_ROOT,
    kaggle_working_dir=KAGGLE_WORKING_DIR,
)

# %% [markdown]
# ## Cài dependency ngữ nghĩa còn thiếu
#
# Source requirements cài `transformers` và `pandas`; cell này chỉ cài thêm
# BERTScore nếu môi trường Kaggle chưa có. Cần bật Internet cho lần cài đầu.

# %%
if INSTALL_DEPENDENCIES:
    packages = []
    if importlib.util.find_spec("transformers") is None:
        packages.append("transformers>=5.17")
    if importlib.util.find_spec("bert_score") is None:
        packages.append("bert-score==0.3.13")
    if packages:
        print("Installing missing metric dependencies:", packages, flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)

# %% [markdown]
# ## Kiểm tra cấu hình

# %%
if DEVICE is None:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Project: {PROJECT_ROOT}")
print(f"Input:   {PREDICTION_PATH}")
print(f"Output:  {OUTPUT_DIR}")
print(f"Device:  {DEVICE}")

# %% [markdown]
# ## Đọc và kiểm tra dữ liệu

# %%
if not PREDICTION_PATH.is_file():
    raise FileNotFoundError(f"Không tìm thấy prediction CSV: {PREDICTION_PATH}")

required_columns = {
    "sample_id", "region", "province_name", "speaker_gender", "reference",
    *MODEL_COLUMNS.values(),
}
wide = pd.read_csv(PREDICTION_PATH, dtype=str, keep_default_na=False)
missing_columns = sorted(required_columns - set(wide.columns))
if missing_columns:
    raise ValueError(f"Prediction CSV thiếu các cột: {missing_columns}")
if wide.empty:
    raise ValueError("Prediction CSV không có sample nào.")
if wide["sample_id"].str.strip().eq("").any():
    raise ValueError("Có sample_id rỗng.")
if wide["sample_id"].duplicated().any():
    duplicates = wide.loc[wide["sample_id"].duplicated(), "sample_id"].head(10).tolist()
    raise ValueError(f"sample_id bị trùng, ví dụ: {duplicates}")
if wide["reference"].str.strip().eq("").any():
    bad_ids = wide.loc[wide["reference"].str.strip().eq(""), "sample_id"].head(10).tolist()
    raise ValueError(f"Reference rỗng, ví dụ sample_id: {bad_ids}")

metadata_columns = ["region", "province_name", "speaker_gender"]
for column in metadata_columns:
    wide[column] = wide[column].str.strip().replace("", "Unknown")
wide["reference"] = wide["reference"].str.strip()
for column in MODEL_COLUMNS.values():
    wide[column] = wide[column].str.strip()

long = wide.melt(
    id_vars=["sample_id", *metadata_columns, "reference"],
    value_vars=list(MODEL_COLUMNS.values()),
    var_name="model",
    value_name="prediction",
)
long["model"] = long["model"].map({value: key for key, value in MODEL_COLUMNS.items()})
long["prediction"] = long["prediction"].fillna("").astype(str)
print(f"Loaded {len(wide):,} samples, {len(MODEL_COLUMNS)} models, {len(long):,} predictions.")
print("Empty predictions:", int(long["prediction"].str.strip().eq("").sum()))

# %% [markdown]
# ## Tính WER và CER
#
# WER/CER được tính trên từng mẫu; phần tổng hợp bên dưới dùng corpus-level
# (tổng lỗi chia tổng đơn vị tham chiếu), không lấy trung bình các rate để thay thế.

# %%
wer_metric = WERMetric()
cer_metric = CERMetric()
wer_errors, reference_words = [], []
cer_errors, reference_chars = [], []
wer_sentence, cer_sentence = [], []

for row in long.itertuples(index=False):
    wer = wer_metric.score(row.reference, row.prediction)
    cer = cer_metric.score(row.reference, row.prediction)
    wer_errors.append(wer.errors)
    reference_words.append(wer.reference_units)
    wer_sentence.append(wer.rate)
    cer_errors.append(cer.errors)
    reference_chars.append(cer.reference_units)
    cer_sentence.append(cer.rate)

long["wer_errors"] = wer_errors
long["reference_words"] = reference_words
long["WER_sentence"] = wer_sentence
long["cer_errors"] = cer_errors
long["reference_chars"] = reference_chars
long["CER_sentence"] = cer_sentence
print("WER/CER calculated for all model/sample rows.")

# %% [markdown]
# ## Tính SemSim-E5 và BERTScore
#
# Các cặp `(reference, prediction)` giống hệt nhau chỉ được encode một lần,
# sau đó ánh xạ ngược về các model/sample tương ứng. Model được chạy lần lượt
# để không giữ đồng thời hai model ngữ nghĩa lớn trong VRAM.

# %%
semantic_pairs = (
    long.loc[long["prediction"].str.strip().ne(""), ["reference", "prediction"]]
    .drop_duplicates(ignore_index=True)
)
pair_keys = list(zip(semantic_pairs["reference"], semantic_pairs["prediction"]))
pair_to_index = {key: index for index, key in enumerate(pair_keys)}
long["_pair_index"] = [
    pair_to_index.get((reference, prediction), -1)
    for reference, prediction in zip(long["reference"], long["prediction"])
]
print(f"Semantic scoring will encode {len(semantic_pairs):,} unique non-empty pairs.")
if not semantic_pairs.empty:
    missing_semantic_packages = [
        package for package in ("transformers", "bert_score")
        if importlib.util.find_spec(package) is None
    ]
    if missing_semantic_packages:
        raise ImportError(
            "Thiếu dependency cho metrics ngữ nghĩa: "
            f"{missing_semantic_packages}. Cài bằng `pip install 'transformers>=5.17' "
            "'bert-score==0.3.13' 'pandas>=2.0'`, hoặc đặt "
            "INSTALL_DEPENDENCIES = True rồi chạy lại."
        )


def score_unique_pairs(metric, label: str):
    results = []
    references = semantic_pairs["reference"].tolist()
    predictions = semantic_pairs["prediction"].tolist()
    started = time.time()
    for start in range(0, len(references), SEMANTIC_BATCH_SIZE):
        end = min(start + SEMANTIC_BATCH_SIZE, len(references))
        results.extend(metric.score_batch(references[start:end], predictions[start:end]))
        print(f"{label}: {end:,}/{len(references):,} unique pairs", flush=True)
    elapsed = time.time() - started
    return results, elapsed


def smoke_test_e5_chunking(metric):
    """Exercise short and chunked E5 paths on real prediction pairs before full scoring."""
    if semantic_pairs.empty:
        print("SemSim-E5 smoke test skipped: there are no non-empty prediction pairs.", flush=True)
        return
    if not hasattr(metric, "_tokenizer") or not hasattr(metric, "_effective_max_length"):
        # Keeps the notebook's lightweight end-to-end test double independent of HF internals.
        smoke_results = metric.score_batch(
            semantic_pairs["reference"].iloc[:1].tolist(),
            semantic_pairs["prediction"].iloc[:1].tolist(),
        )
        if len(smoke_results) != 1 or not -1.0 <= float(smoke_results[0].similarity) <= 1.0:
            raise RuntimeError("SemSim-E5 basic smoke test returned an invalid score.")
        print("SemSim-E5 smoke test passed (test double; chunk path not available).", flush=True)
        return

    metric._load()
    tokenizer = metric._tokenizer
    token_limit = metric._effective_max_length()
    short_pair = None
    chunked_pair = None
    mixed_pair = None
    pair_batch_size = 128

    def measure_pair_batch(start, end):
        nonlocal short_pair, chunked_pair, mixed_pair
        end = min(end, len(semantic_pairs))
        pair_batch = semantic_pairs.iloc[start:end]
        texts = [
            f"query: {text}"
            for row in pair_batch.itertuples(index=False)
            for text in (row.reference, row.prediction)
        ]
        encoded = tokenizer(
            texts, padding=False, truncation=False, add_special_tokens=True, verbose=False
        )["input_ids"]
        lengths = [len(ids) for ids in encoded]
        candidates = []
        for offset, row in enumerate(pair_batch.itertuples(index=False)):
            index = start + offset
            ref_length, pred_length = lengths[offset * 2:offset * 2 + 2]
            ref_long, pred_long = ref_length > token_limit, pred_length > token_limit
            candidate = (index, row.reference, row.prediction, ref_length, pred_length)
            candidates.append(candidate)
            if not ref_long and not pred_long and short_pair is None:
                short_pair = candidate
            if ref_long or pred_long:
                if chunked_pair is None:
                    chunked_pair = candidate
                if ref_long != pred_long and mixed_pair is None:
                    mixed_pair = candidate
        return candidates

    selected = None
    # The Kaggle traceback showed 6,872/8,508 completed with batch size 8;
    # isolate the next exact batch first, then fall back if this input differs.
    failed_batch_start = 6_872
    if failed_batch_start < len(semantic_pairs):
        failed_batch = measure_pair_batch(
            failed_batch_start,
            min(failed_batch_start + SEMANTIC_BATCH_SIZE, len(semantic_pairs)),
        )
        has_mixed_pair = any(
            (pair[3] > token_limit) != (pair[4] > token_limit)
            for pair in failed_batch
        )
        if has_mixed_pair:
            selected = failed_batch
            label = f"exact batch after traceback progress {failed_batch_start:,}"

    if selected is None:
        for start in range(0, len(semantic_pairs), pair_batch_size):
            measure_pair_batch(start, start + pair_batch_size)
            if mixed_pair is not None or (short_pair is not None and chunked_pair is not None):
                break
        if mixed_pair is not None:
            selected = [mixed_pair]
            label = "one real pair with one short and one chunked transcript"
        elif short_pair is not None and chunked_pair is not None:
            selected = [short_pair, chunked_pair]
            label = "one short-only and one chunked real pair"
        elif chunked_pair is not None:
            selected = [chunked_pair]
            label = "one chunked real pair (no short-only pair found)"
        elif short_pair is not None:
            selected = [short_pair]
            label = "one short real pair (no overlength transcript exists in this input)"
        else:
            print("SemSim-E5 smoke test skipped: there are no non-empty prediction pairs.", flush=True)
            return

    old_batch_size = metric.batch_size
    metric.batch_size = max(old_batch_size, len(selected))
    try:
        results = metric.score_batch(
            [pair[1] for pair in selected], [pair[2] for pair in selected]
        )
    finally:
        metric.batch_size = old_batch_size

    if len(results) != len(selected):
        raise RuntimeError(
            f"SemSim-E5 smoke returned {len(results)} results for {len(selected)} pairs."
        )
    if any(not (-1.0 <= float(result.similarity) <= 1.0) for result in results):
        raise RuntimeError("SemSim-E5 smoke test returned a non-finite/invalid similarity.")
    if chunked_pair is not None and not any(result.chunked for result in results):
        raise RuntimeError("SemSim-E5 smoke did not mark its overlength sample as chunked.")

    print(
        "SemSim-E5 smoke test passed before full scoring:", label,
        "| sample lengths:", [(pair[3], pair[4]) for pair in selected],
        "| scores:", [round(float(result.similarity), 5) for result in results],
        flush=True,
    )


e5_metric = SemSimE5(
    device=DEVICE,
    batch_size=SEMANTIC_BATCH_SIZE,
    max_length=E5_MAX_LENGTH,
    long_text_policy=LONG_TEXT_POLICY,
)
smoke_test_e5_chunking(e5_metric)
e5_results, e5_seconds = score_unique_pairs(e5_metric, "SemSim-E5")
e5_metadata = e5_metric.metadata()
e5_by_pair = {
    index: result for index, result in enumerate(e5_results)
}
long["SemSim_E5"] = [
    e5_by_pair[index].similarity if index >= 0 else 0.0
    for index in long["_pair_index"]
]
long["SemDist_E5"] = 1.0 - long["SemSim_E5"]
long["SemSim_E5_chunked"] = [
    e5_by_pair[index].chunked if index >= 0 else False
    for index in long["_pair_index"]
]
del e5_results, e5_by_pair
if getattr(e5_metric, "_encoder", None) is not None:
    del e5_metric._encoder
if getattr(e5_metric, "_tokenizer", None) is not None:
    del e5_metric._tokenizer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

bert_metric = BERTScoreMetric(
    device=DEVICE,
    batch_size=SEMANTIC_BATCH_SIZE,
    max_length=BERTSCORE_MAX_LENGTH,
    long_text_policy=LONG_TEXT_POLICY,
)
bert_results, bert_seconds = score_unique_pairs(bert_metric, "BERTScore")
bert_metadata = bert_metric.metadata()
bert_by_pair = {
    index: result for index, result in enumerate(bert_results)
}
long["BERTScore_precision"] = [
    bert_by_pair[index].precision if index >= 0 else 0.0
    for index in long["_pair_index"]
]
long["BERTScore_recall"] = [
    bert_by_pair[index].recall if index >= 0 else 0.0
    for index in long["_pair_index"]
]
long["BERTScore_F1"] = [
    bert_by_pair[index].f1 if index >= 0 else 0.0
    for index in long["_pair_index"]
]
long["BERTScore_chunked"] = [
    bert_by_pair[index].chunked if index >= 0 else False
    for index in long["_pair_index"]
]
del bert_results, bert_by_pair
if getattr(bert_metric, "_scorer", None) is not None:
    del bert_metric._scorer
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
long.drop(columns=["_pair_index"], inplace=True)
print(f"Semantic metrics finished: E5 {e5_seconds / 60:.1f} min; "
      f"BERTScore {bert_seconds / 60:.1f} min.")

# %% [markdown]
# ## Tổng hợp và lưu artifacts
#
# Tạo một bảng tổng thể và ba bảng phân nhóm theo miền, tỉnh, giới tính. WER/CER
# là corpus-level; các cột `*_sentence_*` và similarity là phân phối theo mẫu.

# %%
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def summarize_group(group: pd.DataFrame, *, level: str, group_name: str) -> dict:
    words = int(group["reference_words"].sum())
    chars = int(group["reference_chars"].sum())
    result = {
        "group_level": level,
        "group": group_name,
        "model": group["model"].iloc[0],
        "n_samples": int(len(group)),
        "wer_errors": int(group["wer_errors"].sum()),
        "reference_words": words,
        "WER": float(group["wer_errors"].sum() / words) if words else None,
        "cer_errors": int(group["cer_errors"].sum()),
        "reference_chars": chars,
        "CER": float(group["cer_errors"].sum() / chars) if chars else None,
    }
    distribution_columns = {
        "WER_sentence": "WER_sentence",
        "CER_sentence": "CER_sentence",
        "SemSim_E5": "SemSim_E5",
        "SemDist_E5": "SemDist_E5",
        "BERTScore_precision": "BERTScore_precision",
        "BERTScore_recall": "BERTScore_recall",
        "BERTScore_F1": "BERTScore_F1",
    }
    for column, output_name in distribution_columns.items():
        values = group[column].astype(float)
        result[f"{output_name}_mean"] = float(values.mean())
        result[f"{output_name}_median"] = float(values.median())
        result[f"{output_name}_p05"] = float(values.quantile(0.05))
    result["SemSim_E5_chunked_samples"] = int(group["SemSim_E5_chunked"].sum())
    result["BERTScore_chunked_samples"] = int(group["BERTScore_chunked"].sum())
    return result


def make_summary(group_columns: list[str] | None = None) -> pd.DataFrame:
    rows = []
    if group_columns is None:
        groups = [("overall", "all", group) for _, group in long.groupby("model", sort=False)]
    else:
        column = group_columns[0]
        groups = [
            (column, str(group_name), group)
            for (_, group_name), group in long.groupby(["model", column], dropna=False, sort=True)
        ]
    for level, group_name, group in groups:
        rows.append(summarize_group(group, level=level, group_name=group_name))
    return pd.DataFrame(rows)


sample_columns = [
    "sample_id", "region", "province_name", "speaker_gender", "model", "reference",
    "prediction", "wer_errors", "reference_words", "WER_sentence", "cer_errors",
    "reference_chars", "CER_sentence", "SemSim_E5", "SemDist_E5",
    "SemSim_E5_chunked", "BERTScore_precision", "BERTScore_recall", "BERTScore_F1",
    "BERTScore_chunked",
]
long[sample_columns].to_csv(OUTPUT_DIR / "metrics_per_sample.csv", index=False, encoding="utf-8-sig")
make_summary().to_csv(OUTPUT_DIR / "metrics_overall.csv", index=False, encoding="utf-8-sig")
make_summary(["region"]).to_csv(OUTPUT_DIR / "metrics_by_region.csv", index=False, encoding="utf-8-sig")
make_summary(["province_name"]).to_csv(OUTPUT_DIR / "metrics_by_province.csv", index=False, encoding="utf-8-sig")
make_summary(["speaker_gender"]).to_csv(OUTPUT_DIR / "metrics_by_gender.csv", index=False, encoding="utf-8-sig")

run_config = {
    "input_path": str(PREDICTION_PATH.resolve()),
    "input_sha256": hash_file(PREDICTION_PATH),
    "input_rows": int(len(wide)),
    "models": MODEL_COLUMNS,
    "semantic_unique_pairs": int(len(semantic_pairs)),
    "device": DEVICE,
    "semantic_batch_size": SEMANTIC_BATCH_SIZE,
    "e5_max_length": E5_MAX_LENGTH,
    "bertscore_max_length": BERTSCORE_MAX_LENGTH,
    "long_text_policy": LONG_TEXT_POLICY,
    "e5_seconds": e5_seconds,
    "bertscore_seconds": bert_seconds,
    "wer": wer_metric.metadata(),
    "cer": cer_metric.metadata(),
    "sem_sim_e5": e5_metadata,
    "bertscore": bert_metadata,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
(OUTPUT_DIR / "metrics_config.json").write_text(
    json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8"
)

print("Wrote metrics to:", OUTPUT_DIR)
print(pd.read_csv(OUTPUT_DIR / "metrics_overall.csv").to_string(index=False))

# %% [markdown]
# ## Dọn source clone khỏi Kaggle Output
#
# Chạy cell cuối sau khi kiểm tra metrics. Chỉ xóa source clone; các file kết quả
# nằm trong `/kaggle/working/outputs/metrics_predictions` được giữ lại.

# %%
if IS_KAGGLE:
    from mdv_asr_module.runtime.cleanup import NotebookOutputCleaner

    removed = NotebookOutputCleaner(working_dir=KAGGLE_WORKING_DIR).remove_source_checkout(
        REPO_DIR, preserve_paths=[OUTPUT_DIR]
    )
    print("Đã dọn source clone:", REPO_DIR if removed else "đã được dọn trước đó")
else:
    print("Bỏ qua cleanup vì runtime không phải Kaggle.")
