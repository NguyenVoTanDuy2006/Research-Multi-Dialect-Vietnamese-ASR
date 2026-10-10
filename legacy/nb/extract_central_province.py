# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Khai phá từ vựng địa phương cấp tỉnh — miền Trung
#
# Notebook này tìm các từ có tính đặc trưng cho từng tỉnh miền Trung từ transcript ViMD.
#
# Pipeline:
#
# 1. Thống kê toàn bộ từ, không áp dụng ngưỡng frequency/document/speaker.
# 2. Tính TF-IPF, log-odds ratio có Dirichlet smoothing và PMI.
# 3. Kiểm định chi-square ở cấp token và cấp phiên thoại.
# 4. Hiệu chỉnh nhiều kiểm định bằng Benjamini-Hochberg FDR.
# 5. Xuất toàn bộ thống kê và danh sách ứng viên từ địa phương.
#
# Lưu ý: nếu transcript đã được chuẩn hóa chính tả, kết quả chỉ phản ánh khác biệt từ vựng được ghi lại, chưa chứng minh khác biệt phát âm. Các ứng viên cuối cùng nên được nghe audio và gán nhãn thủ công.
#

# %% [markdown]
# ## Cài đặt
#
# Nếu môi trường thiếu thư viện, bỏ comment dòng dưới rồi chạy cell này trước. Sau khi cài package nhị phân, nên restart kernel và chạy lại từ đầu.
#

# %% [code]
# %pip install -q pandas numpy scipy pyarrow datasets

from pathlib import Path
from collections import Counter, defaultdict
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.stats import chi2_contingency

PROJECT_ROOT = next(
    (p for p in [Path.cwd(), *Path.cwd().parents]
     if (p / "docs").is_dir() and (p / "nb").is_dir()),
    Path.cwd(),
)

INPUT_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "vimd_central_region" / "central_metadata.json",
    PROJECT_ROOT / "nb" / "outputs" / "vimd_central_region" / "central_metadata.json",
    Path("/kaggle/input/dataset-sample/central_metadata.json"),
    Path("/kaggle/input/vimd-central-region/central_metadata.json"),
]
INPUT_METADATA = next((p for p in INPUT_CANDIDATES if p.exists()), None)

if Path("/kaggle/working").exists():
    OUTPUT_DIR = Path("/kaggle/working/central_province_lexicon")
else:
    OUTPUT_DIR = PROJECT_ROOT / "outputs" / "central_province_lexicon"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Không áp dụng ngưỡng frequency/document/speaker; thống kê toàn bộ từ.
ALPHA = 0.5                         # Dirichlet smoothing cho log-odds
P_VALUE_THRESHOLD = 0.05
TOP_K_PER_PROVINCE = 250
STOPWORDS = set()                   # Có thể bổ sung stopword nếu cần.

print("PROJECT_ROOT:", PROJECT_ROOT)
print("INPUT_METADATA:", INPUT_METADATA)
print("OUTPUT_DIR:", OUTPUT_DIR)
if INPUT_METADATA is None:
    raise FileNotFoundError(
        "Không tìm thấy central_metadata.json. "
        "Hãy điền INPUT_METADATA tới file sinh bởi extract_vimd_central_region."
    )


# %% [code]
def read_metadata(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, keep_default_na=False)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("data", payload))
    return pd.DataFrame(payload)

def normalize_text(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFC", str(value)).lower()
    return " ".join(value.split())

def normalize_label(value):
    if pd.isna(value):
        return ""
    return " ".join(unicodedata.normalize("NFC", str(value)).strip().split())

frame = read_metadata(INPUT_METADATA).copy()

# Chấp nhận một số tên cột thường gặp trong metadata.
aliases = {
    "province": "province_name",
    "speaker_id": "speakerID",
    "transcript": "text",
}
for old, new in aliases.items():
    if old in frame.columns and new not in frame.columns:
        frame = frame.rename(columns={old: new})

required = ["text", "province_name"]
missing = [c for c in required if c not in frame.columns]
if missing:
    raise KeyError(f"Metadata thiếu cột bắt buộc: {missing}; hiện có {list(frame.columns)}")

for col in ["region", "split", "filename", "speakerID"]:
    if col not in frame.columns:
        frame[col] = pd.NA

frame["text"] = frame["text"].map(normalize_text)
frame["province_name"] = frame["province_name"].map(normalize_label)
frame["region"] = frame["region"].map(normalize_text)
frame["split"] = frame["split"].map(normalize_text)
frame["filename"] = frame["filename"].fillna("").astype(str)

# Chỉ giữ miền Trung; nếu metadata đã được lọc sẵn thì điều kiện này vẫn an toàn.
REGION_VALUES = {"central", "trung", "mien trung", "miền trung"}
is_central = frame["region"].isin(REGION_VALUES) | frame["region"].eq("")
frame = frame.loc[is_central & frame["province_name"].ne("") & frame["text"].ne("")].copy()

# Speaker thiếu được coi là một speaker riêng theo bản ghi, tránh gộp các bản ghi không xác định.
frame["speaker_key"] = frame["speakerID"].where(
    frame["speakerID"].notna() & frame["speakerID"].astype(str).ne(""),
    "unknown_row_" + frame.index.astype(str),
)

print("Số phiên thoại miền Trung:", len(frame))
print("Số tỉnh:", frame["province_name"].nunique())
print("Các tỉnh:", sorted(frame["province_name"].unique()))


# %% [code]
TOKEN_RE = re.compile(r"[^\W\d_]+", flags=re.UNICODE)

def tokenize_vi(text):
    # Giữ dấu tiếng Việt; loại số, dấu câu và token quá ngắn.
    tokens = [m.group(0) for m in TOKEN_RE.finditer(normalize_text(text))]
    return [w for w in tokens if len(w) >= 2 and w not in STOPWORDS]

frame["tokens"] = frame["text"].map(tokenize_vi)
frame = frame[frame["tokens"].map(bool)].copy()
frame["unique_tokens"] = frame["tokens"].map(set)

provinces = sorted(frame["province_name"].unique())
n_provinces = len(provinces)

token_counts = {}
document_counts = {}
speaker_sets = {}
province_token_totals = {}

for province, group in frame.groupby("province_name", sort=True):
    counts = Counter()
    docs = Counter()
    speakers = defaultdict(set)
    for tokens, unique_tokens, speaker in zip(
        group["tokens"], group["unique_tokens"], group["speaker_key"]
    ):
        counts.update(tokens)
        for word in unique_tokens:
            docs[word] += 1
            speakers[word].add(str(speaker))
    token_counts[province] = counts
    document_counts[province] = docs
    speaker_sets[province] = speakers
    province_token_totals[province] = sum(counts.values())

global_counts = Counter()
global_documents = Counter()
for province in provinces:
    global_counts.update(token_counts[province])
    global_documents.update(document_counts[province])

total_tokens = sum(global_counts.values())
vocabulary_size = len(global_counts)

print("Tổng token sau lọc:", total_tokens)
print("Kích thước vocabulary:", vocabulary_size)
print("Số token/tỉnh:")
for province in provinces:
    print(f"  {province}: {province_token_totals[province]}")


# %% [code]
def benjamini_hochberg(pvalues):
    """Điều chỉnh p-value theo Benjamini-Hochberg, giữ nguyên thứ tự đầu vào."""
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.empty_like(ranked)
    n = len(ranked)
    running = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        running = min(running, ranked[i] * n / rank)
        adjusted[i] = running
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result

def chi_square_pvalue(table):
    try:
        _, p_value, _, _ = chi2_contingency(table, correction=False)
        return float(p_value)
    except (ValueError, ZeroDivisionError):
        return 1.0

rows = []
for province in provinces:
    counts_p = token_counts[province]
    docs_p = document_counts[province]
    speakers_p = speaker_sets[province]
    tokens_p = province_token_totals[province]
    n_docs_p = int((frame["province_name"] == province).sum())

    tokens_rest = total_tokens - tokens_p
    n_docs_rest = len(frame) - n_docs_p

    for word, count_p in counts_p.items():
        document_count_p = int(docs_p[word])
        speaker_count_p = len(speakers_p[word])

        count_rest = int(global_counts[word] - count_p)
        document_count_rest = int(global_documents[word] - document_count_p)

        # TF-IPF dùng TF chuẩn hóa theo một triệu token để so sánh tỉnh lệch kích thước.
        tf_per_million = count_p / tokens_p * 1_000_000 if tokens_p else 0.0
        province_frequency = sum(word in token_counts[p] for p in provinces)
        ipf = math.log((n_provinces + 1) / (province_frequency + 1))
        tf_ipf = tf_per_million * ipf

        # Log-odds với Dirichlet prior.
        p_p = (count_p + ALPHA) / (tokens_p + ALPHA * vocabulary_size)
        p_rest = (
            (count_rest + ALPHA) / (tokens_rest + ALPHA * vocabulary_size)
            if tokens_rest > 0 else ALPHA / (ALPHA * vocabulary_size)
        )
        log_odds = math.log((p_p / (1 - p_p)) / (p_rest / (1 - p_rest)))

        # PMI giữa word và tỉnh.
        pmi = math.log2(
            (count_p * total_tokens) / (global_counts[word] * tokens_p)
        ) if tokens_p and global_counts[word] else 0.0

        # Kiểm định trên số token và trên số phiên thoại.
        token_table = [
            [count_p, max(tokens_p - count_p, 0)],
            [count_rest, max(tokens_rest - count_rest, 0)],
        ]
        document_table = [
            [document_count_p, max(n_docs_p - document_count_p, 0)],
            [document_count_rest, max(n_docs_rest - document_count_rest, 0)],
        ]

        rows.append({
            "province_name": province,
            "word": word,
            "token_count": int(count_p),
            "document_count": document_count_p,
            "speaker_count": speaker_count_p,
            "province_token_total": int(tokens_p),
            "province_document_total": n_docs_p,
            "global_token_count": int(global_counts[word]),
            "province_frequency": int(province_frequency),
            "tf_per_million": tf_per_million,
            "ipf": ipf,
            "tf_ipf": tf_ipf,
            "log_odds": log_odds,
            "pmi": pmi,
            "p_value_token": chi_square_pvalue(token_table),
            "p_value_document": chi_square_pvalue(document_table),
        })

stats = pd.DataFrame(rows)
if stats.empty:
    raise ValueError("Không có token để thống kê. Kiểm tra lại metadata và cột text.")

stats["p_value_token_fdr"] = benjamini_hochberg(stats["p_value_token"])
stats["p_value_document_fdr"] = benjamini_hochberg(stats["p_value_document"])
stats["is_candidate"] = (
    (stats["log_odds"] > 0)
    & (stats["p_value_token_fdr"] < P_VALUE_THRESHOLD)
    & (stats["p_value_document_fdr"] < P_VALUE_THRESHOLD)
)

print("Số cặp tỉnh-từ được thống kê:", len(stats))
print("Số ứng viên đạt log-odds > 0 và FDR < 0.05:", int(stats["is_candidate"].sum()))


# %% [code]
candidate_words = (
    stats[stats["is_candidate"]]
    .sort_values(
        ["province_name", "token_count", "document_count", "speaker_count", "log_odds"],
        ascending=[True, True, True, True, True],
    )
    .copy()
)
candidate_words["rank_in_province"] = (
    candidate_words.groupby("province_name").cumcount() + 1
)
top_candidates = candidate_words[
    candidate_words["rank_in_province"] <= TOP_K_PER_PROVINCE
].copy()

# Lưu thống kê đầy đủ và danh sách ứng viên.
stats_path = OUTPUT_DIR / "central_province_word_statistics.csv"
candidate_path = OUTPUT_DIR / "central_province_dialect_candidates.csv"
top_path = OUTPUT_DIR / "central_province_top_words.csv"
stats.to_csv(stats_path, index=False, encoding="utf-8-sig")
candidate_words.to_csv(candidate_path, index=False, encoding="utf-8-sig")
top_candidates.to_csv(top_path, index=False, encoding="utf-8-sig")

# Tóm tắt số lượng từ và độ phủ theo tỉnh.
province_summary = (
    stats.groupby("province_name", as_index=False)
    .agg(
        n_words_after_filter=("word", "nunique"),
        n_significant_candidates=("is_candidate", "sum"),
        total_tokens=("province_token_total", "first"),
        total_documents=("province_document_total", "first"),
    )
)
province_summary.to_csv(
    OUTPUT_DIR / "central_province_lexical_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# Lấy các câu ví dụ để con người nghe/kiểm tra thủ công.
example_rows = []
for row in top_candidates.itertuples(index=False):
    subset = frame[
        (frame["province_name"] == row.province_name)
        & frame["tokens"].map(lambda tokens: row.word in tokens)
    ].head(5)
    for sample in subset.itertuples(index=False):
        example_rows.append({
            "province_name": row.province_name,
            "word": row.word,
            "filename": sample.filename,
            "speakerID": sample.speakerID,
            "split": sample.split,
            "text": sample.text,
        })
pd.DataFrame(example_rows).to_csv(
    OUTPUT_DIR / "central_province_candidate_examples.csv",
    index=False,
    encoding="utf-8-sig",
)

manifest = {
    "input_metadata": str(INPUT_METADATA),
    "n_records": int(len(frame)),
    "n_provinces": int(n_provinces),
    "total_tokens": int(total_tokens),
    "vocabulary_size": int(vocabulary_size),
    "frequency_filter": None,
    "document_frequency_filter": None,
    "speaker_coverage_filter": None,
    "dirichlet_alpha": ALPHA,
    "p_value_threshold": P_VALUE_THRESHOLD,
    "n_pairs_after_filter": int(len(stats)),
    "n_candidates": int(len(candidate_words)),
    "output_dir": str(OUTPUT_DIR),
}
(OUTPUT_DIR / "central_province_lexicon_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(json.dumps(manifest, ensure_ascii=False, indent=2))


# %% [code]
# print("\nTop ứng viên theo từng tỉnh:")
# if top_candidates.empty:
#     print("Không có ứng viên đạt toàn bộ điều kiện. Hãy kiểm tra thống kê đầy đủ hoặc giảm ngưỡng.")
# else:
#     for province, group in top_candidates.groupby("province_name", sort=True):
#         print(f"\n[{province}]")
#         # ranked_group = group.sort_values(
#         #     ["token_count", "document_count", "speaker_count", "log_odds"],
#         #     ascending=[True, True, True, True],
#         # ).head(25).reset_index(drop=True)
#         ranked_group = group.reset_index(drop=True)
#         for rank, row in enumerate(ranked_group.itertuples(index=False), start=1):
#             print(
#                 f"  {rank:>2}. {row.word:<16} "
#                 f"freq={row.token_count:<4} speakers={row.speaker_count:<3} "
#                 f"TF-IPF={row.tf_ipf:.2f} log-odds={row.log_odds:.2f} "
#                 f"FDR(token)={row.p_value_token_fdr:.3g}"
#             )

# print("\nCác file kết quả:")
# for path in sorted(OUTPUT_DIR.glob("central_province_*.csv")):
#     print(" ", path)
# print("\nBước tiếp theo: mở central_province_candidate_examples.csv, nghe audio và gán nhãn thủ công cho từng ứng viên.")


print("\nTop ứng viên theo từng tỉnh:")
if top_candidates.empty:
    print("Không có ứng viên đạt toàn bộ điều kiện. Hãy kiểm tra thống kê đầy đủ hoặc giảm ngưỡng.")
else:
    for province, group in top_candidates.groupby("province_name", sort=True):
        print(f"\n[{province}]")
        for row in group.head(250).itertuples(index=False):
            print(
                f"  {row.rank_in_province:>2}. {row.word:<16} "
                f"freq={row.token_count:<4} speakers={row.speaker_count:<3} "
                f"TF-IPF={row.tf_ipf:.2f} log-odds={row.log_odds:.2f} "
                f"FDR(token)={row.p_value_token_fdr:.3g}"
            )
print("\nCác file kết quả:")
for path in sorted(OUTPUT_DIR.glob("central_province_*.csv")):
    print(" ", path)
print("\nBước tiếp theo: mở central_province_candidate_examples.csv, nghe audio và gán nhãn thủ công cho từng ứng viên.")

# %% [markdown]
# ## Diễn giải và kiểm tra thủ công
#
# - central_province_word_statistics.csv chứa toàn bộ token được thống kê, không áp dụng ngưỡng tần suất, document frequency hoặc speaker coverage.
# - central_province_dialect_candidates.csv chỉ giữ từ có log-odds dương và đạt FDR ở cả kiểm định token lẫn kiểm định phiên thoại.
# - central_province_candidate_examples.csv cung cấp các câu mẫu để nghe audio và loại tên riêng, lỗi transcript hoặc idiolect.
# - central_province_top_words.csv là danh sách xếp hạng để xây dựng từ điển địa phương cấp tỉnh.
#
# Các p-value cần được diễn giải cùng với speaker coverage và độ lệch kích thước dữ liệu giữa các tỉnh; không nên xem p-value là bằng chứng duy nhất của một từ địa phương.
#
