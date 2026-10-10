# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
# ---

# %% [markdown]
# # Phân tích WER/CER benchmark
#
# Notebook tính WER và CER của từng model ở ba mức: tổng thể, theo vùng miền và theo tỉnh.
#

# %% [code]
from pathlib import Path
import json
import unicodedata

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('display.precision', 2)


INPUT_CSV = Path("quick_predictions.csv")
OUTPUT_DIR = INPUT_CSV.parent / "quick_predictions_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Input:", INPUT_CSV)
print("Output directory:", OUTPUT_DIR)


# %% [code]
if not INPUT_CSV.is_file():
    raise FileNotFoundError(f"Không tìm thấy file: {INPUT_CSV}")

raw = pd.read_csv(INPUT_CSV)
required = {"model", "reference", "prediction"}
missing = required - set(raw.columns)
if missing:
    raise ValueError(f"Thiếu cột bắt buộc: {sorted(missing)}")

def nonempty(value):
    return not pd.isna(value) and str(value).strip() != ""

error_text = raw.get("error", pd.Series("", index=raw.index)).fillna("").astype(str).str.strip()
is_sample = raw.get("record_type", pd.Series("sample", index=raw.index)).eq("sample")
is_success = is_sample & error_text.eq("")
df = raw.loc[is_success].copy()
if df.empty:
    raise ValueError("Không có sample thành công để tính metric.")

def clean_text(value):
    if pd.isna(value):
        return ""
    value = unicodedata.normalize("NFC", str(value).lower())
    return " ".join("".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in value).split())

def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, token_left in enumerate(left, 1):
        current = [i]
        for j, token_right in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token_left != token_right)))
        previous = current
    return previous[-1]

def calculate_wer(reference, prediction):
    ref = clean_text(reference).split()
    hyp = clean_text(prediction).split()
    return edit_distance(ref, hyp) / max(1, len(ref))

def calculate_cer(reference, prediction):
    ref = clean_text(reference).replace(" ", "")
    hyp = clean_text(prediction).replace(" ", "")
    return edit_distance(list(ref), list(hyp)) / max(1, len(ref))

# Dùng metric trong CSV; nếu một dòng thiếu WER/CER thì tính lại từ reference/prediction.
df["wer"] = pd.to_numeric(df.get("wer"), errors="coerce") if "wer" in df else np.nan
df["cer"] = pd.to_numeric(df.get("cer"), errors="coerce") if "cer" in df else np.nan
for index in df.index[df["wer"].isna()]:
    df.at[index, "wer"] = calculate_wer(df.at[index, "reference"], df.at[index, "prediction"])
for index in df.index[df["cer"].isna()]:
    df.at[index, "cer"] = calculate_cer(df.at[index, "reference"], df.at[index, "prediction"])

for column in ("region", "province_name"):
    if column not in df:
        df[column] = "Unknown"
    df[column] = df[column].fillna("Unknown").replace("", "Unknown")

print("Rows in CSV:", len(raw))
print("Successful samples:", len(df))
print("Error rows:", int((~is_success).sum()))
print("Models:", df["model"].nunique())


# %% [code]
def summarize(frame, group_columns):
    grouped = frame.groupby(group_columns, dropna=False)
    result = grouped.agg(
        # n_success=("prediction", "size"),
        wer=("wer", "mean"),
        cer=("cer", "mean"),
        wer_median=("wer", "median"),
        cer_median=("cer", "median"),
    ).reset_index()
    return result.sort_values(group_columns + ["wer", "cer"]).reset_index(drop=True)

overall = summarize(df, ["model"])
by_region = summarize(df, ["model", "region"])
by_province = summarize(df, ["model", "province_name"])

# Bổ sung số dòng lỗi để phân biệt metric thấp với tỷ lệ chạy thành công thấp.
raw_model_counts = raw.groupby("model", dropna=False).size().rename("n_total").reset_index()
# success_model_counts = df.groupby("model", dropna=False).size().rename("n_success").reset_index()
overall = (
    # overall.drop(columns=["n_success"])
    overall.merge(success_model_counts, on="model", how="left")
    .merge(raw_model_counts, on="model", how="left")
)
overall["n_errors"] = overall["n_total"] - overall["n_success"]
overall["success_rate"] = overall["n_success"] / overall["n_total"]

print("Overall:")
display(overall.sort_values(["wer", "cer"]))
print("Theo vùng miền:")
display(by_region)
print("Theo tỉnh:")
display(by_province.head(30))


# %% [code]
OVERALL_CSV = OUTPUT_DIR / "overall_metrics.csv"
REGION_CSV = OUTPUT_DIR / "metrics_by_region.csv"
PROVINCE_CSV = OUTPUT_DIR / "metrics_by_province.csv"
CLEAN_ROWS_CSV = OUTPUT_DIR / "successful_rows_with_metrics.csv"
SUMMARY_JSON = OUTPUT_DIR / "metrics_summary.json"

overall.sort_values(["wer", "cer"]).to_csv(OVERALL_CSV, index=False)
by_region.to_csv(REGION_CSV, index=False)
by_province.to_csv(PROVINCE_CSV, index=False)
df.to_csv(CLEAN_ROWS_CSV, index=False)
summary = {
    "input_csv": str(INPUT_CSV),
    "n_rows_input": int(len(raw)),
    "n_successful_rows": int(len(df)),
    "n_error_rows": int((~is_success).sum()),
    "n_models": int(df["model"].nunique()),
    "models": sorted(df["model"].dropna().unique().tolist()),
    "aggregation": "macro mean of per-audio WER/CER; errors excluded",
    "outputs": {"overall": str(OVERALL_CSV), "region": str(REGION_CSV), "province": str(PROVINCE_CSV)},
}
SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print("Saved:")
for path in (OVERALL_CSV, REGION_CSV, PROVINCE_CSV, CLEAN_ROWS_CSV, SUMMARY_JSON):
    print(" -", path)


# %% [code]
# Biểu đồ so sánh overall và theo vùng miền.
plot_data = overall.sort_values("wer").set_index("model")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
plot_data["wer"].plot.bar(ax=axes[0], color="#2563eb", title="Overall WER theo model")
plot_data["cer"].plot.bar(ax=axes[1], color="#16a34a", title="Overall CER theo model")
for ax, label in zip(axes, ("WER", "CER")):
    ax.set_xlabel("Model")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
plt.tight_layout()
plt.show()

region_pivot = by_region.pivot(index="region", columns="model", values="wer")
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(region_pivot.values, aspect="auto", cmap="YlOrRd")
ax.set_title("WER theo vùng miền")
ax.set_xlabel("Model")
ax.set_ylabel("Vùng miền")
ax.set_xticks(range(len(region_pivot.columns)), region_pivot.columns, rotation=45, ha="right")
ax.set_yticks(range(len(region_pivot.index)), region_pivot.index)
fig.colorbar(im, ax=ax, label="WER")
plt.tight_layout()
plt.show()


# %% [markdown]
# ## Cách đọc kết quả
#
# - `overall_metrics.csv`: WER/CER tổng thể của từng model.
# - `metrics_by_region.csv`: WER/CER tách theo Bắc, Trung, Nam.
# - `metrics_by_province.csv`: WER/CER tách theo từng tỉnh.
# - Các giá trị WER/CER là **trung bình macro trên từng audio thành công**; các dòng có `error` bị loại khỏi metric nhưng vẫn được đếm trong `n_errors`.
# - Nếu cần tính micro-WER/CER trên toàn bộ chuỗi nối tiếp nhau, phải lưu số phép thay thế/xóa/chèn của từng sample; notebook này dùng đúng cách tổng hợp benchmark hiện tại là trung bình per-sample.
#
