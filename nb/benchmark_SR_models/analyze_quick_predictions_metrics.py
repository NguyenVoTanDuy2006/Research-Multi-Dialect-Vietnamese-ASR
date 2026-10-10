# %% [markdown]
# # Analyze quick benchmark predictions
#
# Load CSV output paths below. The same normalizer and corpus-level CER/WER implementation is reused.

# %%
from pathlib import Path
import csv

try:
    from _bootstrap import initialize_project
except ImportError:
    from nb._bootstrap import initialize_project

ROOT = initialize_project()
from mdv_asr.evaluation.metrics import MetricAggregator
from mdv_asr.evaluation.records import PredictionRecord

PREDICTION_FILES = [ROOT / "outputs/benchmark_models/vimd_three_models/predictions.csv"]

def load_predictions(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for index, item in enumerate(csv.DictReader(stream)):
            rows.append(PredictionRecord(
                dataset_id=str(item.get("dataset_id") or "vimd"),
                sample_id=str(item.get("sample_id") or item.get("filename") or index),
                model_key=str(item.get("model_key") or item.get("model") or "unknown"),
                reference=str(item.get("reference") or ""),
                prediction=str(item.get("prediction") or ""),
                split=str(item.get("split") or "test"),
                filename=str(item.get("filename") or index),
                region=item.get("region") or None,
                province_name=item.get("province_name") or None,
                speaker_id=item.get("speaker_id") or item.get("speakerID") or None,
                error=str(item.get("error") or ""),
            ))
    return rows

predictions = load_predictions(PREDICTION_FILES[0])
aggregator = MetricAggregator()
for model in sorted({row.model_key for row in predictions}):
    rows = [row for row in predictions if row.model_key == model and not row.error]
    print(model, aggregator.overall(rows) if rows else "no successful rows")
