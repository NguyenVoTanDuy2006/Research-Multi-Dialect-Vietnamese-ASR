# Semantic Metrics in ASR Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add resumable SemSim-E5 and BERTScore reporting after ASR inference while preserving existing WER/CER artifacts.

**Architecture:** Add an opt-in semantic metrics configuration to `ExperimentConfig`, a JSONL cache that persists completed pair scores per batch, and a semantic reporter that generates sample-level and grouped reports. `ASRBenchmarkWorkflow` invokes this reporter after prediction artifacts are saved; the benchmark notebook enables the feature and checks its optional dependency.

**Tech Stack:** Python dataclasses, existing `Metric`/`SemSimE5`/`BERTScoreMetric` interfaces, JSONL, existing `ArtifactStore`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-24-semantic-benchmark-design.md`

## Global Constraints

- Run SemSim-E5 and BERTScore after ASR inference, one encoder at a time.
- Preserve current prediction and WER/CER output files and behavior.
- The benchmark notebook enables both metrics; `ExperimentConfig` defaults semantic metrics to disabled.
- Deduplicate exact `(reference, prediction)` pairs; map scores back to every model/sample row.
- Exclude prediction rows with inference errors from semantic aggregation; score empty predictions with the existing metric behavior.
- Persist completed batches and resume only scores whose pair hash and metric fingerprint match.
- Use the existing long-text `chunk` policy and retain metric model/revision/config metadata.
- Keep `bert-score` optional; do not add it to the base Kaggle requirements unconditionally.

## Review Focus

- Identical transcript pairs across models must be scored once and mapped to every matching row — test in `test_semantic_benchmark_report.py`.
- Empty hypotheses score zero while inference-error rows have null scores and do not affect summaries — test in `test_semantic_benchmark_report.py`.
- A changed metric/model fingerprint must not reuse old cache rows — test in `test_semantic_score_store.py`.
- A truncated last JSONL record must not invalidate preceding completed batches — test in `test_semantic_score_store.py`.
- Disabled semantic config must not import/load semantic models or require `bert-score` — test in `test_benchmark_workflow.py`.

---

## File Map

- `src/mdv_asr_module/configs/types.py`: define validated semantic settings and expose them on `ExperimentConfig`.
- `src/mdv_asr_module/configs/loader.py`: construct semantic settings from JSON/overrides.
- `src/mdv_asr_module/evaluation/semantic_store.py`: append and recover batch score cache entries.
- `src/mdv_asr_module/evaluation/semantic_report.py`: deduplicate prediction pairs, score/resume metrics, and write semantic artifacts.
- `src/mdv_asr_module/workflows/benchmark.py`: invoke semantic reporting after existing artifacts are saved, only when enabled.
- `src/nb/benchmark.py`: enable both metrics by default, install/check the optional package, and list semantic artifacts. This is the notebook's current name after the user's uncommitted source reorganization.
- `tests/unit/test_config_and_metrics.py`: configuration validation and loader coverage.
- `tests/unit/test_semantic_score_store.py`: JSONL cache/resume behavior.
- `tests/unit/test_semantic_benchmark_report.py`: score mapping, aggregation, and report artifacts.
- `tests/unit/test_benchmark_workflow.py`: enabled/disabled workflow integration.
- `tests/unit/test_benchmark_notebook_models.py`: notebook configuration and dependency/output wiring.

## Task 1: Add the Semantic Metrics Configuration

**Files:**
- Modify: `src/mdv_asr_module/configs/types.py`
- Modify: `src/mdv_asr_module/configs/loader.py`
- Test: `tests/unit/test_config_and_metrics.py`

**Interfaces:**
- Produces `SemanticMetricsConfig(enabled=False, metrics=("semsim_e5", "bertscore"), device=None, batch_size=8, max_length=512, long_text_policy="chunk", cache_path="semantic_score_cache.jsonl")`.
- Adds `ExperimentConfig.semantic_metrics: SemanticMetricsConfig`.
- `ConfigLoader.from_dict()` and `with_overrides()` accept a `semantic_metrics` dictionary and validate it.

- [ ] **Step 1: Write failing tests** for disabled defaults, valid enabled settings, unknown metric names, invalid batch/token limits, invalid long-text policy, and loading nested semantic settings from a dict.
- [ ] **Step 2: Run the focused tests** with `pytest tests/unit/test_config_and_metrics.py -q`; confirm the new config/import/validation behavior fails before implementation.
- [ ] **Step 3: Implement `SemanticMetricsConfig.validate()`** and add it to `ExperimentConfig`; update `ConfigLoader.from_dict()` to construct the nested dataclass and preserve defaults when omitted.
- [ ] **Step 4: Re-run focused tests** and confirm all config cases pass, including existing config loading.
- [ ] **Step 5: Commit** as `feat: add semantic benchmark configuration`.

## Task 2: Add a Resumable Semantic Score Store

**Files:**
- Create: `src/mdv_asr_module/evaluation/semantic_store.py`
- Test: `tests/unit/test_semantic_score_store.py`

**Interfaces:**
- `SemanticScoreStore(path: str | Path)`.
- `load(metric_fingerprint: str) -> dict[str, dict[str, object]]` returns the last valid score for each pair hash under that fingerprint.
- `append_batch(metric_fingerprint: str, scores: Mapping[str, Mapping[str, object]]) -> None` appends one completed batch and flushes it to disk before returning.

- [ ] **Step 1: Write failing tests** for round-trip, filtering by fingerprint, latest score winning for duplicate pair hashes, and ignoring only a malformed/truncated final JSONL row while preserving earlier rows.
- [ ] **Step 2: Run** `pytest tests/unit/test_semantic_score_store.py -q`; confirm the class/API is missing.
- [ ] **Step 3: Implement `SemanticScoreStore`** using UTF-8 JSONL, append-mode batch writes, deterministic field names, and a reader that tolerates a malformed final line but raises on malformed interior lines. Flush and `fsync` completed batches before returning so a later process can resume from them.
- [ ] **Step 4: Re-run the store tests** and verify a second load recovers all complete batches.
- [ ] **Step 5: Commit** as `feat: persist semantic score batches`.

## Task 3: Implement Semantic Scoring and Reports

**Files:**
- Create: `src/mdv_asr_module/evaluation/semantic_report.py`
- Modify: `src/mdv_asr_module/evaluation/__init__.py`
- Test: `tests/unit/test_semantic_benchmark_report.py`

**Interfaces:**
- `SemanticBenchmarkReporter(output_dir, config, *, resume: bool, metric_factories=None)`.
- `write_all(predictions: Sequence[PredictionRecord]) -> dict[str, Path]` writes semantic sample, overall, region, province, and config artifacts.
- Injected metric factories map canonical names `"semsim_e5"` and `"bertscore"` to zero-argument `Metric` constructors; production defaults build the existing metric classes from `SemanticMetricsConfig`.
- Pair hashes are SHA-256 of a stable JSON encoding of the exact reference/prediction pair. Metric fingerprints include metric metadata (including resolved model revision), configured scoring settings, and the metric package version.

- [ ] **Step 1: Write failing tests** using fake metrics for: duplicate-pair scoring once; correct score mapping for multiple models; successful empty predictions receiving metric-defined zero values; error predictions producing null scores and not entering summaries; output filenames/columns; and resuming without rescoring cached pairs.
- [ ] **Step 2: Run** `pytest tests/unit/test_semantic_benchmark_report.py -q`; confirm the reporter/API is absent.
- [ ] **Step 3: Implement scoring** in `SemanticBenchmarkReporter`: build unique pairs, run a one-pair smoke check on a non-empty successful pair to load/validate each metric and obtain its metadata, load matching cached scores only when `resume=True`, score only missing pairs in configured batches, print per-batch progress, validate returned result counts/finite values, and append each completed batch. If there are no non-empty successful pairs, emit the metric-defined empty scores without loading the semantic models or requiring `bert-score`.
- [ ] **Step 4: Implement artifacts**: write `semantic_predictions.csv`, `semantic_metrics_overall.csv`, `semantic_metrics_by_region.csv`, `semantic_metrics_by_province.csv`, and `semantic_metrics_config.json`; aggregate mean/median/p05 per model and subgroup, including `n_samples` and chunked counts.
- [ ] **Step 5: Ensure model cleanup** after each metric (release encoder/scorer, collect garbage, empty CUDA cache when available), then run the report tests and confirm cache hits skip scoring.
- [ ] **Step 6: Commit** as `feat: report resumable semantic benchmark metrics`.

## Task 4: Integrate the Semantic Stage into the Benchmark Workflow

**Files:**
- Modify: `src/mdv_asr_module/workflows/benchmark.py`
- Test: `tests/unit/test_benchmark_workflow.py`

**Interfaces:**
- `ASRBenchmarkWorkflow.run_test()` retains its current return type and existing artifact behavior.
- When `config.semantic_metrics.enabled` is true, it calls `SemanticBenchmarkReporter(..., resume=config.benchmark.resume).write_all()` with all `PredictionRecord`s after the existing prediction/WER/CER artifacts have been written.

- [ ] **Step 1: Write failing tests** asserting enabled config invokes semantic reporting after the ordinary reporter, while disabled config neither imports nor constructs semantic metrics.
- [ ] **Step 2: Run** `pytest tests/unit/test_benchmark_workflow.py -q`; confirm the enabled-stage behavior fails.
- [ ] **Step 3: Implement the optional workflow stage** after `BenchmarkReporter.write_all()` and before comparison validation; preserve prediction outputs even if paired model comparison is incomplete.
- [ ] **Step 4: Re-run workflow tests** and confirm default-disabled workflows remain backward-compatible.
- [ ] **Step 5: Commit** as `feat: integrate semantic scoring into benchmark workflow`.

## Task 5: Enable Semantic Metrics in the Benchmark Notebook Script

**Files:**
- Modify: `src/nb/benchmark.py`
- Test: `tests/unit/test_benchmark_notebook_models.py`

**Interfaces:**
- Add a notebook settings cell/dictionary that enables `semsim_e5` and `bertscore`, with device auto-selection, batch size 8, max length 512, `long_text_policy="chunk"`, and cache path `semantic_score_cache.jsonl`; semantic cache resume follows the existing `RESUME` setting.
- Apply settings to the loaded config using `ConfigLoader.with_overrides()`.
- Print semantic artifact paths alongside the existing WER/CER artifact paths.

- [ ] **Step 1: Write failing tests** that inspect the notebook script for both enabled metrics, optional-dependency installation/check, config override, and semantic output listing.
- [ ] **Step 2: Run** `pytest tests/unit/test_benchmark_notebook_models.py -q`; confirm notebook wiring is missing.
- [ ] **Step 3: Implement notebook wiring**: before ASR benchmark work, install `bert-score==0.3.13` only if semantic mode is enabled and missing; fail with an actionable message if install fails; document Internet/Hugging Face model downloads in the notebook markdown.
- [ ] **Step 4: Add semantic artifact names** to the output summary and re-run notebook script tests.
- [ ] **Step 5: Commit** as `feat: enable semantic metrics in benchmark notebook`.

## Task 6: Full Verification and Kaggle-Oriented Smoke Check

**Files:**
- Test: all `tests/unit/test_*.py`

- [ ] **Step 1: Run the full unit suite** with `pytest -q`; resolve any regressions in existing WER/CER, prediction resume, config, or semantic tests.
- [ ] **Step 2: Run static checks** with `python -m compileall -q src/mdv_asr_module src/nb/benchmark.py` and `git diff --check`.
- [ ] **Step 3: Run a minimal semantic smoke** using one real prediction pair on a Kaggle GPU; verify finite E5/BERTScore values, cache entries are written, and rerun skips the cached pair.
- [ ] **Step 4: Verify generated files**: compare semantic sample row counts to successful prediction rows and confirm existing WER/CER files remain available and unchanged in schema.
