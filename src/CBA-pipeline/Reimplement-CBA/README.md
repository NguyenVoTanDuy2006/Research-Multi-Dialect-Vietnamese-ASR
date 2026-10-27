# Reimplement-CBA

Reimplementation of the CBA-Whisper training flow for ViMD. The package is separate from the author's checkout in `../CBA-Whisper` and reuses the workspace's `mdv_asr_module` for ViMD dataset access.

## Push the branch and clone it elsewhere

From the workspace root on the development machine, publish this branch first:

```bash
git push -u origin research-CBA-Whisper
```

To clone the branch on another machine:

```bash
git clone --branch research-CBA-Whisper --single-branch https://github.com/NguyenVoTanDuy2006/Research-Multi-Dialect-Vietnamese-ASR.git
cd Research-Multi-Dialect-Vietnamese-ASR
python -m pip install -e '.[data,train,semantic]'
python src/CBA-pipeline/Reimplement-CBA/notebooks/whisper_cl.py
```

On Kaggle, add a secret named `HF_TOKEN`, grant the notebook access to it, and enable Internet. The first executable cell of `notebooks/whisper_cl.py` checks that token, clones this branch into `/kaggle/working/mdv_asr_source`, installs the Kaggle requirements plus semantic metrics, and logs in to Hugging Face before any model download. Set `VIMD_ROOT` if the ViMD small dataset is mounted at a different location. No token belongs in the script or its printed output.

## Validate and inspect the workflow

Run these commands from this directory:

```bash
PYTHONPATH=../../../src python -m reimplement_cba --help
PYTHONPATH=../../../src python -m reimplement_cba validate --config configs/vimd_phowhisper_small.json
PYTHONPATH=../../../src python -m reimplement_cba plan --config configs/vimd_phowhisper_small.json
```

`validate --check-data` additionally opens the configured dataset source and checks the required splits. It does not download data; configure a mounted ViMD root in the JSON file when the existing path is unavailable. Relative `root_override` paths resolve from the CBA config file.

The default CLI config uses ViMD small. Set `dataset.root_override` in the JSON config to your local ViMD small directory before running against data. `validate --check-data` checks that root and its train/validation/test splits.

## Run the complete workflow

From this directory, after installing the dependencies and mounting ViMD small:

```bash
PYTHONPATH=../../../src python -m reimplement_cba run --config configs/vimd_phowhisper_small.json
```

For a smoke run, open [`notebooks/whisper_cl.py`](notebooks/whisper_cl.py) as a Python notebook in VS Code or Kaggle (`# %%` cells), or run it with `python notebooks/whisper_cl.py`. On Kaggle, its first code cell clones the `research-CBA-Whisper` branch into `/kaggle/working/mdv_asr_source` and installs dependencies. It checks dataset splits, then runs one CL round with one training epoch on the first three eligible train samples and the first three validation/test samples. Edit `VIMD_ROOT`, `SMOKE_SAMPLES_PER_SPLIT`, and the round/epoch settings in the script. Artifacts are saved under `/kaggle/working/outputs/cba_whisper/`, outside the source checkout. Run the final cleanup cell after reporting to remove only the source checkout. The smoke metrics are for pipeline checking only.

After a failed run, rerun the script's configuration cell to create a new output directory before retrying. The previous `run-summary.json` is not resumable.

To use PhoWhisper-large with the same ViMD small dataset, select [`configs/vimd_phowhisper_large.json`](configs/vimd_phowhisper_large.json). It uses training batch size 1 with gradient accumulation 6, and batch size 1 for inference and semantic metrics. Set `CONFIG_FILENAME = 'vimd_phowhisper_large.json'` in the smoke script, or run the full workflow with:

```bash
PYTHONPATH=../../../src python -m reimplement_cba run --config configs/vimd_phowhisper_large.json
```

`run` filters the training split, transcribes the retained training samples and full validation split with the unadapted PhoWhisper model, and computes baseline SemScore. Each curriculum round trains AdaLoRA, transcribes validation with every saved epoch checkpoint and the final adapter, ranks them by validation SemScore then WER, and scores the selected adapter on the current training pool. The adaptive rule chooses samples for the next round. The best round checkpoint is selected on validation; test is evaluated once for final reporting. `training.epochs` controls epochs **within** each round; `curriculum.max_rounds` controls the number of training rounds.

Results are written under `output_dir`: `run-summary.json`, initial exclusions in `preprocess/`, validation and training predictions in `predictions/`, per-sample decisions in `curriculum/`, and adapters in `adapters/round-N/`. Use a fresh `output_dir` for each run; the command refuses to reuse an existing run summary or round-1 adapter. For faster-whisper or WhisperX, `run` prepares the required CT2 baseline and checkpoint artifacts automatically unless `inference.ct2_model_path` supplies an existing baseline artifact.

## Pipeline defaults

- Default baseline and fine-tuning base: `vinai/PhoWhisper-medium`.
- `model.base_model` accepts another Hugging Face Whisper-compatible model ID or local checkpoint path; `model.processor_model` can select separate processor files, and defaults to the model checkpoint.
- Language/task: Vietnamese transcription (`vi`, `transcribe`).
- Curriculum domain: ViMD is treated as in-domain; threshold 50 is configurable.
- Initial training filtering removes configured transcript annotation markers and exact sample IDs in `manifests/vimd_english_sample_ids.json`. The English list is manually reviewed; it does not use automatic language detection. Filtering applies before the baseline and each AdaLoRA round.
- Fine-tuning accepts audio from 0.5 to 30 seconds. Records outside this range are excluded from training and from the validation subset passed to `Seq2SeqTrainer`; each round writes their IDs, split, duration, and reason to `outputs/.../preprocess/round-N-rejected.jsonl`. Full validation and test records remain available for long-form inference and checkpoint metrics.
- AdaLoRA starts at rank 12 and targets rank 4; two curriculum rounds resume the previous adapter.
- PhoWhisper-medium configs use a per-device training batch of 1 with gradient accumulation of 6 to reduce activation memory. Trainer scales the effective batch by the visible GPU count.
- Validation checkpoint ordering is SemScore descending, then WER ascending. Test is reserved for final reporting.
- The sample configs allow up to 256 newly generated tokens for short-form ASR and 512 tokens per NLI reference/hypothesis pair. Short-form inference converts the token budget to a total decoder length that includes Whisper's four prefix tokens, avoiding a conflict with the checkpoint's saved generation length. NLI tokenization truncates pairs to `metrics.nli.max_length` so a long transcript does not exceed the model's input limit.

## PhoWhisper fine-tuning and inference backends

Fine-tuning still uses Transformers and PEFT. Inference defaults to the Transformers backend; select `faster_whisper` or `whisperx` in `inference.backend` to use a CTranslate2 checkpoint. Install the matching extras from the repository root:

```bash
pip install -e '.[data,train,inference,whisperx]'
```

Use Python 3.11 for the WhisperX environment. The current WhisperX package requires Python below 3.14 and installs a Torch/CUDA-compatible dependency stack. `faster-whisper` and WhisperX both run CTranslate2 weights; they do not load the Transformers checkpoint or PEFT adapter directly. The Transformers backend continues to load the configured model and optional adapter as before.

Prepare the unadapted baseline checkpoint once:

```bash
PYTHONPATH=../../../src python -m reimplement_cba prepare-ct2 \
  --config configs/vimd_phowhisper_medium.json \
  --output ../outputs/ct2/phowhisper-medium-base
```

For an AdaLoRA checkpoint, merge and convert that adapter into its own artifact:

```bash
PYTHONPATH=../../../src python -m reimplement_cba prepare-ct2 \
  --config configs/vimd_phowhisper_medium.json \
  --adapter-path ../outputs/cba_vimd_phowhisper_medium/adapters/round-1/checkpoint-final \
  --output ../outputs/ct2/phowhisper-medium-round-1
```

Set `inference.backend` to `faster_whisper` or `whisperx`, then run inference with the artifact for that exact base/adapter combination:

```bash
PYTHONPATH=../../../src python -m reimplement_cba infer \
  --config configs/vimd_phowhisper_medium.json \
  --split validation \
  --ct2-model-path ../outputs/ct2/phowhisper-medium-round-1 \
  --adapter-path ../outputs/cba_vimd_phowhisper_medium/adapters/round-1/checkpoint-final
```

WhisperX runs ASR with VAD and cut-and-merge, configured by `inference.vad` (defaults: method `pyannote`, chunk size 30 seconds, onset/offset 0.3). It does not run forced alignment or diarization. Direct faster-whisper inference does not use VAD. Both stages write the selected backend and CT2 artifact path into each prediction row.

The `prepare-ct2` command requires `ct2-transformers-converter`, PyTorch, Transformers, and PEFT when an adapter is supplied. It refuses to overwrite an existing artifact. Conversion uses float16; inference follows `model.compute_type` and changes float16/bfloat16 to float32 on CPU. The artifact includes provenance metadata and inference checks its base model, configured revision, and adapter against the requested run.

The model and processor are loaded lazily when a stage runs. `model.base_model` selects the checkpoint for that config; for example, a config can point at `vinai/PhoWhisper-large`, another Whisper checkpoint, or a compatible local model directory. `model.processor_model` and `model.revision` are optional. A PEFT adapter is loaded on top of the configured base model and stays separate from the base weights during Transformers inference. Round adapters keep their initial factor dimensions and save pruned singular values as zeros, so the next curriculum round can resume from those weights and start a fresh AdaLoRA allocation schedule.

Transformers inference accepts long-form audio up to `audio.long_form_max_duration_seconds` (600 seconds in the sample config) and uses Whisper's timestamp-based long-form decoding. faster-whisper and WhisperX use their CTranslate2 long-form handling. Fine-tuning examples remain limited by `audio.max_duration_seconds` (30 seconds). The ViMD record schema has no transcript timestamps, so training samples over 30 seconds are excluded and reported rather than truncated or assigned mismatched labels. Audio is validated after ViMD decoding; loudness normalization and additional VAD are not applied by the shared preprocessor.

The Python stages accept records from `ViMDAdapter` and its `read_audio` method:

```python
from reimplement_cba.adapters.vimd import ViMDAdapter
from reimplement_cba.config import CBAConfig
from reimplement_cba.stages.initial_inference import InitialInferenceStage
from reimplement_cba.stages.inference import AdaptedInferenceStage
from reimplement_cba.stages.finetune import AdaLoRATrainingStage

config = CBAConfig.from_file("configs/vimd_phowhisper_medium.json")
with ViMDAdapter(config) as vimd:
    records = vimd.open().index.records
    train_records = [record for record in records if record.split == "train"]
    validation_records = [record for record in records if record.split == "validation"]

    baseline_rows = InitialInferenceStage(config).run(train_records, vimd.read_audio)
    adapter_path = AdaLoRATrainingStage(config).run(
        train_records,
        validation_records,
        audio_reader=vimd.read_audio,
        round_index=1,
    )
    adapted_rows = AdaptedInferenceStage(config, adapter_path).run(
        validation_records,
        vimd.read_audio,
    )
```

Standalone baseline predictions are written to `outputs/.../predictions/baseline.jsonl`; adapted predictions use an adapter-specific JSONL filename. A complete `run` writes split-specific prediction files as described above. Rows contain sample IDs, reference text, the NFC/whitespace-normalized `hypothesis`, unmodified `hypothesis_raw`, and available region/province metadata. Postprocessing preserves Vietnamese diacritics, punctuation, case, and repeated or disfluent words. One AdaLoRA round writes adapter checkpoints under `outputs/.../adapters/round-N/`. The model is not merged automatically unless a CT2 artifact is requested for inference.

## Implementation status

ViMD config validation/audio access, transcript filtering, training-duration preprocessing, minimal Vietnamese postprocessing, curriculum decisions, checkpoint selection, Transformers/faster-whisper/WhisperX inference, CT2 checkpoint preparation, prediction JSONL, AdaLoRA training, Vietnamese WER/SemScore, and end-to-end `run` orchestration are implemented. The English exclusion manifest starts empty and must be filled with approved ViMD `sample_id` values to exclude those records from training. WhisperX VAD/cut-and-merge is available only when that backend is selected.

The paper's English SAPC SemScore implementation is not included here. The current Vietnamese SemScore is a configurable combination of multilingual NLI, BERTScore-F1, and paper-derived phonetic similarity; it still needs calibration on ViMD before its values are treated as validated.

## Vietnamese phonetics resources

The paper-based phoneme inventory and dialect transformation tables are in `resources/phonetics/`; the scoring converter is in `reimplement_cba/adapters/vietnamese_phonetics.py`. See [`docs/phonetics.md`](docs/phonetics.md) for the conversion flow, source-page references, and transcription caveats.

The metric implementation and its score definition are documented in [`docs/metrics.md`](docs/metrics.md).
