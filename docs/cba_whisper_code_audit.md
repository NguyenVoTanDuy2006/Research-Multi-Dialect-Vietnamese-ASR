# CBA-Whisper: code audit for Qwen adaptation

Repository source: `third_party/CBA-Whisper/`

The repository contains the CBA-Whisper parent project and the `Whisper-Finetune` submodule. The parent checkout has one filename containing `>` (`sapc-submissions/4_>norm_volumn.py`) that cannot be materialized on this mounted filesystem; its source remains available through the Git object. A sanitized copy can be created later if needed.

## Main training path

`whisper-finetune/finetune.py` does the following:

1. Loads `WhisperProcessor`.
2. Reads JSONL records through `utils/reader.py::CustomDataset`.
3. Resamples/mono-converts audio, filters to `[0.5, 30]` seconds, and optionally applies speed/shift/volume/noise augmentation.
4. Creates `WhisperForConditionalGeneration`.
5. Adds PEFT adapters to `k_proj`, `q_proj`, `v_proj`, `out_proj`, `fc1`, and `fc2`.
6. Uses `AdaLoraConfig(init_r=12, target_r=4, tinit=200, tfinal=1000, deltaT=10, lora_alpha=32, lora_dropout=0.1)` by default.
7. Trains with `Seq2SeqTrainer` and saves the PEFT checkpoint.

`utils/data_utils.py` pads audio features and labels separately and replaces label padding with `-100`.

## Important AdaLoRA finding

The repository calls `get_peft_model(..., AdaLoraConfig(...))`, but there is no explicit `update_and_allocate(global_step)` call in `finetune.py`, the callback, or the utility files. Current PEFT documentation requires this call at every training step (or a Trainer subclass that performs it). Therefore the repository should not automatically be assumed to execute adaptive rank allocation correctly under a current PEFT version. This must be verified or fixed before comparing it with a Qwen AdaLoRA experiment.

The configuration also does not set `total_step`; current PEFT versions use this value to define AdaLoRA's three phases. For a new implementation, compute total optimizer updates before constructing the configuration.

There is also a code/paper mismatch: the paper reports up to two epochs, while this script defaults to `num_train_epochs=10`. It also calls `prepare_model_for_kbit_training` unconditionally, even when `use_8bit=False`; this behavior should be reviewed before porting it to Qwen.

## Curriculum/data-selection implementation

The curriculum is mostly represented by successive JSONL files rather than by an in-training sampler:

| file | records | approximate hours |
|---|---:|---:|
| `step1_train.json` | 20,026 | 32.95 |
| `step2_train.json` | 23,472 | 53.48 |
| `step3_train.json` | 41,292 | 93.43 |
| `step4_train.json` | 51,193 | 101.07 |
| `step5_train.json` | 55,192 | 111.81 |

The later steps add/replace examples, but the repository does not include a complete reproducible script that computes all selection scores from raw data. The score-based selection described in the paper was performed externally during the challenge iterations.

## Inference/post-processing path

The numbered scripts under `sapc-submissions/` implement successive submissions:

- baseline inference and optional loudness increase;
- repeated-word deduplication;
- fixed gain/loudness normalization;
- NeMo English text normalization;
- second/third fine-tuning iterations.

These scripts are WhisperX/English-specific. For ViMD, the English NeMo normalizer and the ASCII-only filtering rules must not be copied unchanged because Vietnamese diacritics, English words, and regional spellings are valid output.

## Security warning

`sapc-submissions/1_step0_finetune.py` contains a hard-coded DeepSeek API key. Do not run that file or reuse the key. It should be revoked/removed in any derivative implementation; use an environment variable or skip the external LLM correction stage.

## Mapping to Qwen3-ASR

| CBA component | Qwen implementation |
|---|---|
| Whisper feature extractor | `AutoProcessor.apply_chat_template(..., output_labels=True)` |
| Whisper seq2seq labels | Qwen assistant target `language Vietnamese<asr_text>...` |
| `k/q/v/out/fc` targets | Dynamically inspect Qwen linear module names; start with decoder `q_proj/v_proj`, then test audio projector targets separately |
| `Seq2SeqTrainer` | Custom loop or Trainer subclass; call `update_and_allocate()` after every optimizer update |
| English dedup/normalization | Vietnamese Unicode normalization plus repetition detection; preserve valid English and dialect words |
| SemScore curriculum | WER/CER + phonetic error + repetition/length checks, with region/province quotas |

The existing Qwen notebook is therefore a useful AdaLoRA baseline, but a faithful Qwen version of CBA still needs: (1) explicit rank-allocation updates, (2) Qwen chat-template collation, and (3) Vietnamese/dialect-aware data scoring.
