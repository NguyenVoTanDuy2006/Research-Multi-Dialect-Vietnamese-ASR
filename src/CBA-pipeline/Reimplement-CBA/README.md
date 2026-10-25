# Reimplement-CBA

Scaffold for adapting the CBA-Whisper training flow to ViMD. The package is intentionally separate from the author's checkout in `../CBA-Whisper` and reuses the workspace's `mdv_asr_module` for ViMD dataset access.

## Validate and inspect the workflow

Run these commands from this directory:

```bash
PYTHONPATH=../../../src python -m reimplement_cba --help
PYTHONPATH=../../../src python -m reimplement_cba validate --config configs/vimd_phowhisper_medium.json
PYTHONPATH=../../../src python -m reimplement_cba plan --config configs/vimd_phowhisper_medium.json
```

`validate --check-data` additionally opens the configured dataset source and checks the required splits. It does not download data; configure a mounted ViMD root in the JSON file when the existing path is unavailable. Relative `root_override` paths resolve from the CBA config file.

`run` currently stops at the first unimplemented model stage. It does not load a checkpoint, contact the Hub, or start training.

## Pipeline defaults

- Baseline and fine-tuning base: `vinai/PhoWhisper-medium`.
- Language/task: Vietnamese transcription (`vi`, `transcribe`).
- Curriculum domain: ViMD is treated as in-domain; threshold 50 is configurable.
- Initial filtering removes only configured annotation markers from the transcript.
- AdaLoRA starts at rank 12 and targets rank 4; two curriculum rounds resume the previous adapter.
- Validation checkpoint ordering is SemScore descending, then WER ascending. Test is reserved for final reporting.

## Implementation status

ViMD config validation/audio access, the initial transcript rule, curriculum decisions, checkpoint selection, config parsing and CLI planning are scaffolded. Loudness/VAD/cut-and-merge, ASR model loading, AdaLoRA training, CBA-compatible SemScore, and Vietnamese postprocessing remain explicit TODO stages.

The paper's English SAPC SemScore implementation is not included here. Implement or connect a Vietnamese-compatible scorer before enabling curriculum scoring or checkpoint selection.
