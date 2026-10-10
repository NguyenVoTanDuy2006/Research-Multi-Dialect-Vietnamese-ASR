# ASR performance demos

Four 16:9 videos compare the predictions in `predictions.csv` from
`pretrained` and `epoch_3_adapter` for the same audio. The words appear
progressively with the audio as a **visualization of saved predictions**;
these videos do not measure streaming ASR latency.
At the end, correctly matched prediction words become green and substituted
or inserted words become red. Omitted reference words have no prediction word
to color, but remain included in WER.

From the repository root, render all four videos:

```bash
.venv/bin/python demo/render_demos.py
```

To render one selected case, pass its `filename` from `predictions.csv`; for example:

```bash
.venv/bin/python demo/render_demos.py --case-id 37_0338.wav
```

The renderer needs ManimGL 1.7.2, `setuptools<81`, Noto Sans, FFmpeg, and an
EGL-capable OpenGL installation. It writes `demo/videos/30_0309.mp4`,
`59_0313.mp4`, `37_0338.mp4`, and `super_hard.mp4`. CER and WER are the saved error counts
divided by their reference-unit counts; those counts use normalized text.

For a one-panel word-alignment check, render `PreviewPretrainedAlignment` from
`asr_performance.py`. The checked preview is saved as
`demo/videos/preview_pretrained_alignment_30_0309.mp4`. It uses the same panel
width as the comparison videos. A single Pango-laid-out paragraph keeps all
Vietnamese words on their typographic baselines while labelled word glyphs
appear over time.

The first three audio clips are from the ViMD test split. The super-hard clip
is an external video, loaded from `outputs/super_hard_instance/audio/super_hard.wav`;
its source license is not recorded. Verify rights before redistributing that
video. ViMD clips remain subject to the ViMD license and attribution requirements.
