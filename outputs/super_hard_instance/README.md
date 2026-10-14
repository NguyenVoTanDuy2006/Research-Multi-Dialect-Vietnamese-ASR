# Super-hard audio — one ASR test instance

The test split is stored in `test-00000-of-00001.parquet`, a single-row Hugging Face-compatible shard. Its `audio` field contains both the relative path and embedded WAV bytes, so readers can load the audio directly from the shard. `super_hard.wav` is retained as a standalone copy for convenient inspection. The transcript is copied verbatim, without spelling or whitespace normalization.

`sample_metadata.json` and `sample_metadata.csv` describe the sample and point to its shard/row. This is an external-video example, **not a ViMD record**. Province, region, speaker, gender, and license are unknown; verify redistribution rights before using this package publicly. Keep it separate from the ViMD train/valid/test data unless you intentionally merge it.
