# Kokoro + JParaCrawl pre-production changes

The [input manifest I/O report](kokoro-input-index-report.md) documents the later input-side optimization.

## Git

- branch: `feature/kokoro-tts`
- base commit: `295a58d`
- This report describes the work on the feature branch at the time of implementation.

## Changes

| Area | Behavior |
| --- | --- |
| Release splits | Export QC-accepted pairs separately for train, dev, and test. `accepted.jsonl` includes all splits. |
| Checkpoint frequency | Default to 100 utterances in Kokoro production, counting Japanese and English together. |
| QC checkpoints | Decouple checkpoint frequency from ASR batch size. Save every 100 utterances and at language, phase, exception, and shard boundaries. |
| Directory scans | Scan each monitored root once per CLI process, then track byte-size deltas from atomic replacements. |
| Streaming consolidation | Store input and QC JSON in a local temporary SQLite index. Write one record at a time in input order and hold about one shard of QC in memory. |
| Benchmark selection | Select train only using seed 42 and SHA-256 ranking of pair IDs. Request 20/150/400/330/100 pairs for bins 0/1/2/3/4, then restore input order. |
| Duration statistics | Count duration failures by length bin and language, using the existing 0.7–30 second bounds. |

The changes preserve external validation, text, splits, voices, model revisions, audio formats, shared KModel, Whisper QC, resume, provenance, audio SHA-256 checks, and production approval. Each record retains its provenance object for compatibility with existing checkpoints.

SQLite uses local temporary storage outside Drive. A UNIQUE constraint checks input ID uniqueness and consolidation checks the ID set and canonical order. For external profiles, `validate_records()` checks at most one shard at a time, including metadata, provenance, selected audio, QC status, WAV format, hash, and duration.

Publication stages each file as `.tmp` and flushes and syncs it to disk. It invalidates the previous summary before replacing the JSONLs, then publishes the new summary last. Treat the summary as the completion marker for the set of files. A validation error leaves the previous release intact. A publication error leaves the new release without a completion marker.

## Tests

```text
pytest: 85 passed, 2 existing tar-extraction DeprecationWarnings
legacy Qwen regression: passed (production-qwen17b.yaml / colab-pro.yaml)
Kokoro tests: passed
streaming consolidate: passed; canonical order, split output, ID validation,
  corrupt WAV, metadata/provenance/selected-audio mismatch, incomplete,
  interrupted publication and completion marker
checkpoint interval: passed; ASR batch_size=8 / interval=100,
  language/phase end and exception writes, TTS utterance counting
stratified sampling: passed; 1000 synthetic records selected from 2500,
  exact quota, train only, deterministic seed, unique IDs, unchanged records
```

The Windows validation used a temporary directory under the workspace virtual environment.

```powershell
$env:TEMP = Join-Path $PWD '.venv'
$env:TMP = $env:TEMP
.venv\Scripts\python.exe -m pytest -q --basetemp=.venv/test-scale-05
```

The existing Kokoro runtime check also passed with one fixed sentence in each language on CPU. It verified conversion from 24 kHz native audio to 16 kHz mono PCM16. This check exercised TTS only.

```text
report: data/kokoro-scale-runtime-check/report.json (local output)
load: 2.399 seconds
JA: audio 3.875 seconds / TTS 1.647 seconds / RTF 0.425
EN: audio 3.800 seconds / TTS 1.232 seconds / RTF 0.324
returncode: 0
```

The local test used `kokoro==0.9.4`, CPU Torch 2.7.1, UniDic 3.1.0+2021-08-31, and the Windows test dependency pyopenjtalk-plus. Colab uses standard pyopenjtalk. The known Windows phonemizer cleanup warning appeared after successful generation and WAV verification. Colab compute-unit estimates require a Colab benchmark.

## Release files

| File | Contents |
| --- | --- |
| `all.jsonl` | All QC results, including accepted and rejected pairs, in input order |
| `train.jsonl` | Accepted train pairs in input order |
| `dev.jsonl` | Accepted dev pairs in input order |
| `test.jsonl` | Accepted test pairs in input order. Empty when the test split has zero pairs. |
| `accepted.jsonl` | Accepted pairs from all splits in input order |
| `summary.json` | Completion marker with counts and acceptance rates by split, audio duration, RTF, I/O, and duration failures by bin |

For the current `alignment-transfer-s2ut` workflow, use `accepted.jsonl` to supply train and dev in the same manifest.

`train_ja_hours` and `train_en_hours` describe training audio. `ja_hours` and `en_hours` report accepted duration across all splits.

The benchmark records selection method, seed, quotas, source manifest SHA-256, and selected IDs in `input.selection.json` and the release summary. The notebook displays quality, length-bin counts, RTF, ASR and full wall time, checkpoint and scan times, GPU, and peak VRAM. The 1,000-pair benchmark and production require explicit opt-in.

## Performance-related defaults

```text
num_shards: 128
max_pairs_per_shard: 3000
checkpoint_every: 100 utterances
ASR batch_size: 8
```

The notebook reuses existing Drive settings. If `run.checkpoint_every` is still 10 in a Kokoro profile, set it to 100 to use the new interval. Existing Qwen profiles use their own checkpoint settings.

I/O metrics include `checkpoint_writes` (attempted saves), `checkpoint_write_seconds`, `directory_scan_count`, and `directory_scan_seconds`. `current_run_written_bytes` counts bytes written, including replacements. The baseline includes all files in the output directory. Later WAV, checkpoint, and report replacements update usage by their size difference. Run one writer per output directory.

## Validation pending at the time

The report was written before the following production checks.

- Validation of the real JParaCrawl export
- Colab T4 10 pair smoke
- Colab T4 100 pair smoke
- Colab T4 1000 pair benchmark
- Full 300,000-pair production run

Production approval was gated on real-data quality, VRAM, I/O, and compute-unit checks. The CLI requires `--allow-production`, and the notebook defaults to `APPROVE_PRODUCTION=False` and `RUN_ALL_REMAINING=False`.
