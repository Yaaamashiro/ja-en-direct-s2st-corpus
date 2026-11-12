# Kokoro + JParaCrawl runbook

[Back to README](../README.md) · [Open in Colab](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_kokoro_jparacrawl.ipynb)

## Input and pipeline

```text
JParaCrawl v3 → jparacrawl-filtering → text corpus export
→ Kokoro-82M (native 24 kHz) → 16 kHz / mono / PCM16 WAV → Whisper QC
```

`jparacrawl-filtering` handles filtering, deduplication, selection of 300,000 pairs, and the train/dev split. Run its `scripts/export_text_corpus.py` script to produce the two input files for this pipeline, `manifest.jsonl` and `summary.json`.

By default, `configs/production-kokoro-jparacrawl.yaml` validates the following files.

```text
/content/drive/MyDrive/jparacrawl-filtering/outputs/text-corpus/manifest.jsonl
/content/drive/MyDrive/jparacrawl-filtering/outputs/text-corpus/summary.json
```

Validation checks the summary's completion flag, corpus, and counts, the actual file's SHA-256 digest, and each record's ID uniqueness, nonempty text, split, corpus, and length bin. The configurable expected counts default to 300,000 total, 298,000 train, 2,000 dev, and zero test. Validation streams the records and preserves their contents and order. The official KPipeline receives the exported text as-is. G2P and QC evaluation normalization take place within their respective stages.

## Running in Colab

The notebook targets an NVIDIA L4 and uses `production-kokoro-jparacrawl-l4.yaml`. Generation requires CUDA and at least 20 GiB of VRAM. BF16 support is optional. A different GPU produces a warning with the target and actual GPU names. A T4 falls below the VRAM requirement. The original `production-kokoro-jparacrawl.yaml` profile is also available. Kokoro runs in float32 and Whisper in float16, in separate phases. The defaults are one Kokoro worker and an ASR batch size of eight. Clear any Qwen `S2ST_TTS_BATCH_SIZE` override before using the Kokoro CLI.

Choose the cells for your current stage.

| Stage | Cell order | Settings |
| --- | --- | --- |
| First run | **1 → 2 → 3** | L4. Test 10 pairs, then 100. |
| Performance tuning and final test | **4 → 5 → 6** | Enable tuning, then test 1,000 pairs with the selected settings. |
| Start or resume production | **1 → 2 → 5 → 7 → 8** | Confirm the saved settings. Start with one shard and review the result before processing the rest. |
| Export a finished release | **1 → 2 → 9** | CPU is supported. Set `RELEASE_ONLY = True` in cell 2 and `WRITE_RELEASE = True` in cell 9. |

Cell 1 mounts Drive, fetches the code, and installs dependencies. In the same runtime, reuse the existing setup. Cell 8 generates audio and runs QC for the remaining shards. Cell 9 writes the release.

The generation CLI requires `--allow-production`. To defer publication as the notebook does, run `run-remaining-shards --skip-consolidate`, then run `consolidate` separately.

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml prepare
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 10
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 100
# Optional benchmark after reviewing the smoke-test results
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 1000
```

The notebook uses its virtual environment's `PYTHON` and the Drive configuration `kokoro-jparacrawl-l4-config.yaml`. The 10- and 100-pair smoke tests take the first records in export order. The 1,000-pair benchmark requests 20/150/400/330/100 training pairs from length bins 0/1/2/3/4, ranks pair IDs by SHA-256 with seed 42, and restores input order. By default, the CLI stops on a shortfall and reports the input path, affected bins, and available training counts. With `--benchmark-shortfall redistribute`, the sampler caps each short bin at its available count and reallocates the deficit by the original quota weights, using largest remainders and bin ID for ties. An insufficient overall count still stops the run. Set `--benchmark-seed` to choose the seed. The sampled records retain their text and splits. `input.selection.json` and the release summary record the method, seed, quotas, source SHA-256, and selected IDs. Each size writes to `kokoro-l4-smoke/pairs-N/production/`. Use a fresh smoke output directory if it contains an earlier 1,000-pair benchmark that selected the first records.

## L4 performance tuning

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml benchmark-performance --stage tts
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml benchmark-performance --stage asr
# Run both stages in sequence in the same runtime
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml benchmark-performance --stage all
```

TTS tuning uses 200 training pairs with seed 42 and requested length-bin quotas of 4/30/80/66/20. When a bin is short, redistribution preserves the total without duplicating records. Logs and selection metadata record the requested and actual quotas, available training counts, and the policy. All worker counts (1, 2, 4, and 6 by default) use the same text and voices. Audio goes to the local SSD at `/content/kokoro-l4-benchmark`. Results, timestamped snapshots, and logs go to Drive under `kokoro-performance/`, with the latest report at `l4-performance.json`. The report also records input and audio SHA-256 digests. Override candidates and paths with `--local-dir`, `--report-dir`, `--workers 1 2 4 6`, and `--asr-batches 8 16 24 32`. Include eight workers explicitly to test that count. Each TTS benchmark creates a fresh local run so the measured time reflects new generation.

Set parallelism with `tts.worker_count` and leave `tts.batch_size` at 1 for Kokoro. One worker uses the serial path. Two or more workers use `multiprocessing.get_context('spawn')`. The parent fetches the pinned model and voice assets, then each worker loads one local KModel shared by its Japanese and English pipelines. Workers use one Torch CPU thread each. The queue orders utterances by descending Unicode character count as a proxy for processing time, with ties resolved by pair ID, language, and attempt. Each worker converts 24 kHz audio to 16 kHz and writes the WAV atomically. The queue carries task and result metadata, and source text remains intact.

The parent matches results to records and checkpoints every 100 utterances. A worker crash or fatal CUDA error saves received results and exits the CLI with an error. A subsequent run regenerates audio missing from the checkpoint and verifies hashes for audio it reuses. The worker pool exits before Whisper loads.

Whisper tuning reuses the same TTS audio and runs batch sizes 8, 16, 24, and 32 in fresh spawned subprocesses. It records out-of-memory candidates as `status: oom` and proceeds after the failed process exits. Rerun TTS tuning if the runtime lost the benchmark audio. The parent fetches the pinned ASR model and processor, and candidates use the offline cache. Run both stages from the same Git commit. If a commit is missing or has changed, the CLI stops before launching ASR candidates, preserves the existing report, and asks for a new TTS benchmark.

Recommendations rank successful candidates by `audio_seconds / wall_seconds`. Candidates with a headroom warning above 90% VRAM, generation failures, or quality differences are excluded. TTS compares text, voice, and duration differences greater than 0.25 seconds or 5%. Whisper compares transcripts, metrics, and QC decisions on identical audio. A change in any of those three excludes the ASR candidate. Run the 1,000-pair end-to-end benchmark with the selected settings to evaluate quality before production.

The report separates allocated and reserved peaks per worker, the largest worker peak, and the sum as a reference. That sum combines peaks from different times and omits CUDA contexts. Benchmarking also samples overall GPU usage with `nvidia-smi` about every five seconds. Brief peaks can fall between samples. Production records `gpu_utilization` as `null`. Benchmark wall time includes model loading, worker startup, and local WAV I/O. Per-attempt TTS RTF uses inference time.

Apply `recommended_tts_workers` and `recommended_asr_batch_size` manually after reviewing the report. The baseline configuration uses one worker and ASR batch size eight. Provenance for new runs records `initial_performance.kokoro_worker_count`. Each invocation records the worker count, ASR batch size, phase throughput, and VRAM. Changing worker count alone allows valid checkpoint audio to be reused. ASR batch size is part of provenance, so changing it in an existing output directory causes a mismatch. Confirm the settings with a separate 1,000-pair benchmark and keep them fixed for production.

Implementation and historical validation details are in the [L4 implementation report](kokoro-l4-report.md).

## Voices and reproducibility

The default voices are Japanese `jf_alpha` and English `af_heart`, each at speed 1.0. They are distinct speakers. The pipeline pins official `kokoro==0.9.4` and model revision `f3ff3571791e39611d31c381e3a41a3af07b4987` and fetches the voice tensors from the same revision. The Japanese and English KPipeline instances share a KModel. The pipeline concatenates all nonempty chunks in order. An utterance with zero nonempty chunks is a generation failure.

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml voice-samples --output-dir /content/drive/MyDrive/ja-en-direct-s2st-corpus/kokoro-voice-samples --en-voices af_heart af_bella am_adam
```

Japanese samples use a fixed sentence with `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, and `jm_kumo`. Samples are saved separately from production data. Choose the voices before the smoke tests. Changing voice, speed, model, input, or QC settings in an existing output directory triggers a `provenance mismatch`. Restore the original settings or choose a new output directory.

## Storage and resuming

```text
MyDrive/ja-en-direct-s2st-corpus/
├── kokoro-jparacrawl-l4-config.yaml
├── kokoro-logs/                    # Streaming execution logs
├── kokoro-environment/             # pip freeze and implementation commit
├── kokoro-l4-smoke/pairs-{10,100,1000}/production/
└── kokoro-jparacrawl-l4/           # L4 production output
    ├── provenance.json
    ├── audio/16k/{ja,en}/shard-.../*.a0.wav
    ├── manifests/generated/       # In-progress checkpoints
    ├── manifests/qc/              # Completed shards
    ├── failures/
    ├── reports/invocations/        # Timing and GPU measurements for each invocation
    └── manifests/releases/{train.jsonl,dev.jsonl,test.jsonl,accepted.jsonl,all.jsonl,summary.json}
```

After a disconnect, mount Drive, restore the repository, dependencies and configuration, validate the input and GPU, check status, and run the next shard. A new runtime needs a fresh virtual environment and model load. The pipeline reuses audio with a valid checkpoint and continues pending QC. Orphaned WAVs are regenerated. Corrupt or empty WAVs and format, hash, duration, or metadata mismatches cause resume to fail. Previously recorded duration rejections retain their QC status.

`status` checks IDs and provenance in completed manifests. Shard resume verifies WAV content, as does consolidation in `full` mode. Run one writer per output directory. The default output contains 16 kHz WAVs. For a debug run that needs native 24 kHz audio, set `save_native: true` and use a separate output directory.

Each CLI process reads the external JSONL once to compute the SHA-256 digest and build a local SQLite index. Export validation and run identity are shared within the process. Shard lookup uses `WHERE shard = ?` and status queries pair IDs from that index. Planning, smoke record validation, and release export also reuse the index and preserve the input text, order, and shard assignments.

Colab stores the index under `/content/s2st-input-index/`. Set `S2ST_INPUT_INDEX_DIR` to override the directory. The fallback is the operating system's temporary directory. A new process revalidates the input and builds a new index. Each shard checks the manifest and summary's file attributes and stops if it detects a change. Consolidation rechecks the SHA-256 digests of the manifest, summary, and any smoke subset immediately before publication. `run-shard`, `run-next-shard`, and `run-remaining-shards --skip-consolidate` also recheck input hashes at exit.

## Reading quality and performance results

Whisper QC requires Japanese CER ≤ 0.10, English WER ≤ 0.10, and audio durations of 0.7–30 seconds. Kokoro uses zero content retries. `[done]` means the shard completed processing. Check its accepted and rejected counts separately. Pass the `accepted.jsonl` containing train and dev to `alignment-transfer-s2ut`, which uses the `split` field to assign training and validation records.

Kokoro production defaults to `checkpoint_every: 100`, counting utterances across both languages. For example, 50 pairs contain 100 utterances. The checkpoint interval is independent of ASR batch size. Checkpoints are written every 100 utterances, when a language or phase finishes, on caught exceptions, and at shard end. Batch results stay in memory between saves. Resume reuses validated checkpoint WAVs and regenerates orphans. To adopt this interval in an older Drive configuration, set `run.checkpoint_every` to 100. Qwen profiles have their own checkpoint settings.

Each CLI process scans the output and cache directories on its first storage check. Subsequent atomic WAV, checkpoint, and report replacements update the budget by their byte-size deltas. A new process or a manual `preflight` takes a fresh baseline. Delta accounting assumes a single writer per output directory.

Use `checkpoint_writes` (attempted saves), `checkpoint_write_seconds`, `directory_scan_count`, and `directory_scan_seconds` to inspect I/O. `current_run_written_bytes` includes replacement writes. `duration_failures_by_length_bin` and `duration_failures_by_language_and_length_bin` count utterances outside the 0.7–30 second bounds.

`summary.json` reports accepted counts, counts by corpus, `ja_hours` and `en_hours`, and provenance for inputs and generation settings. For each language, `attempt_metrics` reports total generated duration, TTS inference time, RTF, and mean CER or WER across evaluated attempts. These are per-attempt averages. The report also counts generation failures, empty audio, duration failures, and ASR failures.

RTF divides TTS inference wall time by generated audio duration. Use `tts_load_seconds`, `asr_load_seconds`, phase times, and `full_pipeline_wall_seconds` in `invocations` to account for loading, WAV conversion, and writing. Each invocation records the GPU and PyTorch peak allocated VRAM. Rates use accumulated wall time from saved successful and failed invocations. A forced disconnect can leave the final invocation unrecorded and make the measured throughput look higher. Estimate compute-unit use from an uninterrupted benchmark's observed consumption and wall time.

After production, run `consolidate` and check for `complete: true` and `missing_shards: []`. Use `consolidate --allow-incomplete` for an interim release.

For external manifests, `consolidate --audio-verification inventory` compares per-shard file listings with the saved QC records. It checks that WAVs exist and relies on the recorded format and SHA-256 rather than reopening audio files. The summary records `audio_verification: inventory`. Use the default `full` mode to revalidate WAV contents, format, and SHA-256. Consolidation can run on CPU.

## Verification

```bash
python -m pytest -q
# Synthesize fixed Japanese and English sentences on CPU
python scripts/check_kokoro_runtime.py --device cpu
```

See the historical implementation and validation notes in the [Kokoro implementation report](kokoro-implementation-report.md) and the [pre-production changes report](kokoro-scale-report.md). JParaCrawl-derived outputs and Kokoro weights have separate terms. [Attribution and official terms](../ATTRIBUTION.md)
