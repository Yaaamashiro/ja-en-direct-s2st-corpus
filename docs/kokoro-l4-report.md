# L4 Kokoro multiprocessing and Whisper tuning report

## Git

- branch: `feature/kokoro-tts`
- base commit: `2cf2814`
- This report captures the feature-branch implementation before the full production run.

## Parallel implementation

| Area | Implementation |
| --- | --- |
| Kokoro multiprocessing | A dedicated pool handles two or more workers. One worker uses the serial baseline. |
| Start method | Workers use `spawn`. A `fork` context is rejected. |
| Worker model loading | The parent fetches and verifies pinned assets. Each worker loads a local KModel once and shares it across Japanese and English KPipeline instances and voices. |
| Task scheduling | A bounded queue starts longer utterances first by Unicode character count. Ties follow pair ID, language, and attempt. Duplicate task or result keys are rejected. |
| WAV writing | Each worker converts 24 kHz audio to mono, resamples with soxr HQ, and writes 16 kHz PCM16 WAV atomically. Waveforms stay in the worker. |
| Parent checkpoints | The parent maps result metadata to records and checkpoints every 100 utterances, at phase end, and on exceptions. |
| Resume | Validated checkpoint WAVs are reused. Orphaned WAVs are regenerated and corrupt files are rejected. |
| Failure | A worker exception, fatal CUDA error, or process crash triggers a checkpoint and CLI failure. Recovery occurs on a subsequent run. |
| Phase boundary | The pool closes and joins before Whisper loads. TTS and ASR run in separate phases. |

Each worker uses one Torch CPU thread. Workers report the byte-size delta of atomic WAV replacements so the parent can update the storage budget. On Colab, SIGTERM triggers pool cleanup, which joins children and kills them if needed. A forced kill can leave the final checkpoint unwritten. The next run regenerates audio missing from the checkpoint.

The audio identity excludes worker count. New runs record `initial_performance.kokoro_worker_count`, and each invocation records the worker count and ASR batch actually used. Changing only the worker count allows valid saved audio to be reused. Provenance still checks voice, revision, speed, input, QC settings, and ASR batch size.

## L4 profile

```text
config: configs/production-kokoro-jparacrawl-l4.yaml
target GPU: L4
minimum VRAM: 20 GiB
CUDA required: true
BF16 required: false
Kokoro: float32
Whisper: float16
candidate workers: 1,2,4,6 (8 is optional, explicit only)
candidate ASR batches: 8,16,24,32
```

A GPU other than the target generates a warning with both GPU names. A T4 fails the minimum VRAM check. The L4 profile has its own configuration and production and smoke output paths. The earlier Kokoro and Qwen profiles are also available.

## Performance benchmark

The benchmark command accepts `--stage tts`, `--stage asr`, or `--stage all`.

- Select 200 training pairs using seed 42 and length-bin quotas of 4/30/80/66/20. All candidates use the same records and text.
- Store WAVs and checkpoints on the local SSD under `/content/kokoro-l4-benchmark`. Benchmark audio requires a local output path.
- Write the report, timestamped snapshots, and logs to Drive under `kokoro-performance/`. The latest report is `l4-performance.json`.
- Run each candidate in a fresh spawned subprocess. Record ASR out-of-memory failures as `status=oom`, then try the next candidate after that subprocess exits.
- Use audio from the first fully successful TTS candidate for all ASR trials. Verify WAV SHA-256 hashes before and after each trial.
- The parent fetches the pinned Kokoro and Whisper revisions. Candidates use cached assets offline.
- Compare quality under the same conditions. For ASR, require matching transcripts, metrics, and QC. For TTS, check text, voice, and substantial duration differences. Cross-worker ASR quality on real L4 output was still pending.
- Require the same Git commit for TTS and ASR tuning. Missing or mismatched commits stop the run before launching ASR candidates and preserve the prior report. `--stage all` rechecks the commit before ASR.
- Recommend the eligible candidate with the highest throughput. Insufficient VRAM headroom, generation failure, or quality differences disqualify a candidate. An empty candidate set yields `null`.
- Apply the recommended settings manually. Production uses the supplied configuration without autotuning. A separate performance lock was outside the implementation scope.

The benchmark records generated audio duration, wall time, RTF, pairs and utterances per second, generation and ASR failures, empty audio, out-of-memory errors, GPU name, and per-process allocated and reserved memory peaks. Benchmark wall time includes loading, worker startup, and local WAV I/O.

The report lists each worker's PyTorch peak, the sum as a reference, and the largest worker peak. During benchmarks, `nvidia-smi` samples total GPU memory about every five seconds. Recommendations exclude candidates above 90% VRAM using both observed usage and the reserved-memory reference. Short peaks may occur between samples, so the selected setup still needs a 1,000-pair end-to-end benchmark. Production reports `gpu_utilization` as `null`.

## Notebook

```text
Drive / repo / deps / L4 config / validation / GPU / preload / voice
→ 10 pairs → 100 pairs
→ TTS tuning → ASR tuning → recommendation
→ manual config confirmation
→ 1000 pairs full benchmark
→ plan/status → production 1 shard → remaining shards → consolidate
```

The default flags are `RUN_L4_TUNING=False`, `RUN_BENCHMARK_1000=False`, `PERFORMANCE_CONFIG_CONFIRMED=False`, `APPROVE_PRODUCTION=False`, and `RUN_ALL_REMAINING=False`. The notebook saves the L4 configuration to `kokoro-jparacrawl-l4-config.yaml` on Drive and reuses it on subsequent runs.

## Tests

- Local pytest reported **125 passing tests**, including the previous 97 and follow-up review coverage. It emitted two existing tar-extraction deprecation warnings.
- Real spawned processes with FakeBackend exercised 20 fixed utterances at 1, 2, and 4 workers. Tests checked one result per task, path and record mapping, voice, 16 kHz mono PCM16, duration, SHA-256, and atomic writes.
- Fault tests injected worker CUDA errors, a real process exit with code 17, and empty generation. Tests also covered parent checkpoints, 100-utterance intervals, reversed completion order, resume, orphan regeneration, and storage deltas.
- Tests covered the L4 profile, T4 rejection, warnings for other GPUs, recommendations based on throughput and 90% VRAM headroom, and exclusion of candidates with ASR quality differences.
- Mocked candidates exercised 200-pair selection, reuse of the same sample and audio, progression after OOM, JSON reporting, and preservation of configuration.
- Batch sizes 8, 16, 24, and 32 produced identical metrics and QC from identical mocked transcripts. Real Whisper comparisons on L4 were still pending.
- Follow-up tests covered longest-first scheduling, deterministic ties, exact source text, rejection of transcript-only differences, same-commit ASR resume, and rejection of missing or different commits while retaining the existing report.
- A separate serial Kokoro CPU check produced 3.875 seconds of Japanese and 3.800 seconds of English from fixed sentences and verified both WAV formats. The report is at `data/kokoro-l4-runtime-check/report.json`. The known Windows phonemizer cleanup warning appeared and the process returned 0.

## Production defaults

```text
kokoro worker_count: 1
asr batch_size: 8
checkpoint_every: 100 utterances
num_shards: 128
max_pairs_per_shard: 3000
```

These defaults provided a baseline before measuring on an L4. Optimal worker and ASR batch sizes remained to be measured.

## Validation pending at the time

- Colab L4 10 pair smoke
- Colab L4 100 pair smoke
- L4 worker tuning
- L4 ASR batch tuning
- 1,000-pair end-to-end benchmark with the selected settings, checking CER, WER, duration, RTF, VRAM, and compute units
- production 1 shard
- 300k production

These checks were planned before production. The implementation preserved the input index, exported text, train/dev split, pinned voice and revision, QC thresholds, 0.7–30 second duration bounds, WAV integrity checks, streaming releases, and canonical ordering.

## Length-bin shortfall handling

- An L4 run found one training pair in bin 0, below the TTS quota of four. Sampling stopped before GPU work or model loading.
- TTS tuning now redistributes shortages deterministically to keep 200 training pairs, in input order with the original text. All candidates use the same sample.
- The 1,000-pair smoke-test CLI uses strict quotas by default. `--benchmark-shortfall redistribute` enables redistribution. The notebook documents and sets this option.
- Selection metadata records requested and actual quotas, available training counts, and the policy. An insufficient total causes the command to stop and preserve the existing sample.
- After this fix, `pytest -q --basetemp=.venv/test-quota-shortfall-full-01` reported 130 passing tests and two existing tar-extraction deprecation warnings. L4 remeasurement was still pending.
