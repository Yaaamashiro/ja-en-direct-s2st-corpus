# Qwen + JESC/KFTT runbook

[Back to README](../README.md)

## Profile

This profile synthesizes paired Japanese and English audio from official JESC and KFTT data using the same speaker for both languages.

- Training text combines 70% filtered JESC and 30% KFTT by estimated Japanese audio duration.
- TTS：Qwen3-TTS 1.7B CustomVoice / `Ono_Anna`
- Audio uses 16 kHz mono PCM16 WAV.
- Quality control uses OpenAI Whisper large-v3-turbo.
- The target is about 300 hours of training audio plus the official dev and test splits.

## Requirements

- NVIDIA GPU with BF16 support and at least 16 GiB VRAM
- Docker with NVIDIA GPU access
- At least 1 GiB for the five-pair smoke test and 80 GiB for the production corpus
- At least 8 GiB for the Hugging Face cache

## Running the pipeline

Start with a five-pair smoke test on the production GPU.

```powershell
docker compose run --build --rm corpus smoke-test
```

The test selects three JESC pairs and two KFTT pairs across sentence lengths and generates ten utterances using production Qwen, Whisper, and QC settings. Smoke output goes to `data/smoke/`, separate from `data/production/`.

```text
data/smoke/production/
├── audio/16k/{ja,en}/
└── manifests/releases/{accepted,all}.jsonl
```

After reviewing the smoke-test results, generate the production corpus with the following command.

```powershell
docker compose run --build --rm corpus
```

The command runs the following stages in order.

1. Download the official JESC and KFTT archives and verify their SHA-256 hashes.
2. Extract safely, normalize text, and filter by language, length, and symbols.
3. Remove exact duplicates across splits, select at a 70:30 corpus ratio, and stratify by length.
4. Generate Japanese and English audio with the same Qwen3-TTS voice.
5. Check Japanese CER and English WER with Whisper and retry failed content where applicable.
6. Consolidate all shards and publish the release manifests.

### Google Colab Pro

[Open the notebook in Colab](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_pro.ipynb)

Colab uses its GPU runtime directly. Audio and progress are saved to Google Drive so a new runtime can resume from the saved state. [Colab configuration](../configs/colab-pro.yaml) · [Notebook source](../notebooks/colab_pro.ipynb)

The Colab profile divides production into 512 shards. Each shard generates Japanese and English audio, checks quality, and regenerates failed content when applicable before saving its decisions. Use the same `num_shards`, input manifest, and output path throughout a run, with one writer per output directory.

#### First-time setup

Select a GPU runtime in Colab. Run the following cells in order and review the smoke-test result before starting production.

| Cell | Action |
|---|---|
| 1. Mount Google Drive | Authorize Drive access and set the audio and progress directory. |
| 2. Fetch the code | Clone or update the repository and set the configuration path. |
| 3. Install dependencies | Prepare Python and pinned dependencies in each fresh runtime. |
| 4. Check the GPU | Verify GPU access, VRAM, and Qwen imports. |
| 5. Set batch sizes | Set the number of utterances processed per batch. |
| 6. Run a five-pair smoke test | Prepare JESC/KFTT and generate and check five Japanese–English pairs. |

This setup was run with Python 3.13 and an A100 80 GB. The GPU check requires at least 14 GiB of VRAM and selects BF16 when supported or FP16 otherwise. The Colab production limit is 160 GiB, including accepted audio, retry and rejected audio, and manifests. The storage check adds existing production output to the filesystem's reported free space and requires at least 160 GiB. Check the overall Drive account quota separately. Allow 8 GiB of local model cache and 1 GiB for smoke-test output.

Review `all_pairs_accepted: true` and listen to accepted audio. `returncode: 0` confirms command completion. The QC summary reports acceptance separately. Results go to the separate `smoke/production/` directory.

#### Batch sizes and GPU memory

Edit these values in cell 5 and run the cell to apply them.

```python
TTS_BATCH_SIZE = 72
ASR_BATCH_SIZE = 144
```

Qwen handles text-to-speech (TTS) and Whisper handles automatic speech recognition (ASR). The pipeline releases the TTS model before loading ASR. Batches group utterances within one shard and shards run sequentially.

These values were used on an A100 80 GB. Memory demand varies with the GPU and utterance length. The notebook sets both batch sizes to one when it detects under 70 GiB of VRAM. After an out-of-memory error on a larger GPU, stop the production cell, try 48/96 or 32/64, and rerun the settings cell before resuming.

Use GPU memory during `[tts-batch]` to tune TTS and memory during `[qc-batch]` to tune ASR. Long audio can raise memory use. Compare peak memory, elapsed time, and accepted pairs per compute unit when choosing a batch size.

Cell 5 sets `S2ST_TTS_BATCH_SIZE` and `S2ST_ASR_BATCH_SIZE`. These override `tts.batch_size` and `asr.batch_size` in YAML. The fallback is one for each. TTS uses a seed per batch, so regenerating with a different batch composition or size can change the waveform. The run records the seed and pair IDs in each batch.

#### Production

Check the completed count in cell 7, then run cell 8. Choose `False` to process and measure one shard or `True` to run all remaining shards in order.

```python
RUN_ALL_REMAINING = True  # False processes the next unfinished shard only.
```

The all-shards option runs `run-remaining-shards`. The single-shard option runs `run-next-shard`. Both skip completed shards and reuse saved generation and QC from a partial shard. The all-shards option also consolidates after all 512 shards finish. If execution fails, check the error before resuming.

Example progress output

```json
{"completed_shards": 24, "next_shard": 24, "remaining_shards": 488, "total_shards": 512}
```

Shard numbering starts at zero. This example has completed shards 0 through 23 and will run shard 24 next. A shard counts as complete when its final manifest is saved under `manifests/qc/`.

#### Storage and accepted audio

The default persistent directory is `MyDrive/ja-en-direct-s2st-corpus-data` in Google Drive.

```text
ja-en-direct-s2st-corpus-data/
├── input/pairs.jsonl                 # Prepared Japanese–English text
├── production/
│   ├── audio/16k/{ja,en}/shard-…/    # Initial and regenerated audio
│   ├── manifests/generated/         # In-progress checkpoints
│   ├── manifests/qc/                # Final shard decisions
│   ├── manifests/releases/
│   │   ├── accepted.jsonl           # Accepted pairs at export
│   │   ├── all.jsonl                # All pairs at export
│   │   └── summary.json             # Acceptance and audio duration
│   └── failures/                    # Rejected pairs and attempts
├── smoke/production/                # Smoke-test audio and results
├── logs/                            # Production execution logs
├── reports/preparation/             # Text preparation reports
└── sources/                         # Source corpora
```

`.a0.wav` is the initial attempt and `.a1.wav` is a regeneration. The `ja_wav_16k` and `en_wav_16k` fields in `accepted.jsonl` identify the selected files. For each language the pipeline chooses the passing attempt with the lowest error rate. A pair is accepted when both languages pass.

Models are cached under `/content/huggingface` and `/content/s2st-models` and fetched again in a fresh runtime. Smoke logs go to `/content/s2st-smoke-*.log`. Production logs are saved to Drive.

#### Reading logs and QC results

| Log | Meaning |
|---|---|
| `[preflight]` | GPU, pair count, and effective batch sizes. `vram_gib` is total GPU capacity. |
| `[tts] loading` / `[asr] loading` | Loading the synthesis or transcription model. |
| `[tts-batch] 72/946 ja attempt=0` | 72 of 946 utterances processed for generation. `ja` is Japanese and `en` is English. |
| `[qc-batch] 144/946 ... seconds=...` | QC progress. `seconds` measures the ASR call for that batch. |
| `[retry] content failures=284` | 284 utterances lack a passing attempt. Japanese and English count separately. |
| `[done] shard=6 pairs=485 accepted=306 rejected=179` | Shard 6 completed with 306 accepted and 179 rejected pairs. |
| `[run-all] completed=7/512 remaining=505` | Progress after the last completed shard. |
| `returncode: 0` | The command completed. Check the release summary for an all-shards run. |
| `Traceback` / `returncode: 1` | The command failed. Read the exception at the end of the traceback. |

QC requires Japanese character error rate (CER) and English word error rate (WER) of 10% or less and audio duration of 0.7–30 seconds. Audio outside the duration range fails before ASR. An ordinary QC failure allows up to one regeneration. Attempt 0 is the initial audio and attempt 1 is the retry. Resume reuses any saved retry audio.

#### Resuming after a disconnect or compute-unit limit

A resumed run reuses audio and progress already saved to Drive. Work from an in-flight or unsaved batch may need to run again.

1. In a fresh GPU runtime, run setup cells 1 through 4.
2. Run the batch settings in cell 5 to restore the runtime's environment variables.
3. Run cell 7 to see completed shards and the next shard.
4. Run production in cell 8. Reuse a completed smoke test when resuming normally.

Reuse installed dependencies if the same runtime is still active. To fetch current GitHub code, stop production and rerun cell 2. To load updated notebook cells, reopen the notebook from the Colab link.

Save a copy to Drive to keep local cell edits. Publishing a notebook update to GitHub requires a commit and push. Run outputs and checkpoints are saved independently in Drive.

#### Interim releases and cost estimates

Stop production and run cell 9 to export completed shards with `consolidate --allow-incomplete`. Read the saved summary to inspect the result.

```python
import json

summary = json.loads(
    (DATA_ROOT / "production/manifests/releases/summary.json").read_text(encoding="utf-8")
)
print(json.dumps(
    {key: value for key, value in summary.items() if key != "missing_shards"},
    ensure_ascii=False, indent=2,
))
```

`ja_hours` and `en_hours` report accepted audio duration. `complete: false` is expected for an interim release with pending shards. Each consolidation updates the release files.

The approximately 300-hour target estimates Japanese duration during text selection. Check both the completion of all 512 shards and the actual accepted duration after QC.

Estimate remaining usage by multiplying observed compute units and time per completed shard by the number of remaining shards. GPU, text length, and regeneration rate affect the estimate. Manage Colab balance, purchases, and reconnection in Colab.

#### Troubleshooting

- **`output safety limit exceeded`** indicates that the run reached `run.maximum_output_gib`. Older Colab configurations used 78 GiB. If Drive has sufficient capacity, fetch the current configuration with the 160 GiB limit and resume with the existing shards, output path, and audio.
- **`flash-attn is not installed`** is an informational message about an optional library. The profile uses `sdpa`.
- **`operator torchvision::nms does not exist`** suggests a Torch and torchvision mismatch. Notebook setup installs Torch 2.7.1, torchvision 0.22.1, and torchaudio 2.7.1 from the same CUDA 12.6 index.
- **`Could not load libtorchcodec`** can result from incompatible preinstalled TorchCodec. Setup removes TorchCodec, and WAV input uses FFmpeg.
- **`CalledProcessError`** reports a child process failure. Inspect the underlying exception in the production log.
- **`CUDA out of memory`** can be addressed by stopping production, lowering the affected TTS or ASR batch size, rerunning the settings cell, and resuming. An unsaved batch may need to run again.

If an earlier ASR environment error caused all smoke pairs to fail, fix the environment and recheck the saved audio. This command reuses the existing audio and backs up the old reports under `smoke/production/qc-backups/`.

```bash
python -m s2st_corpus.cli --config configs/colab-pro.yaml recheck-smoke
```

### Docker storage and resuming

Progress is saved per shard. Rerun the same command after an interruption to reuse completed work.

Set the data root before running to save to an external SSD.

```powershell
$env:CORPUS_DATA_ROOT = "D:\ja-en-direct-s2st-corpus"
docker compose run --build --rm corpus
```

## Outputs

```text
data/
├── input/pairs.jsonl
├── production/
│   ├── audio/16k/{ja,en}/
│   ├── manifests/releases/accepted.jsonl
│   ├── manifests/releases/all.jsonl
│   └── manifests/releases/summary.json
├── reports/preparation/prepare-summary.json
└── sources/
```

The runtime downloads official archives and caches model assets. Generated audio and manifests are stored under the data root. [Attribution and terms](../ATTRIBUTION.md) · [Individual commands and resume instructions](production-runbook.md)

Across profiles, `all.jsonl` follows the order of `config.run.input_jsonl` and `accepted.jsonl` preserves that order for accepted records, including interim releases. A complete release verifies that the QC and input pair ID sets match and rejects missing, extra, and duplicate IDs. `summary.json` records `release_order: "input_manifest"` and `missing_pair_count`.

Pass `accepted.jsonl` to `alignment-transfer-s2ut` so the trainer can read the train and dev splits from one manifest. `train.jsonl`, `dev.jsonl`, and `test.jsonl` contain the accepted records for each split in input order. A zero-sized test split produces an empty `test.jsonl`. The summary reports total, accepted, rejected, acceptance rate, and accepted audio duration by split. Training duration is reported as `train_ja_hours` and `train_en_hours`.

Consolidation builds a SQLite index on local temporary storage and reads QC about one shard at a time. It checks input metadata, provenance, selected audio, and WAV format, SHA-256, and duration where the profile requires those checks. It stages files as `.tmp`, invalidates the prior summary before replacing the files, and publishes the new `summary.json` last. Check the summary for completion. Allow local disk space for the temporary index of input and QC JSON.
