# Production corpus runbook

## Full run

```powershell
docker compose run --build --rm corpus
```

The default `build` command downloads and selects JESC/KFTT data, generates and checks audio for all 128 shards with Whisper, and writes the release manifests. Rerun it with the same configuration and data root to reuse completed manifests, audio, and shards.

## Five-pair smoke test on the production GPU

Before production, run Qwen synthesis, Whisper quality checks, and release export on five pairs from JESC and KFTT.

```powershell
docker compose run --build --rm corpus smoke-test
```

The sample includes one short, one medium, and one long JESC pair and one medium and one long KFTT pair. It uses training pairs with complete sentences in both languages. If a category has too few pairs, the sampler fills the gap deterministically from the same training split. The test shares the production manifest and official archives and writes its own input, audio, and manifests under the following directory.

```text
data/smoke/
├── input/pairs.jsonl
└── production/
    ├── audio/16k/{ja,en}/shard-00000-of-00001/
    ├── manifests/releases/accepted.jsonl
    ├── manifests/releases/all.jsonl
    └── manifests/releases/summary.json
```

In `summary.json`, `total_pairs: 5` confirms that all five pairs were processed. `accepted_pairs` counts pairs that passed QC in both languages. Inspect the audio and `all.jsonl` for failed attempts and rejection reasons. Rerunning the command reuses completed smoke-test work.

## Requirements

- Windows with Docker Desktop and the WSL2 NVIDIA backend, or Linux with NVIDIA Container Toolkit
- NVIDIA GPU with BF16 support
- At least 16 GiB of VRAM
- About 16 GB for the Docker image
- About 8 GB for the Hugging Face model cache
- At least 80 GiB in the corpus data root

To store the corpus on an external SSD, set the data root before running Docker Compose.

```powershell
$env:CORPUS_DATA_ROOT = "D:\ja-en-direct-s2st-corpus"
docker compose run --build --rm corpus
```

## Data preparation

The pipeline downloads the official archives and verifies their configured SHA-256 digests.

| Corpus | Archive | SHA-256 |
|---|---|---|
| JESC | `split.tar.gz` (2019-05-12) | `9cc6f2b31225d84b204a1bc4599e5fbb158d5739ba4aa76eab30fea05ed8ccd5` |
| KFTT | `kftt-data-1.0.tar.gz` | `fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638` |

Extraction rejects absolute paths, `..`, symbolic links, hard links, and device files. The pipeline reads JESC from English–Japanese TSVs and KFTT from separate language files. It registers dev and test records from both corpora first, so any matching pair is excluded from train.

The training selection applies the following checks and rules.

- Normalize text with Unicode NFKC and handle HTML entities, whitespace, control characters, and zero-width characters.
- Exclude URLs, email addresses, markup, sound-effect-only subtitles, and empty lines.
- Check the proportion of Japanese characters and English Latin characters.
- Apply corpus-specific length bounds and check the Japanese-to-English length ratio.
- Remove exact duplicates using SHA-256 hashes of normalized sentence pairs.
- Select reproducibly with a fixed seed.
- Target a 70:30 JESC-to-KFTT ratio by estimated Japanese audio duration.
- Target a 20:50:30 duration ratio across short, medium, and long strata. Fill shortfalls deterministically from the remaining candidates in the same corpus.

Preparation writes these files.

```text
data/input/pairs.jsonl
data/reports/preparation/prepare-summary.json
```

Extracted data and the selection database use temporary storage inside the container. The finished manifest and audit report are saved to the data root.

## Individual commands

Use individual commands when diagnosing a particular stage.

Download and select data.

```powershell
docker compose run --build --rm corpus prepare
```

Check the manifest and storage estimate.

```powershell
docker compose run --rm corpus plan
```

Generate a specific shard.

```powershell
docker compose run --rm corpus run-shard --shard-index 0
```

Consolidate after all shards finish.

```powershell
docker compose run --rm corpus consolidate
```

Export an interim release from completed shards.

```powershell
docker compose run --rm corpus consolidate --allow-incomplete
```

## Shard processing

Each shard follows this sequence.

```text
Qwen attempt 0
  → Release Qwen
  → Whisper QC
  → Qwen attempt 1 for languages that failed QC
  → Release Qwen
  → Whisper QC
  → Select the best attempt for each language
```

The pipeline writes an atomic manifest checkpoint every 10 utterances. A CUDA out-of-memory error or device-side assert can leave the CUDA context unusable, so the container exits with an error. Check the configuration and GPU, then rerun the same command.

## Fixed settings

- Qwen：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Speaker for both languages is `Ono_Anna`.
- TTS：CUDA BF16
- ASR：`openai/whisper-large-v3-turbo`
- Japanese reading-normalized CER must be 10% or less.
- English Whisper-normalized WER must be 10% or less.
- Audio duration must be 0.7–30.0 seconds.
- Each language gets one regeneration attempt after a content failure.

Keep the model revision, speaker, dependencies, seed rules, and QC thresholds consistent throughout a run. Use a new `CORPUS_DATA_ROOT` when changing these settings so the new run can prepare its own manifest.

## Outputs

```text
data/production/
├── audio/16k/{ja,en}/shard-xxxxx-of-00128/
├── manifests/generated/
├── manifests/qc/
├── manifests/releases/
│   ├── all.jsonl
│   ├── accepted.jsonl
│   ├── train.jsonl
│   ├── dev.jsonl
│   ├── test.jsonl
│   └── summary.json
└── failures/
```

`train.jsonl`, `dev.jsonl`, and `test.jsonl` contain the accepted pairs for each split, for training, validation, and evaluation respectively. For `alignment-transfer-s2ut`, use `accepted.jsonl`, which includes the train and dev splits in one file. `all.jsonl` includes rejected pairs for auditing. The pipeline publishes `summary.json` last as the completion marker. It reports counts and acceptance rates by split, Japanese and English audio duration, and counts by corpus. Release manifests preserve the original input order.

At 16 kHz mono PCM16, 300 hours in each language takes approximately 64.37 GiB. With a 10% regeneration allowance, the estimate is 70.81 GiB. Budget separately for the model cache, Docker image, and any native WAV files. The standard run saves the 16 kHz WAVs.
