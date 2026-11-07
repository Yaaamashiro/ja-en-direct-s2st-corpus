# Japanese–English Direct S2ST Corpus

A pipeline for building paired Japanese and English speech for direct speech-to-speech translation (S2ST). It synthesizes both sides of parallel text, checks speech quality by comparing Whisper transcriptions with the source text, and exports accepted pairs in their original input order.

**[Open Kokoro in Colab](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_kokoro_jparacrawl.ipynb)** · [Kokoro guide (Japanese)](docs/kokoro-runbook.md) · [Data and model attribution](ATTRIBUTION.md)

## Corpus results

Results from a production run using Kokoro-82M and 300,000 selected text pairs from JParaCrawl v3.

| Metric | Result |
| --- | ---: |
| Text pairs processed | 300,000 |
| Pairs accepted by quality control | 217,765 (72.59%) |
| Accepted pairs by split | Train 216,312 / dev 1,453 |
| Accepted audio | Japanese 686.73 hours / English 603.55 hours |

This repository contains the pipeline code and configuration files. Generated audio and manifests are saved to the Google Drive or local disk used for the run.

## Getting started

| Profile | Input and speech synthesis | Resources |
| --- | --- | --- |
| **Kokoro + JParaCrawl** | Selected JParaCrawl v3 text · Kokoro-82M | [Colab](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_kokoro_jparacrawl.ipynb) · [Configuration](configs/production-kokoro-jparacrawl-l4.yaml) |
| Qwen + JESC/KFTT | JESC and KFTT · Qwen3-TTS | [Colab](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_pro.ipynb) · [Docker and operations guide (Japanese)](docs/qwen-runbook.md) |

The Kokoro profile reads `manifest.jsonl` and `summary.json` exported by `jparacrawl-filtering`. In Colab, select an L4 GPU, mount Google Drive, and start with the 10-pair test to check audio quality and quality-control (QC) results. The notebook's opening table covers the 100-pair test, production runs, resuming interrupted work, and final release export. Final export can run on a CPU runtime.

Progress is saved in shards. To resume an interrupted run, use the same input, configuration, and output directory.

## Outputs

| Path | Contents |
| --- | --- |
| `audio/16k/{ja,en}/` | 16 kHz mono PCM16 WAV files |
| `manifests/releases/accepted.jsonl` | All QC-accepted pairs, including train and dev splits |
| `manifests/releases/{train,dev,test}.jsonl` | QC-accepted pairs for each split |
| `manifests/releases/all.jsonl` | All pairs and QC records, including rejected pairs |
| `manifests/releases/summary.json` | Completion status, pair counts, audio duration, and verification mode |

Check for `complete: true` in `summary.json` before using a release. **Pass `accepted.jsonl` to `alignment-transfer-s2ut`**, which uses the train and dev splits from the same manifest for training and validation.

## Development and licensing

- [Tests and runtime checks (Japanese)](docs/kokoro-runbook.md#検証)
- [Qwen commands and operations (Japanese)](docs/production-runbook.md)
- [MIT license for the source code](LICENSE)
- [Attribution and terms for datasets, models, and derived outputs](ATTRIBUTION.md)
