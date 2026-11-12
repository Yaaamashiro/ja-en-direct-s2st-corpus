# JParaCrawl + Kokoro implementation report

This report records the initial implementation. The later split releases, I/O reductions, streaming consolidation, and stratified benchmark are described in the [pre-production changes report](kokoro-scale-report.md).

Validation date was 2026-10-31 (JST). At that stage, validation with real JParaCrawl input on a Colab T4 and the 300,000-pair production run were still pending.

## Git

- branch: `feature/kokoro-tts`
- Base commit was `efe9879`, the starting HEAD.
- See the feature branch history for the implementation commit.
- At the time of this report, the work was on the feature branch. The tracked files were clean at the start. The existing `.test-tmp-final/` directory had a permission warning and was left in place.

## Changed files

Added:

- `configs/production-kokoro-jparacrawl.yaml`
- `requirements/kokoro-colab.txt`
- `notebooks/colab_kokoro_jparacrawl.ipynb`
- `src/s2st_corpus/external.py` for export validation and provenance
- `src/s2st_corpus/kokoro_backend.py` for the official runtime, shared model, chunk concatenation, and voice samples
- `src/s2st_corpus/integrity.py` for checkpoint and audio validation
- `src/s2st_corpus/metrics.py` for RTF, QC, and failure counts
- `scripts/check_kokoro_runtime.py` for a fixed-sentence check with the real model
- `tests/test_external_kokoro.py`
- This report

Modified:

- `config.py`, `corpora.py`, `runner.py`, `smoke.py`, `cli.py`
- `README.md`, `ATTRIBUTION.md`

The existing Qwen configuration, notebook, and outputs remained available.

## JParaCrawl input

Config defaults:

```text
manifest: /content/drive/MyDrive/jparacrawl-filtering/outputs/text-corpus/manifest.jsonl
summary:  /content/drive/MyDrive/jparacrawl-filtering/outputs/text-corpus/summary.json
total pairs: 300000 (expected; not measured)
train: 298000 (expected)
dev: 2000 (expected)
test: 0 (expected)
SHA256 validation: implemented; real export not yet validated
```

The input was available only on Drive. Inspection of `jparacrawl-filtering/outputs` found `selected`, `stats`, and `audit`. The `text-corpus` directory and `manifest.jsonl` were still unavailable, so the counts above remained expectations. Validation used exported-text fixtures.

Fixtures covered the summary digest, counts, completion flag, and corpus, along with duplicate IDs, empty Japanese or English text, splits, length bins, unsafe IDs, and mismatched smoke subsets. Tests also checked that the export text bypasses transformations such as NFKC normalization.

## Kokoro

```text
package version: 0.9.4 (official package, actual inference tested)
model: hexgrad/Kokoro-82M
revision: f3ff3571791e39611d31c381e3a41a3af07b4987
JA voice: jf_alpha
EN voice: af_heart
speed: 1.0
production device: cuda (no silent CPU fallback)
locally tested device: CPU, torch 2.7.1+cpu, Windows / Python 3.12
native sample rate: 24000
canonical sample rate: 16000 / mono / PCM16
Japanese dictionary: UniDic 3.1.0+2021-08-31
```

The Windows machine lacked a C++ build environment, so building `pyopenjtalk==0.4.1` failed. The local check used `pyopenjtalk-plus==0.4.1.post9`. Colab uses a dedicated virtual environment with `misaki[ja,en]==0.9.4` and standard `pyopenjtalk==0.4.1`. A fresh Colab installation remained to be tested.

The notebook downloads UniDic using the official dictionary identifier because the Python package and dictionary are distributed separately. Provenance includes the dictionary hashes. The measurements below use the full dictionary.

## Smoke

The 10-, 100-, and 1,000-pair real-data tests were pending because the export and a Colab T4 runtime were unavailable. Corpus acceptance counts, Japanese CER, English WER, ASR time, end-to-end time, peak GPU VRAM, and corpus RTF remained unmeasured.

Local validation used the real Kokoro model to synthesize one fixed sentence in each language on CPU. This check covered TTS only. Whisper QC and corpus-level quality still required separate validation.

| Fixed-sentence TTS on CPU | Japanese | English |
| --- | ---: | ---: |
| Generated seconds | 3.875 | 3.800 |
| Generated hours | 0.0010764 | 0.0010556 |
| TTS wall-clock seconds | 1.600 | 1.260 |
| RTF | 0.413 | 0.332 |

- Total TTS wall time was 2.860 seconds for 7.675 seconds of audio, giving an RTF of about 0.373.
- Warm-cache model and G2P load took 2.472 seconds, measured separately.
- Real output files confirmed nonempty 24 kHz native float and 16 kHz mono PCM16 audio with valid durations.
- Local outputs were `data/kokoro-runtime-unidic31/report.json`, `ja-16k.wav`, and `en-16k.wav`.
- Windows phonemizer emitted one cleanup warning at exit. Generation, writing, and verification returned code 0. The warning remained unverified on Linux and Colab.
- Use a Colab benchmark to estimate compute units and time for 300,000 pairs.

## Resume

```text
checkpoint: verified with real WAV I/O and simulated inference
existing WAV skip: verified (no second TTS model load)
interrupted run resume: verified via injected QC interruption
corrupt/0-byte WAV: explicitly rejected on resume and consolidate
text/input/voice/model/speed/config mismatch: refused
actual Colab disconnect recovery: not yet tested
```

The pipeline verifies selected audio and attempts in the accepted manifest. It regenerates orphaned audio without a checkpoint and records ordinary QC failures without a content retry. Run one writer per output directory.

## Tests

- Local pytest passed 56 tests, including existing Qwen, JESC, and KFTT coverage.
- Tests covered input validation, existing profiles, shared KModel, pinned voices, multiple and empty chunks, audio conversion, resume, corruption rejection, the production gate, and Python syntax in the notebook.
- Automated inference tests used mocks. The fixed-sentence check exercised the real model separately.
- The run reported two existing tar-extraction deprecation warnings.

## Checks pending before production

1. Generate the text export in `jparacrawl-filtering` or point to an existing export, then verify the full SHA-256 digest and record counts.
2. Set up the dedicated Colab notebook and run 10- and 100-pair TTS plus Whisper QC on a T4. Check CER, WER, voices, failure reasons, and VRAM.
3. Use an uninterrupted 1,000-pair benchmark if needed. Estimate production cost from observed compute-unit consumption and wall time. Audio duration depends on the input and QC results.
4. Use saved invocations to track full wall time. A forced disconnect can lose the last report and undercount elapsed time. Compare inference-only TTS RTF and phase wall time separately.
5. Review the JParaCrawl terms and confirm voices, speed, and output location. See the official links in `ATTRIBUTION.md`.
6. Keep `APPROVE_PRODUCTION=False` until ready to start production.
