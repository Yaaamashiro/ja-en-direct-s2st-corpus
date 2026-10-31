# JParaCrawl + Kokoro implementation report

検証日：2026-10-31（JST）。実装済みですが、実JParaCrawl + T4での10/100件full smokeは未実施です。30万件の本番生成は開始していません。

## Git

- branch: `feature/kokoro-tts`
- base commit: `efe9879`（開始時HEAD）
- implementation commit: この文書を含むfeatureブランチの `git log -1` を参照。最終報告にも記載します。
- mainへmergeしていません。開始時に追跡ファイルの未コミット変更はありませんでした。既存 `.test-tmp-final/` には権限警告があり、削除・変更していません。

## Changed files

Added:

- `configs/production-kokoro-jparacrawl.yaml`
- `requirements/kokoro-colab.txt`
- `notebooks/colab_kokoro_jparacrawl.ipynb`
- `src/s2st_corpus/external.py`（export検証・provenance固定）
- `src/s2st_corpus/kokoro_backend.py`（公式runtime・共有モデル・全chunk連結・試聴）
- `src/s2st_corpus/integrity.py`（checkpoint・音声の検証）
- `src/s2st_corpus/metrics.py`（RTF・品質・失敗集計）
- `scripts/check_kokoro_runtime.py`（固定文の実モデル確認）
- `tests/test_external_kokoro.py`
- このレポート

Modified:

- `config.py`, `corpora.py`, `runner.py`, `smoke.py`, `cli.py`
- `README.md`, `ATTRIBUTION.md`

Deleted: なし。既存のQwen config・ノートブック・生成物は変更していません。

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

ユーザー確認では入力はDriveのみ。接続されたDriveの `jparacrawl-filtering/outputs` を読み取り確認したところ、`selected`, `stats`, `audit` は見つかりましたが `text-corpus` と `manifest.jsonl` は見つかりませんでした。上記件数を実データ検証済みと扱っていません。raw/parquetは取得・処理していません。

テスト用fixtureではsummaryのSHA・件数・complete・corpus、重複ID、空の日英文、split、length_bin、危険なID、不一致のsmoke subsetを検査しました。export文字列がNFKC等で変化しないことも検査済みです。

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

WindowsにはC++ビルド環境がなく `pyopenjtalk==0.4.1` のビルドが失敗したため、**ローカル確認のみ** `pyopenjtalk-plus==0.4.1.post9` を使用しました。Colab用は `misaki[ja,en]==0.9.4` と標準 `pyopenjtalk==0.4.1` を使用する別環境です。Colabの新規インストール完走は未確認です。依存は専用venvに導入し、Qwen環境へ重ねません。

UniDic本体はpackage installだけでは入らないため、ノートブックで正式な辞書識別子を指定してダウンロードします。辞書ハッシュもprovenanceに含めます。最初の簡易辞書での確認と本辞書での確認は区別し、以下は本辞書で測った値です。

## Smoke

**実コーパス10件/100件/1000件：未実施。** 入力exportが取得できず、Colab T4への実行接続もありません。accepted/rejected、JA CER、EN WER、ASR時間、full pipeline時間、GPU peak VRAM、実コーパスRTFは未計測です。これらを0や合格として報告しません。

代わりに、固定文の日英各1文について実KokoroのTTS単体を確認しました。Whisper QC・コーパス品質評価を代替するものではありません。

| 固定文TTS単体（CPU） | 日本語 | 英語 |
| --- | ---: | ---: |
| 生成秒数 | 3.875 | 3.800 |
| 生成時間（hours） | 0.0010764 | 0.0010556 |
| TTS wall秒 | 1.600 | 1.260 |
| RTF | 0.413 | 0.332 |

- TTS total wall: 2.860秒、生成合計7.675秒、RTF total ≈ 0.373。
- warm-cache model/G2P load: 2.472秒（別計測）。
- 24 kHz native floatと16 kHz/mono/PCM16の保存・非空・durationを実ファイルで確認。
- local result: `data/kokoro-runtime-unidic31/report.json`、`ja-16k.wav`、`en-16k.wav`（Git対象外）。
- Windows版phonemizerの終了時cleanupで警告が1回出ました。生成・保存・検証のreturn codeは0。Linux/Colabでの同警告発生は確認していません。
- この2文のCPU RTFからColab CU・30万件の所要時間を推定しないでください。

## Resume

```text
checkpoint: verified with real WAV I/O and simulated inference
existing WAV skip: verified (no second TTS model load)
interrupted run resume: verified via injected QC interruption
corrupt/0-byte WAV: explicitly rejected on resume and consolidate
text/input/voice/model/speed/config mismatch: refused
actual Colab disconnect recovery: not yet tested
```

学習用accepted manifestの選択音声とattemptも照合します。checkpointがない孤立音声は再生成します。通常の品質不合格は保持し、同条件でcontent retryしません。同じ出力先への同時書き込みは非対応です。

## Tests

- 既存Qwen/JESC/KFTTを含むpytest：56件成功（最終実行結果は最終報告を参照）。
- 入力validation、既存config回帰、共有KModel、固定voice、複数chunk、空chunk、音声変換、再開・破損拒否、本番gate、ノートブックの全Pythonセル構文を検査。
- CIの推論部分はmock。実モデル確認は上記固定文スクリプトで別実施。
- 既存tar展開に関するPythonのDeprecationWarning 2件あり。今回の変更とは無関係。

## Remaining issues / production前の確認

1. `jparacrawl-filtering` 側でtext corpus exportを生成するか、既存exportの正しいパスを指定し、全件SHA256/件数検証を通す。
2. 専用Colabノートブックで依存環境を構築し、T4で10件→100件full TTS+Whisper QCを実行する。CER/WER、voiceの聞き取り、失敗理由、VRAMを確認する。
3. 必要なら切断なしの1000件benchmarkを実行し、CUの実減少とwall時間で本番費用を見積もる。入力30万件が必ず300時間になるわけではない。
4. full wallは保存済みinvocationの累計。強制切断で最後のreportが書けない場合は過小評価される。TTS RTFにはload・I/Oを含めず、phase wallと別に確認する。
5. JParaCrawl由来の利用条件を確認し、既定voice・speed・出力先を確定する。ライセンス根拠は `ATTRIBUTION.md` の公式リンクを参照。
6. 本番を進めると決めるまでは `APPROVE_PRODUCTION=False` を維持する。mainへのmergeは行っていない。
