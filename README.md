# Japanese-to-English Direct S2ST corpus

このリポジトリには2つの独立したprofileがあります。

| Profile | 入力 | TTS | Colab |
| --- | --- | --- | --- |
| Legacy / original | JESC・KFTT | Qwen3-TTS / Ono_Anna | `notebooks/colab_pro.ipynb` |
| New Kokoro + JParaCrawl | 検証済みtext corpus export | Kokoro-82M / jf_alpha・af_heart | [専用ノートブック](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/feature/kokoro-tts/notebooks/colab_kokoro_jparacrawl.ipynb) |

Kokoro版は `feature/kokoro-tts` ブランチです。[新profileの手順](#new-kokoro--jparacrawl-profile)を参照してください。以下の従来手順はQwen用です。

## Legacy / original profile

公式JESC・KFTTから、同一話者の日英合成音声ペアを作成します。

- 学習用テキスト：Filtered JESC 70% + KFTT 30%（推定日本語音声時間比）
- TTS：Qwen3-TTS 1.7B CustomVoice / `Ono_Anna`
- 音声：16 kHz / mono / PCM16 WAV
- 品質検査：OpenAI Whisper large-v3-turbo
- 目標：学習用約300時間 + 公式dev/test

## 必要環境

- BF16対応NVIDIA GPU（VRAM 16 GiB以上）
- NVIDIA GPUを利用できるDocker
- 5文スモークテスト用に1 GiB以上、本番コーパス用に80 GiB以上
- Hugging Face cache用に8 GiB以上

## 実行

本番GPUで、最初に実コーパス5文だけを生成・検査できます。

```powershell
docker compose run --build --rm corpus smoke-test
```

JESCから3文、KFTTから2文を文長別に選び、本番と同じQwen・Whisper・QC設定で日英10音声を生成します。結果は`data/smoke/`へ保存され、`data/production/`とは混ざりません。

```text
data/smoke/production/
├── audio/16k/{ja,en}/
└── manifests/releases/{accepted,all}.jsonl
```

スモークテストが完了したら、本番コーパスは次の1コマンドで生成します。

```powershell
docker compose run --build --rm corpus
```

このコマンドが次を順番に実行します。

1. 公式JESC・KFTTアーカイブの取得とSHA-256検証
2. 安全な展開、正規化、言語・長さ・記号フィルタ
3. split間の完全重複除去と、70:30・文長層化選別
4. Qwen3-TTSによる日英同一話者音声の生成
5. Whisperによる日本語CER・英語WER検査と再生成
6. 全shardの統合とrelease manifest作成

### Google Colab Pro

[Colabノートブックを開く](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_pro.ipynb)

Dockerを使わず、ColabのGPUで音声を生成します。音声・進捗はGoogle Driveへ保存し、ランタイムが終了しても保存済み地点から再開します。設定は [configs/colab-pro.yaml](configs/colab-pro.yaml)、ノートブック本体は [notebooks/colab_pro.ipynb](notebooks/colab_pro.ipynb) です。

本番データは512個の「shard」（処理単位）に分けています。各shardで日英音声の生成、品質検査、必要な再生成を行い、採用・除外を確定します。**生成開始後は `num_shards`・入力manifest・保存先を変更しないでください。** 別のセルやランタイムから同じ保存先へ本番処理を重ねて実行することも避けてください。

#### 初回セットアップ

Colabの「ランタイムのタイプを変更」でGPUを選び、以下の順にセルを実行します。「すべてのセルを実行」ではなく、スモークテストの結果を確認してから本番へ進んでください。

| セル | 実行する内容 |
|---|---|
| 1. Google Driveをマウント | Driveへのアクセスを許可し、音声・進捗の保存先を設定します |
| 2. 最新コードを取得 | リポジトリを取得・更新し、使用する設定ファイルを指定します |
| 3. 依存関係を導入 | ノートブックのPythonと固定バージョンのライブラリを準備します。新しいランタイムごとに必要です |
| 4. GPUを確認 | GPU、VRAM、Qwenのインポートが利用可能か確認します |
| 5. バッチ件数 | 同時に処理する音声数を設定します |
| 初回だけ：5文スモークテスト | JESC・KFTTを準備し、5ペアの日英音声を生成・検査します |

Python 3.13とA100 80GBで実行した構成です。GPU確認はVRAM 14 GiB以上を条件とし、BF16対応GPUではBF16、それ以外ではFP16を選択します。Colab用設定の本番出力上限は160 GiBで、採用音声だけでなく再生成・除外音声とmanifestも含みます。容量チェックは既存の本番出力とファイルシステムの報告する空き容量の合計が160 GiB以上あるかを確認します。Driveのアカウント全体の容量も別途確認してください。モデルキャッシュ用はローカルに8 GiB、スモークテストの出力予算は1 GiBです。

スモークテストは `all_pairs_accepted: true` と採用音声の試聴で確認します。`returncode: 0` はコマンドの正常終了を示すだけで、すべての音声の品質合格を意味しません。結果は本番と分離した `smoke/production/` に保存されます。

#### バッチ数とGPUメモリ

ノートブックの「バッチ件数」セルにある次の2行を編集し、セルを実行すると反映されます。

```python
TTS_BATCH_SIZE = 72
ASR_BATCH_SIZE = 144
```

TTSはQwenによる音声生成、ASRはWhisperによる文字起こしです。**TTSとASRは同時実行せず、生成モデルを解放してからASRモデルを読み込みます。** 1つのshard内で複数音声をまとめて処理する方式で、複数shardの同時実行ではありません。

上記はA100 80GBで使用した設定で、すべての文章・GPUでのメモリ収まりを保証する値ではありません。現在のノートブックは、検出したVRAMが70 GiB未満なら両方を1件に制限します。メモリ不足時は、本番セルの停止後に48／96や32／64へ下げ、設定セルを実行してから再開してください。

`[tts-batch]` の間のメモリ使用量はTTS、`[qc-batch]` の間はASRの調整に使います。バッチ数を増やしても必ず速くなるわけではなく、長い音声で使用メモリが増えるため、ピーク使用量・処理時間・採用ペア数／CUを比較してください。

設定セルは `S2ST_TTS_BATCH_SIZE` / `S2ST_ASR_BATCH_SIZE` 環境変数を設定します。環境変数はYAMLの `tts.batch_size` / `asr.batch_size` より優先され、どちらも未設定なら各1件です。TTSの乱数はバッチ単位になるため、バッチ構成・サイズを変えた再生成は単件実行と同一波形にはなりません。使用したseedとバッチ内pair IDは記録されます。

#### 本番を実行する

「現在の進捗」で完了数を確認した後、本番セルを実行します。最初に1 shardだけ測定する場合は `False`、残りを順番に処理する場合は `True` を選びます。

```python
RUN_ALL_REMAINING = True  # Falseなら次の未完了shardを1つだけ処理
```

全件モードは `run-remaining-shards`、単一shardモードは `run-next-shard` を実行します。完了済みshardを飛ばし、途中のshardは保存済みの生成・QC結果を再利用します。全件モードは全512 shard完了後に最終集計も実行します。実行エラー時は停止し、自動で無限再試行はしません。

進捗の例：

```json
{"completed_shards": 24, "next_shard": 24, "remaining_shards": 488, "total_shards": 512}
```

shard番号は0から始まるので、この例は「0〜23が完了し、次は24」です。音声ファイルが存在するだけでは完了とは限らず、`manifests/qc/` に最終manifestが保存されたshardを完了として数えます。

#### 保存先と採用音声

既定の永続保存先は、Google Driveの「マイドライブ → ja-en-direct-s2st-corpus-data」です。

```text
ja-en-direct-s2st-corpus-data/
├── input/pairs.jsonl                 # 準備済みの日英テキスト
├── production/
│   ├── audio/16k/{ja,en}/shard-…/    # 初回・再生成の音声
│   ├── manifests/generated/         # 途中進捗：バッチ処理ではバッチごとに保存
│   ├── manifests/qc/                # shardごとの最終採用・除外結果
│   ├── manifests/releases/
│   │   ├── accepted.jsonl           # 集計時点の採用ペア
│   │   ├── all.jsonl                # 集計時点の全ペア
│   │   └── summary.json             # 採用率・日英の音声時間
│   └── failures/                    # 除外ペアと試行記録
├── smoke/production/                # スモークテストの音声・結果
├── logs/                            # 本番の実行ログ
├── reports/preparation/             # テキスト準備の集計
└── sources/                         # 元コーパス
```

`.a0.wav` は初回、`.a1.wav` は再生成です。両方が残っていても、最終採用データが二重登録されているわけではありません。`accepted.jsonl` の `ja_wav_16k` / `en_wav_16k` が採用された音声のパスです。日英それぞれで合格した試行から誤り率の低いものを選び、両言語に合格音声があるペアだけを採用します。

モデルはDriveではなく `/content/huggingface` と `/content/s2st-models` に保存します。新しいランタイムでは再取得が必要です。スモークテストの実行ログも `/content/s2st-smoke-*.log` にあり、Driveへ自動保存される本番ログとは異なります。

#### ログと品質検査の読み方

| ログ | 意味 |
|---|---|
| `[preflight]` | GPU・対象ペア数・適用バッチ数。`vram_gib` は使用量ではなくGPUの総容量です |
| `[tts] loading` / `[asr] loading` | 音声生成／文字起こしモデルの読み込み |
| `[tts-batch] 72/946 ja attempt=0` | 今回生成対象の946音声のうち72件を処理。`ja` は日本語、`en` は英語です |
| `[qc-batch] 144/946 ... seconds=...` | 品質判定の進捗。`seconds` はそのバッチのASR呼び出し時間で、shard全体の時間ではありません |
| `[retry] content failures=284` | 合格音声がない対象が284件。ペア単位ではなく日英それぞれを1件として数えます |
| `[done] shard=6 pairs=485 accepted=306 rejected=179` | shard 6が完了。485ペア中306採用、179除外です |
| `[run-all] completed=7/512 remaining=505` | 全件実行の更新後の進捗 |
| `returncode: 0` | コマンド正常終了。全件モードでは最終集計も確認します |
| `Traceback` / `returncode: 1` | 実行エラー。Traceback末尾の具体的な例外を確認します |

現在のQC基準は、日本語CER（文字誤り率）10%以下、英語WER（単語誤り率）10%以下、音声長0.7〜30秒です。長さの範囲外の音声はASR前に不合格とし、通常の品質不合格は最大1回再生成します。`attempt=0` が初回、`attempt=1` が再生成です。途中再開では保存済みの再生成音声も再利用するので、retryの件数すべてを新しく生成するとは限りません。

#### 切断・CU切れから再開する

Driveへの保存が完了した音声・進捗は再利用できます。処理中や保存前のバッチは、再実行が必要になる場合があります。

1. 新しいGPUランタイムで、セットアップのセル1〜4を実行します。
2. 割り当てられたGPUを確認し、バッチ設定セルを実行します。新しいランタイムでは環境変数が失われます。
3. 「現在の進捗」で完了数と次のshardを確認します。
4. 本番セルを実行します。合格済みのスモークテストを通常の再開時にやり直す必要はありません。

同じランタイムが生きている場合、依存関係の再インストールは通常不要です。GitHubの最新版を使うときは、本番停止後に「最新コードを取得」を実行してください。`git pull` はリポジトリを更新しますが、既に開いているノートブックのセルは置き換えないため、セル自体の変更は冒頭のリンクから最新版を開き直します。

自分で変更したセルを残すには、Colabの「ドライブにコピーを保存」などでノートブックを保存してください。GitHub上のノートブックの変更には別途コミット・プッシュが必要ですが、生成データ・進捗はGitへのコミットとは独立してDriveに保存されます。

#### 途中集計と費用の見積もり

本番セルを停止した状態で「現在までの途中集計」を実行します。これは `consolidate --allow-incomplete` で、完了済みshardのみを集計します。出力が `CompletedProcess(..., returncode=0)` だけの場合は、次のセルで保存された結果を読めます。

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

`ja_hours` / `en_hours` が採用済み音声の時間です。`complete: false` は未完了shardがあることを表し、途中集計では正常です。releaseファイルは集計コマンドで更新されるので、本番の進行中に常時最新になるわけではありません。

「約300時間」はテキスト選別時の推定日本語音声時間で、品質検査後に日英各300時間が採用される保証ではありません。全512 shardの処理完了と、採用音声の目標時間は別に確認してください。

CU残量の自動取得・購入や、Colabランタイムへの自動再接続は行いません。実測した「消費CU ÷ 完了shard数」「実行時間 ÷ 完了shard数」に残りshard数を掛けて見積もります。GPU・文章量・再生成率によって変わるので、固定のCU数や所要時間は保証しません。

#### よくある問題

- **`output safety limit exceeded`**：Driveの満杯を直接示すものではなく、`run.maximum_output_gib` に指定した本番出力上限を超えたという通知です。旧Colab設定は78 GiBでした。Driveに十分な空き容量がある場合は最新コードを取得し、160 GiBの設定で再開してください。shard数や保存先を変えたり、既存音声を削除したりする必要はありません。
- **`flash-attn is not installed`**：追加の高速化ライブラリがないという案内です。現在は `sdpa` を指定しており、この表示だけで失敗ではありません。
- **`operator torchvision::nms does not exist`**：torchとtorchvisionの不整合が疑われます。ノートブックのセットアップはtorch 2.7.1・torchvision 0.22.1・torchaudio 2.7.1を同じCUDA 12.6配布元から導入します。
- **`Could not load libtorchcodec`**：既存TorchCodecとの不整合を避けるため、セットアップでTorchCodecを削除します。このパイプラインのWAV入力はFFmpegで読み込みます。
- **`CalledProcessError` だけが見える**：その表示は原因ではなく、子プロセスの失敗通知です。本番ログの末尾にある元のTracebackを確認してください。
- **`CUDA out of memory`**：停止後に該当するTTS／ASRのバッチ数を下げ、設定セルを再実行してから本番を再開します。未保存バッチはやり直しになる場合があります。

以前のASR環境エラーでスモークテストの最終結果が全件不合格になった場合は、環境修正後に次のコマンドで保存済み音声を再検査できます。旧レポートは `smoke/production/qc-backups/` へバックアップされ、音声生成は行いません。これは通常の本番再開では不要です。

```bash
python -m s2st_corpus.cli --config configs/colab-pro.yaml recheck-smoke
```

### Docker実行時の保存先と再開

処理はshard単位で保存されます。途中停止した場合も、同じコマンドを再実行すれば完了済み処理を再利用します。

外付けSSDへ保存する場合は、実行前に保存先を指定します。

```powershell
$env:CORPUS_DATA_ROOT = "D:\ja-en-direct-s2st-corpus"
docker compose run --build --rm corpus
```

## 完成物

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

公式配布物、生成音声、モデルcacheはGitへ入りません。データの出典と利用条件は[ATTRIBUTION.md](ATTRIBUTION.md)、詳しい再開・個別実行方法は[production-runbook.md](docs/production-runbook.md)を参照してください。

## New Kokoro + JParaCrawl profile

```text
JParaCrawl v3 → jparacrawl-filtering → text corpus export
→ Kokoro-82M (native 24 kHz) → 16 kHz / mono / PCM16 WAV → Whisper QC
```

フィルタリング・重複除去・30万件選択・train/dev分割は `jparacrawl-filtering` 側で完結させます。そちらの `scripts/export_text_corpus.py` が出力した `manifest.jsonl` と `summary.json` だけを読みます。rawデータや `selected.parquet` は読みません。

`configs/production-kokoro-jparacrawl.yaml` は既定で次の2ファイルを検証します。

```text
/content/drive/MyDrive/jparacrawl-filtering/outputs/text-corpus/manifest.jsonl
/content/drive/MyDrive/jparacrawl-filtering/outputs/text-corpus/summary.json
```

検査内容はsummaryのcomplete・corpus・件数、実ファイルのSHA256、全recordのID重複・空テキスト・split・corpus・length_binです。期待値はconfigで変更できます（既定 total=300000 / train=298000 / dev=2000 / test=0）。全本文を巨大listにせず検証します。検証時にレコードを除外・並べ替え・修正しません。export文字列をそのまま公式KPipelineへ渡します。モデル内部のG2P処理やQC用の評価正規化とは区別してください。

### Colabの手順

専用ノートブックを上から実行します。T4を対象にCUDA必須・BF16不要にしており、T4の実測結果はsmokeで確認してください。Kokoro float32とWhisper float16を順番にロードするため、同時に両モデルでVRAMを埋める実装ではありません。Kokoroは1文ずつ生成し、ASRは既定8件バッチです。Qwen用の `S2ST_TTS_BATCH_SIZE` を引き継ぐと停止します。

1. Driveをmountし、専用checkoutで `feature/kokoro-tts` を取得します。
2. Python 3.12の専用venvにTorch 2.7.1/cu126と固定依存を導入します。Colab本体のPythonには依存しません。`espeak-ng` とUniDic `3.1.0+2021-08-31` の辞書本体も導入します。
3. Driveに保存する設定を作成・復元し、入力2ファイルを確認します。
4. 入力検証→GPU/容量確認→preload→任意のvoice試聴を実行します。
5. **10件→100件のfull TTS＋Whisper QC**を実行し、ここで結果を確認します。1000件benchmarkは既定で無効です。
6. 結果を確認して本番を実行する場合だけ、`APPROVE_PRODUCTION=True` にします。最初は次の1 shardです。全件実行にはさらに `RUN_ALL_REMAINING=True` が必要です。

本番は自動では開始しません。CLIもexternal profileの生成コマンドには `--allow-production` が必要です。

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl.yaml prepare
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl.yaml smoke-test --pairs 10
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl.yaml smoke-test --pairs 100
# 任意のbenchmark（結果を見てから）
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl.yaml smoke-test --pairs 1000
```

ノートブックでは `python` の代わりに専用venvの `PYTHON` を使い、設定はDrive上の `kokoro-jparacrawl-config.yaml` です。smokeはexport順の先頭N件を使い、従来の文末・長さによる選別をしません。10件・100件・1000件はそれぞれ `kokoro-smoke/pairs-N/production/` に独立して保存します。

### Voiceと再現性

既定は日本語 `jf_alpha`、英語 `af_heart`、speed=1.0。日英で同一人物を表す設定ではありません。公式 `kokoro==0.9.4` とモデルrevision `f3ff3571791e39611d31c381e3a41a3af07b4987` を固定しています。voice tensorも同じrevisionから取得します。1つのKModelを日英KPipelineで共有し、空チャンクを除く全チャンクを順番に連結します。全チャンクが空ならgeneration failureです。

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl.yaml voice-samples --output-dir /content/drive/MyDrive/ja-en-direct-s2st-corpus/kokoro-voice-samples --en-voices af_heart af_bella am_adam
```

日本語はjf_alpha・jf_gongitsune・jf_nezumi・jf_tebukuro・jm_kumoを固定文で試聴できます。試聴は本番データとは別です。configのvoiceを確定してからsmokeを開始してください。同じ出力先でvoice・speed・モデル・入力・QC等を変更すると `provenance mismatch` で停止します。設定を元に戻すか、新しい出力先を指定してください。

### 保存先と再開

```text
MyDrive/ja-en-direct-s2st-corpus/
├── kokoro-jparacrawl-config.yaml
├── kokoro-logs/                    # リアルタイム実行ログ
├── kokoro-environment/             # pip freezeと実装commit
├── kokoro-smoke/pairs-{10,100,1000}/production/
└── kokoro-jparacrawl/              # 本番（Qwenとは別）
    ├── provenance.json
    ├── audio/16k/{ja,en}/shard-.../*.a0.wav
    ├── manifests/generated/       # 途中checkpoint
    ├── manifests/qc/              # 完了したshard
    ├── failures/
    ├── reports/invocations/        # 実行単位の時間・GPU計測
    └── manifests/releases/{accepted.jsonl,all.jsonl,summary.json}
```

切断後はDrive mount→repo/依存/設定の復元→入力検証・GPU確認→status→run-next-shardで再開します。新規ランタイムではvenvの再作成とモデル再ロードが必要です。元のcheckpointとcanonical WAVが有効なら生成を省略し、QC未実施分から続けます。checkpointに対応しない孤立WAVは再生成します。壊れたWAV、0 byte、フォーマット・ハッシュ・duration・metadata不一致は再開時に明示的なエラーにし、黙って正常扱いしません。QCでduration落ちとして記録された音声は不合格のまま保持します。

`status` は完了manifestのID・provenanceを検査します。`run-shard` の再開と `consolidate` はWAVの実体も検査します。**同じ出力先を複数プロセスから同時実行しないでください。** 既定でnative 24 kHz音声は保存しません。必要なdebug runだけ `save_native: true` と別出力先を使えます。

### 結果・速度の読み方

WhisperとQC閾値は従来どおり（JA CER≤0.10、EN WER≤0.10、0.7〜30秒）。Kokoroのcontent retryは0です。`[done]` はshardの処理終了を意味し、全ペア合格とは限りません。学習には `accepted.jsonl` のペアを使ってください。

`summary.json` は採用件数、corpus別件数、採用音声のja_hours/en_hours、入力と生成設定のprovenanceを含みます。`attempt_metrics` の各言語には全生成音声の時間・TTS推論時間・RTF・評価済み試行の平均CER/WERが入ります（平均値はcorpus全体の文字数/単語数重み付き指標ではありません）。生成失敗・空音声・duration落ち・ASR失敗も集計します。

RTF = TTS推論wall秒 / 生成音声秒。モデルload・WAV変換/保存はRTFから除き、`invocations` の `tts_load_seconds`、`asr_load_seconds`、各phase時間、`full_pipeline_wall_seconds` で別に確認します。GPUとPyTorchのpeak allocated VRAMも実行単位で保存します。pairs/secと音声時間/wall時間は成功・失敗した保存済みinvocationの累積wall時間を使います。**強制切断でfinallyまで実行されなかったinvocationはwall計測が残らず、速度が過大に見えることがあります。CU見積もりには切断なしの新しいbenchmarkを使ってください。** 実行ログのCU差分とwall時間から消費率を計算し、RTFだけからCUを断定しないでください。

本番完了後の `consolidate` は全shardと音声の整合性を確認します。`complete: true` / `missing_shards: []` を確認してください。途中集計だけなら `consolidate --allow-incomplete` を明示します。

### 検証

```bash
python -m pytest -q
# 固定文の日英TTS単体チェック（入力exportもWhisperも使わない）
python scripts/check_kokoro_runtime.py --device cpu
```

実装時の実行範囲と未検証項目は [Kokoro実装レポート](docs/kokoro-implementation-report.md) を参照してください。JParaCrawl由来データの利用条件はKokoro重みのライセンスとは別です。[ATTRIBUTION.md](ATTRIBUTION.md)に公式情報を記載しています。

## ライセンス

このリポジトリのソースコードは[MIT License](LICENSE)で公開します。JESC・KFTTの元テキストと、それらから作成する生成物には、各データセットのライセンス条件が別途適用されます。Qwen3-TTS・Whisperを含む第三者資源の詳細は[ATTRIBUTION.md](ATTRIBUTION.md)を確認してください。
