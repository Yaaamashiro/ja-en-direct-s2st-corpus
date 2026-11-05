# Japanese-to-English Direct S2ST corpus

このリポジトリには2つの独立したprofileがあります。

| Profile | 入力 | TTS | Colab |
| --- | --- | --- | --- |
| Legacy / original | JESC・KFTT | Qwen3-TTS / Ono_Anna | `notebooks/colab_pro.ipynb` |
| Kokoro + JParaCrawl | 検証済みtext corpus export | Kokoro-82M / jf_alpha・af_heart | [専用ノートブック](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_kokoro_jparacrawl.ipynb) |

Kokoro版とQwen版は `main` ブランチにあります。Kokoro版は [専用profileの手順](#new-kokoro--jparacrawl-profile) を参照してください。以下の従来手順はQwen用です。

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

全profileのrelease `all.jsonl` は `config.run.input_jsonl` の行順を保証し、`accepted.jsonl` はその順序を保った採用subsetです（途中集計も同様）。完全なreleaseでは入力とQC結果のpair_id集合一致を検査し、不足・余分・重複を拒否します。`summary.json` に `release_order: "input_manifest"` と `missing_pair_count` を記録します。

**`alignment-transfer-s2ut` の学習には、QC acceptedの全splitを入力順で保持する `accepted.jsonl` をcanonical speech manifestとして使用します。** trainerは同一manifest内にtrain/devの両方を必要とするため、`train.jsonl` 単独では使用できません。`train.jsonl`・`dev.jsonl`・`test.jsonl` は各split単体の確認・利用向けで、それぞれのsplitのQC acceptedのみを入力順で保持します。test=0なら空の `test.jsonl` を生成します。summaryにはsplit別のtotal/accepted/rejected/acceptance_rate・採用音声時間を記録し、学習用時間は `train_ja_hours` / `train_en_hours` です。

統合ではランタイムのローカル一時ディスクにSQLite索引を作り、QCレコードは最大1 shard程度ずつ読み込みます。全件の辞書やlistは作りません。入力metadata・provenance・選択音声・WAV形式/SHA256/durationの既存検証も維持します。出力を `.tmp` に全て書いてから置換し、旧summaryを無効化した後、最後に新しい `summary.json` をcompletion markerとして公開します。利用側はJSONLの存在だけで完成と判断せず、summaryを確認してください。生成途中の一時索引にはinput/QC JSON相当のローカルディスク空きが必要です。

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

専用ノートブックはNVIDIA L4を本番targetとし、`production-kokoro-jparacrawl-l4.yaml` を使います。CUDA必須・最低VRAM 20 GiB・BF16不要です。L4以外はtarget/actual GPUのwarningを表示し、T4はVRAM条件で停止します。従来の `production-kokoro-jparacrawl.yaml` は残しています。Kokoro float32とWhisper float16を順番にロードします。初期値はKokoro worker=1・ASR batch=8です。Qwen用の `S2ST_TTS_BATCH_SIZE` を引き継ぐと停止します。

1. Driveをmountし、専用checkoutで `main` を取得します。
2. Python 3.12の専用venvにTorch 2.7.1/cu126と固定依存を導入します。Colab本体のPythonには依存しません。`espeak-ng` とUniDic `3.1.0+2021-08-31` の辞書本体も導入します。
3. Driveに保存する `kokoro-jparacrawl-l4-config.yaml` を作成・復元し、入力2ファイルを確認します。
4. 入力検証→GPU/容量確認→preload→任意のvoice試聴を実行します。
5. **10件→100件のfull TTS＋Whisper QC**で動作・音質を確認します。
6. `RUN_L4_TUNING=True` にして固定200 pairのTTS worker比較→同じ音声のWhisper batch比較を実行します。初期状態はOFFです。
7. recommendationとheadroom・品質差を確認し、worker/batchを手動で設定へ反映します。最適値はL4実測前には決めません。
8. `PERFORMANCE_CONFIG_CONFIRMED=True` と `RUN_BENCHMARK_1000=True` にして、最終設定で1000件full benchmarkを1回実行します。初期状態はどちらもFalseです。
9. 結果を確認して本番を実行する場合だけ、`APPROVE_PRODUCTION=True` にします。最初は次の1 shardです。全件実行にはさらに `RUN_ALL_REMAINING=True` が必要です。

本番は自動では開始しません。CLIもexternal profileの生成コマンドには `--allow-production` が必要です。

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml prepare
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 10
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 100
# 任意のbenchmark（結果を見てから）
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 1000
```

ノートブックでは `python` の代わりに専用venvの `PYTHON` を使い、設定はDrive上の `kokoro-jparacrawl-l4-config.yaml` です。10件・100件smokeはexport順の先頭N件を使います。1000件benchmarkはtrainからlength-bin 0/1/2/3/4を20/150/400/330/100件ずつ、seed=42とpair_idのSHA256 rankingで選び、最後に元input順に戻します。既定では区分ごとの必要件数が不足すると、入力パス・不足区分・train実件数を表示して停止します。`--benchmark-shortfall redistribute` を明示すると、不足区分を実件数まで減らし、不足分を他の区分へ元のquota比率と最大剰余法（同率はbin ID順）で再配分し、総数1000件を維持します。train全体の対象件数が不足する場合は再配分でも停止します。`--benchmark-seed` でseedを指定できます。本文やsplitは変更しません。選択方法・seed・quota・元manifestのSHA256・選択IDは `input.selection.json` とrelease summaryに記録します。10件・100件・1000件はそれぞれ `kokoro-l4-smoke/pairs-N/production/` に独立して保存します。以前の先頭1000件benchmarkを残している場合は新しいsmoke出力先を指定してください。

### L4 performance tuning

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml benchmark-performance --stage tts
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml benchmark-performance --stage asr
# 同一ランタイムで両方を順番に実行する場合
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml benchmark-performance --stage all
```

TTS tuningはtrainから固定seed=42で200 pair（希望配分はbin別4/30/80/66/20件）を選びます。区分ごとの件数が足りない場合は上記の再配分で200件を維持し、希望配分・実配分・train実件数・policyをログとselectionに記録します。存在しない短文を生成したり、同じpairを重複抽出したりはしません。この200 pairを共通の入力として、同じtext/voiceでworker=1/2/4/6を比較します。音声は `/content/kokoro-l4-benchmark` のlocal SSDへ保存し、Drive上の `kokoro-performance/l4-performance.json` と日時付きsnapshot・logsへ結果を保存します。入力と音声のSHA256も記録します。`--local-dir` / `--report-dir` / `--workers 1 2 4 6` / `--asr-batches 8 16 24 32` で明示的に変更できます。8 workerは指定した場合のみ候補になります。TTS tuningを再実行すると新しいlocal runを作り、古い音声を使ったresume速度を測らないようにしています。

Kokoroの並列数は `tts.worker_count` です。`tts.batch_size` は1のままにし、Qwenのbatch経路へ入りません。worker=1は既存の逐次経路、2以上は `multiprocessing.get_context('spawn')` の複数processです。親がモデル・voiceの固定assetを取得後、各workerはlocal assetからKModelを1回だけloadし、JA/EN pipelineを共有します。worker内のTorch CPU threadsは1です。文字数は処理時間の近似で、本文自体は変更しません。Unicode文字数の長いutteranceから先に並べ、同じ長さはpair_id/language/attemptで順序を固定してdynamic queueへ配り、各workerが24→16 kHz変換とatomic WAV保存まで行い、波形をIPC転送しません。

親だけが完了metadataをrecordへ対応付け、100 utterance間隔でcheckpointを保存します。worker crashやCUDA fatal errorでは自動再起動せず、受信済みcheckpointを保存してCLIをfailure終了します。checkpointに記録されていないWAVはorphanとして再生成し、記録済みWAVのSHA256等の検証を緩めません。全TTS workerを終了・joinしてからWhisperをloadし、TTS/ASRをGPU上でoverlapさせません。

Whisper tuningはTTS tuningで作った同じ音声を使い、batch=8/16/24/32をそれぞれ新しいspawn subprocessで実行します。OOM候補は `status: oom` として記録し、そのprocess終了後に次を試します。音声がランタイム消去で失われた場合はTTS tuningから再実行してください。ASR model/processorは親が固定revisionを取得し、候補側はoffline cacheを使います。TTS→ASR tuningは同じGit commitで実行してください。commitが変わった場合や前回・現在のcommitを確認できない場合は、ASR候補起動前に停止し、TTS tuningの再実行を求めます。既存レポートのcommitを黙って更新しません。

推奨値は `audio_seconds / wall_seconds` 最大の成功候補から計算します。headroom警告（総VRAMの90%超）・生成失敗・品質差がある候補は除外します。TTSはtext/voiceと大きなduration差（0.25秒または5%超）を比較し、Whisperは同じ音声のASR text・metric・QC判定を比較します。transcript/metric/QCのいずれかが変わったASR候補は推奨しません。実worker数によるASR品質差のL4確認は未実施なので、microbenchmarkだけで本番へ進まず、選択設定で1000件full benchmarkを実行してください。

VRAMはworkerごとのallocated/reserved peak・合計参考値・最大worker値を明示します。合計参考値は同時刻のGPU全体peakではなく、CUDA context等も含みません。benchmark時だけ約5秒間隔のnvidia-smiでGPU全体の使用量もサンプルし、瞬間的なpeakを捕捉できない可能性を残した上でreferenceと併用します。productionでは大量pollingを行わず、`gpu_utilization` は未取得（null）です。benchmarkのRTF/wallにはmodel load・worker startup・local WAV I/Oを含み、従来のattempt別TTS RTF（推論時間）とは区別します。

reportの `recommended_tts_workers` / `recommended_asr_batch_size` は候補値で、**production configを自動変更しません**。L4実測前の初期値はworker=1 / ASR batch=8のままです。worker_countは音声identityから分離し、新規runのprovenanceには `initial_performance.kokoro_worker_count`、各invocationには実際のworker数・ASR batch・phase throughput・VRAMを記録します。worker_countだけの変更では正常checkpointの音声を無効化しません。ASR batchは従来のprovenance条件に含むため、同じ保存先で変更すると不一致になります。最終設定で独立した1000件benchmarkを通し、本番開始後はworker/batchを固定してください。既存のprovenanceファイルは書き換えません。performance-lockは今回追加していません。

詳しい実装範囲と未実施項目は [L4実装レポート](docs/kokoro-l4-report.md) を参照してください。

### Voiceと再現性

既定は日本語 `jf_alpha`、英語 `af_heart`、speed=1.0。日英で同一人物を表す設定ではありません。公式 `kokoro==0.9.4` とモデルrevision `f3ff3571791e39611d31c381e3a41a3af07b4987` を固定しています。voice tensorも同じrevisionから取得します。1つのKModelを日英KPipelineで共有し、空チャンクを除く全チャンクを順番に連結します。全チャンクが空ならgeneration failureです。

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml voice-samples --output-dir /content/drive/MyDrive/ja-en-direct-s2st-corpus/kokoro-voice-samples --en-voices af_heart af_bella am_adam
```

日本語はjf_alpha・jf_gongitsune・jf_nezumi・jf_tebukuro・jm_kumoを固定文で試聴できます。試聴は本番データとは別です。configのvoiceを確定してからsmokeを開始してください。同じ出力先でvoice・speed・モデル・入力・QC等を変更すると `provenance mismatch` で停止します。設定を元に戻すか、新しい出力先を指定してください。

### 保存先と再開

```text
MyDrive/ja-en-direct-s2st-corpus/
├── kokoro-jparacrawl-l4-config.yaml
├── kokoro-logs/                    # リアルタイム実行ログ
├── kokoro-environment/             # pip freezeと実装commit
├── kokoro-l4-smoke/pairs-{10,100,1000}/production/
└── kokoro-jparacrawl-l4/           # L4本番（Qwen・従来profileとは別）
    ├── provenance.json
    ├── audio/16k/{ja,en}/shard-.../*.a0.wav
    ├── manifests/generated/       # 途中checkpoint
    ├── manifests/qc/              # 完了したshard
    ├── failures/
    ├── reports/invocations/        # 実行単位の時間・GPU計測
    └── manifests/releases/{train.jsonl,dev.jsonl,test.jsonl,accepted.jsonl,all.jsonl,summary.json}
```

切断後はDrive mount→repo/依存/設定の復元→入力検証・GPU確認→status→run-next-shardで再開します。新規ランタイムではvenvの再作成とモデル再ロードが必要です。元のcheckpointとcanonical WAVが有効なら生成を省略し、QC未実施分から続けます。checkpointに対応しない孤立WAVは再生成します。壊れたWAV、0 byte、フォーマット・ハッシュ・duration・metadata不一致は再開時に明示的なエラーにし、黙って正常扱いしません。QCでduration落ちとして記録された音声は不合格のまま保持します。

`status` は完了manifestのID・provenanceを検査します。`run-shard` の再開と `consolidate` はWAVの実体も検査します。**同じ出力先を複数プロセスから同時実行しないでください。** 既定でnative 24 kHz音声は保存しません。必要なdebug runだけ `save_native: true` と別出力先を使えます。

external manifestは**CLIプロセスごとに最初の1回だけ**元JSONLを読み、同じ読み込みでSHA256を計算してローカルSQLiteへ索引化します。exportの検証結果とrun identityもプロセス内で共有します。各shardは `WHERE shard = ?`、statusは同じ索引のpair_id検索を使い、完了shard数に比例して元manifestを読み直しません。plan・smokeの元record照合・release統合でも索引を再利用します。本文・順序・shard割り当ては従来どおりです。

索引はColabでは `/content/s2st-input-index/` に保存します（`S2ST_INPUT_INDEX_DIR` で変更可能。未指定時はOSのローカル一時領域）。索引を別CLIプロセスへ引き継がず、再接続・新しいCLIでは元ファイルを検証して作り直します。各shardではmanifest・export summaryのsize/mtime等を確認し、途中変更があればキャッシュを黙って更新せず停止します。統合時はJSONL公開の直前に元manifest・summary・smoke subsetのSHA256を再確認します。単独の `run-shard` / `run-next-shard` CLIも終了時に全ハッシュを再確認します。従って `run-remaining-shards` では元manifestの全読み込みは通常「初回の索引化＋最後のハッシュ確認」の2回です。WAVの整合性検査は省略しません。

### 結果・速度の読み方

WhisperとQC閾値は従来どおり（JA CER≤0.10、EN WER≤0.10、0.7〜30秒）。Kokoroのcontent retryは0です。`[done]` はshardの処理終了を意味し、全ペア合格とは限りません。`alignment-transfer-s2ut` にはtrain/devを含む `accepted.jsonl` を渡し、trainerが `split` に従って学習・検証に分けます。

Kokoro productionの `checkpoint_every` は初期値100で、単位はpair数ではなく**日英合計のutterance数**です。例えば50 pairs×2言語で100 utterancesです。ASR batch_size=8とは独立し、100 utteranceごと、言語の処理終了時、phase終了時、捕捉した例外時、shard終了時に保存します。ASR batchの結果はRAMに保持し、8件ごとの全checkpoint書き換えを避けます。正常checkpointのWAVのみ再利用し、checkpointにない孤立WAVは従来どおり再生成します。既存Drive設定で値が10の場合は、設定の `run.checkpoint_every` を100へ変更してください。Qwenの既存config値は維持しています。

容量はCLIプロセス開始後、対象outputとcacheを最初に確認する際だけ全走査し、それ以降のatomic WAV/checkpoint/report置換は旧サイズとの差分を加算して上限を監視します。各shardの開始・終了で同じoutputを全走査しません。新しいCLIプロセスではbaselineを取得し直し、手動 `preflight` でも最新の容量を確認できます。この差分監視は同じ出力先に他プロセスが書き込まない前提です。

summaryの `checkpoint_writes` / `checkpoint_write_seconds`、`directory_scan_count` / `directory_scan_seconds` でI/O負担を確認できます。checkpoint_writesは保存試行数です。`current_run_written_bytes` は置換を含む書き込み量で、保存サイズの純増とは異なります。duration不合格は `duration_failures_by_length_bin` と `duration_failures_by_language_and_length_bin` にutterance単位で集計します。閾値30秒は変更していません。

`summary.json` は採用件数、corpus別件数、採用音声のja_hours/en_hours、入力と生成設定のprovenanceを含みます。`attempt_metrics` の各言語には全生成音声の時間・TTS推論時間・RTF・評価済み試行の平均CER/WERが入ります（平均値はcorpus全体の文字数/単語数重み付き指標ではありません）。生成失敗・空音声・duration落ち・ASR失敗も集計します。

RTF = TTS推論wall秒 / 生成音声秒。モデルload・WAV変換/保存はRTFから除き、`invocations` の `tts_load_seconds`、`asr_load_seconds`、各phase時間、`full_pipeline_wall_seconds` で別に確認します。GPUとPyTorchのpeak allocated VRAMも実行単位で保存します。pairs/secと音声時間/wall時間は成功・失敗した保存済みinvocationの累積wall時間を使います。**強制切断でfinallyまで実行されなかったinvocationはwall計測が残らず、速度が過大に見えることがあります。CU見積もりには切断なしの新しいbenchmarkを使ってください。** 実行ログのCU差分とwall時間から消費率を計算し、RTFだけからCUを断定しないでください。

本番完了後の `consolidate` は全shardと音声の整合性を確認します。`complete: true` / `missing_shards: []` を確認してください。途中集計だけなら `consolidate --allow-incomplete` を明示します。

外部manifestを使う実行では、`consolidate --audio-verification inventory` で、shard単位のファイル一覧と保存済みQCを照合して書き出せます。WAVの存在は確認し、内容・形式・SHA256の再検証は省きます。結果の `summary.json` に `audio_verification: inventory` を記録します。WAVの内容まで確認する場合は、既定の `full` を使います。集計はGPUを必要としません。

### 検証

```bash
python -m pytest -q
# 固定文の日英TTS単体チェック（入力exportもWhisperも使わない）
python scripts/check_kokoro_runtime.py --device cpu
```

実装時の実行範囲と未検証項目は [Kokoro実装レポート](docs/kokoro-implementation-report.md) と [本番前修正レポート](docs/kokoro-scale-report.md) を参照してください。JParaCrawl由来データの利用条件はKokoro重みのライセンスとは別です。[ATTRIBUTION.md](ATTRIBUTION.md)に公式情報を記載しています。

## ライセンス

このリポジトリのソースコードは[MIT License](LICENSE)で公開します。JESC・KFTTの元テキストと、それらから作成する生成物には、各データセットのライセンス条件が別途適用されます。Qwen3-TTS・Whisperを含む第三者資源の詳細は[ATTRIBUTION.md](ATTRIBUTION.md)を確認してください。
