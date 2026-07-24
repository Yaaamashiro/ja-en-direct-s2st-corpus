# 本番コーパス生成Runbook

## 完全実行

```powershell
docker compose run --build --rm corpus
```

デフォルトの`build`コマンドは、JESC/KFTTの取得・選別、全128 shardの
音声生成とWhisper品質検査、release manifestの統合までを順番に実行する。
すべての段階は再実行可能であり、同じ設定とデータルートでは完成済みの
manifest、音声、shardを再利用する。

## 必要環境

- Windows + Docker Desktop + WSL2 NVIDIA backend、または
  NVIDIA Container Toolkitが動くLinux
- BF16対応NVIDIA GPU
- VRAM 16 GiB以上
- Dockerイメージ用に約16 GB
- Hugging Faceモデルcache用に約8 GB
- コーパスデータルートに80 GiB以上

外付けSSDへ置く場合：

```powershell
$env:CORPUS_DATA_ROOT = "D:\ja-en-direct-s2st-corpus"
docker compose run --build --rm corpus
```

## データ準備

パイプラインは次の公式アーカイブを自動取得し、設定済みSHA-256と照合する。

| Corpus | Archive | SHA-256 |
|---|---|---|
| JESC | `split.tar.gz` (2019-05-12) | `9cc6f2b31225d84b204a1bc4599e5fbb158d5739ba4aa76eab30fea05ed8ccd5` |
| KFTT | `kftt-data-1.0.tar.gz` | `fcfcaa670d6d59aa691b0e909c0d7c393852dd2fb1d6310fda9b3282dc6d1638` |

展開時は絶対パス、`..`、symlink、hardlink、device fileを拒否する。
JESCは英語・日本語TSV、KFTTは言語別ファイルとして読み込む。両コーパスの
dev/testを先に登録するため、同じ文対がtrainにもある場合はtrain側が除かれる。

学習候補には次を適用する。

- Unicode NFKC、HTML entity、空白、制御・ゼロ幅文字の正規化
- URL、メール、markup、効果音だけの字幕、内容のない行の除外
- 日本語文字比率・英語ラテン文字比率の検査
- コーパス別の文長範囲と日英長さ比の検査
- 正規化済み日英文対SHA-256による完全重複除去
- 固定seedによる再現可能な選別
- JESC/KFTTの推定日本語時間比70:30
- short/medium/longの推定時間比20:50:30を目標に層化し、不足する層は
  同一コーパスの残候補から決定論的に補完

準備結果：

```text
data/input/pairs.jsonl
data/reports/preparation/prepare-summary.json
```

展開物と選別用SQLiteは速度のためコンテナ内の一時領域に置かれ、
完成manifestと監査レポートだけがデータルートへ永続化される。

## 個別実行

問題の切り分けが必要なときだけ個別コマンドを使う。

取得・選別のみ：

```powershell
docker compose run --build --rm corpus prepare
```

manifestと容量計画の検査：

```powershell
docker compose run --rm corpus plan
```

特定shardの生成：

```powershell
docker compose run --rm corpus run-shard --shard-index 0
```

全shard完了後の統合：

```powershell
docker compose run --rm corpus consolidate
```

未完了状態の途中集計だけが必要な場合：

```powershell
docker compose run --rm corpus consolidate --allow-incomplete
```

## shard処理

各shardでは次を行う。

```text
Qwen attempt 0
  → Qwenを解放
  → Whisper QC
  → 不合格言語だけQwen attempt 1
  → Qwenを解放
  → Whisper QC
  → 言語別に最良attemptを採用
```

10言語音声ごとにmanifestをatomic checkpointする。CUDA OOMや
device-side assertはCUDA contextを壊す可能性があるため、コンテナを
失敗終了させる。設定やGPUを確認した後、同じコマンドを再実行する。

## 固定条件

- Qwen：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- speaker：`Ono_Anna`（日英共通）
- TTS：CUDA BF16
- ASR：`openai/whisper-large-v3-turbo`
- 日本語：読み正規化CER 10%以下
- 英語：Whisper正規化WER 10%以下
- 長さ：0.7〜30.0秒
- 内容不合格の再生成：言語ごとに1回

モデルrevision、speaker、依存関係、seed規則、QC閾値を途中で変えない。
設定を変更すると既存prepared manifestは再利用されないため、新しい
`CORPUS_DATA_ROOT`を使用する。

## 成果物

```text
data/production/
├── audio/16k/{ja,en}/shard-xxxxx-of-00128/
├── manifests/generated/
├── manifests/qc/
├── manifests/releases/
│   ├── all.jsonl
│   ├── accepted.jsonl
│   └── summary.json
└── failures/
```

`accepted.jsonl`がS2ST学習へ使う音声対、`all.jsonl`が不合格を含む監査用、
`summary.json`が件数・採用率・日英時間・コーパス別件数である。

16 kHz / mono / PCM16は300時間×2言語で約64.37 GiB、再生成10%を含む
見積りは約70.81 GiBである。native WAVは保存しない。モデルcacheとDocker
イメージはこの値に含まれない。
