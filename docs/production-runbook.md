# 本番コーパス生成 Runbook

対象は、選別済みの日英対訳から Qwen3-TTS 1.7B の同一話者音声を生成し、OpenAI Whisper large-v3-turboで品質管理する工程である。GTX 1650用smoke testとは分離している。

## 必要環境

- Windows + Docker Desktop + WSL2 NVIDIA backend、またはNVIDIA Container Toolkitが動くLinux
- BF16対応NVIDIA GPU
- VRAM 16 GiB以上
- Dockerイメージ用に約16 GB
- Hugging Faceモデルキャッシュ用に約7 GB
- 生成先に80 GiB以上の空き

生成先は80 GiB、モデルcache側は8 GiBを別々にpreflightするため、外付けSSDへ生成する場合にシステムドライブへ80 GiBを要求しない。

GTX 1650はBF16非対応かつVRAM 4 GBのため、本番プロファイルは実行できない。GTX 1650では既存のsmoke testだけを使う。

## 入力manifest

既定では、ホスト側の `data/input/pairs.jsonl` を読む。JESC/KFTTの配布物そのものはGitやDockerイメージへ含めない。

1行1 JSON objectとし、次の5項目を必須とする。

```json
{"pair_id":"jesc_train_000001","corpus":"jesc","split":"train","ja_text":"今日は早く帰ります。","en_text":"I will go home early today."}
```

- `pair_id`: 全体で一意な英数字・`_`・`-`、96文字以下
- `corpus`: `jesc`、`kftt`など
- `split`: `train`、`dev`、`test`など
- `ja_text` / `en_text`: TTSへ渡す前の採用済み対訳
- その他の`source_id`、`license_id`、文長、語順差などの列はそのまま保持される

形式例は `samples/production_manifest.example.jsonl` にある。入力の対訳選別、重複除去、split固定はTTSより前に完了させる。

## 保存先

100 GB未満に収める既定設定では、Qwenのnative音声を保存せず、16 kHz・mono・PCM16だけを保存する。

```text
output/
  audio/16k/{ja,en}/shard-xxxxx-of-00128/<pair_id>.a<attempt>.wav
  manifests/generated/   # 再開用checkpoint
  manifests/qc/          # shardごとの確定結果
  manifests/releases/    # 統合manifest
  failures/              # 不合格例と理由
```

300時間×2言語は約64.37 GiB、再生成10%を含む見積りは約70.81 GiBである。設定上のhard limitは78 GiBである。モデルキャッシュとDockerイメージはこの78 GiBに含まれない。

外付けSSDを使う場合は、PowerShellで生成先と入力先を指定する。

```powershell
$env:CORPUS_INPUT_ROOT = "D:\direct-s2st\input"
$env:CORPUS_OUTPUT_ROOT = "D:\direct-s2st\production"
```

## ビルドと計画検査

```powershell
docker compose build
docker compose run --rm corpus `
  --config /workspace/configs/production-qwen17b.yaml plan
```

`plan`はGPUやモデルを使わず、次を検査する。

- JSONL形式と必須列
- `pair_id`重複
- 128 shardの件数分布
- 1 shard 2,500文の上限
- 300時間分の容量見積り

## 2,000文pilot

本番manifestを作る前に、同じ列構造で2,000文のpilot manifestを用意する。設定ファイルを複製し、`input_jsonl`、`num_shards`、`expected_hours_per_language`、`maximum_output_gib`だけをpilot用に変える。TTS、話者、QC閾値は変えない。

pilotで実測するもの:

- 日英の平均秒数と採用率
- CER/WER分布
- attempt 1の発生率
- 生成RTFと総所要時間
- 文長・コーパス別の失敗率

## shard実行

1 GPUにつきworkerは1つにする。複数コンテナを同じGPUへ同時投入しない。

```powershell
docker compose run --rm corpus `
  --config /workspace/configs/production-qwen17b.yaml `
  run-shard --shard-index 0
```

順次実行する例:

```powershell
0..127 | ForEach-Object {
  docker compose run --rm corpus `
    --config /workspace/configs/production-qwen17b.yaml `
    run-shard --shard-index $_
  if ($LASTEXITCODE -ne 0) { throw "shard $_ failed" }
}
```

各shardでは次の順に処理する。

```text
Qwen attempt 0
  → Qwenを解放
  → Whisper QC
  → 不合格言語だけQwen attempt 1
  → Qwenを解放
  → Whisper QC
  → 言語別に最良attemptを採用
```

10言語音声ごとにmanifestをatomic checkpointする。途中停止後に同じコマンドを実行すると、保存済みattemptを再生成せず続行する。完成済みshardは即終了する。

CUDA OOMやdevice-side assertはCUDA contextが不正になり得るため、当該コンテナを失敗終了させる。自動的に処理を続けず、設定やハードウェアを確認して同じshardを再実行する。

## 統合

全128 shardの完了後:

```powershell
docker compose run --rm corpus `
  --config /workspace/configs/production-qwen17b.yaml consolidate
```

成果物:

- `manifests/releases/all.jsonl`: 合格・不合格を含む全例
- `manifests/releases/accepted.jsonl`: 学習へ使う合格例
- `manifests/releases/summary.json`: 件数、採用率、日英時間、corpus別件数

途中集計だけが必要な場合は`consolidate --allow-incomplete`を使う。この出力は正式releaseにしない。

## 固定された主要条件

- Qwen: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- Qwen revision: `f00cf133b78d3c2c35857faba3b1be9b98c4f971`
- speaker: `Ono_Anna`（日英共通）
- Qwen: CUDA BF16、batch size 1相当
- ASR: `openai/whisper-large-v3-turbo`
- Whisper revision: `60be3615a4d667e1258e8ad29130467587c489aa`
- 日本語: 読み正規化CER 10%以下
- 英語: Whisper正規化WER 10%以下
- 長さ: 0.7〜30.0秒
- 内容不合格の再生成: 言語ごとに1回

依存関係、モデルrevision、speaker、seed規則、QC値を本番途中で変更しない。変更する場合は別output directoryを使い、コーパス版を上げる。
