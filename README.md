# Japanese-to-English Direct S2ST corpus

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

## ライセンス

このリポジトリのソースコードは[MIT License](LICENSE)で公開します。JESC・KFTTの元テキストと、それらから作成する生成物には、各データセットのライセンス条件が別途適用されます。Qwen3-TTS・Whisperを含む第三者資源の詳細は[ATTRIBUTION.md](ATTRIBUTION.md)を確認してください。
