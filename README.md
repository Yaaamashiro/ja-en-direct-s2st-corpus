# Japanese-to-English Direct S2ST production corpus

選別済みの日英対訳から、同一話者の合成音声ペアを本番規模で生成します。

```text
日英対訳manifest
  → Qwen3-TTS 1.7B / Ono_Anna / BF16
  → 16 kHz PCM16
  → Whisper large-v3-turbo QC
  → 再生成・shard manifest
  → release manifest
```

## 必要環境

- BF16対応NVIDIA GPU
- VRAM 16 GiB以上
- NVIDIA GPUを利用できるDocker
- 生成先に80 GiB以上
- モデルcache側に8 GiB以上

## 入力

選別済み対訳を次へ置きます。

```text
data/input/pairs.jsonl
```

形式は `samples/production_manifest.example.jsonl` を参照してください。

## 実行

```powershell
docker compose build
docker compose run --rm corpus `
  --config /workspace/configs/production-qwen17b.yaml plan
```

1 shardを生成:

```powershell
docker compose run --rm corpus `
  --config /workspace/configs/production-qwen17b.yaml `
  run-shard --shard-index 0
```

全shard完了後:

```powershell
docker compose run --rm corpus `
  --config /workspace/configs/production-qwen17b.yaml consolidate
```

詳しい運用方法は `docs/production-runbook.md`、研究設計は `docs/corpus-design.md` を参照してください。
