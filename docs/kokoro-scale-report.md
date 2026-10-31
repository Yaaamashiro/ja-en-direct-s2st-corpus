# Kokoro + JParaCrawl 本番前修正レポート

## Git

- branch: `feature/kokoro-tts`
- base commit: `295a58d`
- commit: このレポートを含むコミットを最終回答に記載。mainへmergeしていません。

## 修正内容

| 項目 | 動作 |
| --- | --- |
| release split | train/dev/test別にQC acceptedを出力。accepted.jsonlは全splitの互換用として維持 |
| checkpoint frequency | Kokoro本番の初期値を100 utterancesに変更。日英合計で数える |
| QC checkpoint | batch sizeと保存間隔を分離。100 utterances、言語終了、phase終了、例外時に保存。shard終了時の保存も維持 |
| directory scan | CLIプロセスで各監視rootのbaselineを初回だけ走査し、atomicファイル置換のサイズ差分で更新。新規プロセスでは再走査 |
| streaming consolidate | ローカル一時SQLite索引にinput/QC JSONを保存。入力順で1行ずつreleaseへ出力。RAMに保持するQCは最大1 shard程度 |
| benchmark selection | trainのみ、seed=42、pair_idのSHA256 ranking、bin 0/1/2/3/4 = 20/150/400/330/100件。最終順は元input順 |
| duration stats | length-bin別・日英別にduration不合格utterance数を集計。0.7〜30秒の閾値は維持 |

external validation、本文、split、voice、モデルrevision、音声形式、KModel共有、Whisper QC、checkpoint resume、provenance、音声SHA256検証、本番gateを維持しています。大きなprovenance objectは今回も各recordへ保持し、既存checkpointとの互換性を優先しました（共通ファイル参照への移行は未実施）。

SQLiteはDrive外のOS一時ディスクを使い、全レコードをPythonのlist/dictへ保持しません。入力の一意性もSQLite UNIQUE制約で検査し、従来のcanonical順とID集合一致検査を維持します。external profileでは最大1 shardずつ既存の `validate_records()` を呼び、metadata・provenance・選択音声・QC status・WAV形式/ハッシュ/durationを検証します。

releaseは全ファイルを `.tmp` へ書き、flush/fsync後、既存summaryを無効化してJSONLを置換します。新しいsummaryを最後に公開します。複数ファイルのrename全体を1トランザクションにはできないため、利用側はsummaryをcompletion markerとして確認してください。検証失敗時は公開前の旧releaseを維持し、公開中の例外ではsummaryを残しません。

## Tests

```text
pytest: 85 passed, 2 existing tar-extraction DeprecationWarnings
legacy Qwen regression: passed (production-qwen17b.yaml / colab-pro.yaml)
Kokoro tests: passed
streaming consolidate: passed; canonical order, split output, ID validation,
  corrupt WAV, metadata/provenance/selected-audio mismatch, incomplete,
  interrupted publication and completion marker
checkpoint interval: passed; ASR batch_size=8 / interval=100,
  language/phase end and exception writes, TTS utterance counting
stratified sampling: passed; 1000 synthetic records selected from 2500,
  exact quota, train only, deterministic seed, unique IDs, unchanged records
```

検証コマンド（このWindows sandboxの既定一時フォルダにはアクセス制約があるため、テスト時だけTEMP/TMPを作業領域のvenvへ指定）：

```powershell
$env:TEMP = Join-Path $PWD '.venv'
$env:TMP = $env:TEMP
.venv\Scripts\python.exe -m pytest -q --basetemp=.venv/test-scale-05
```

既存Kokoro runtime checkも実行成功（CPU、固定文の日英各1文、Whisperなし）。24 kHz nativeから16 kHz/mono/PCM16の保存を確認しました。

```text
report: data/kokoro-scale-runtime-check/report.json (Git対象外)
load: 2.399 seconds
JA: audio 3.875 seconds / TTS 1.647 seconds / RTF 0.425
EN: audio 3.800 seconds / TTS 1.232 seconds / RTF 0.324
returncode: 0
```

ローカル環境は前回と同じ `kokoro==0.9.4`, CPU torch 2.7.1, UniDic 3.1.0+2021-08-31、Windows検証用pyopenjtalk-plusです。Colab用標準pyopenjtalkとは区別しています。既知のWindows版phonemizer終了時cleanup警告は出ましたが、生成・保存確認は成功しています。このCPU固定文の値をColab CU見積もりには使用しません。

## Release files

| ファイル | 内容 |
| --- | --- |
| all.jsonl | accepted/rejectedを含むQC結果全体、元input順 |
| train.jsonl | acceptedかつsplit=train、元input順。S2UT training用 |
| dev.jsonl | acceptedかつsplit=dev、元input順。検証用 |
| test.jsonl | acceptedかつsplit=test、元input順。test=0なら空ファイル |
| accepted.jsonl | 全splitのaccepted、元input順。互換・確認用でtrainingには使用しない |
| summary.json | 最終completion marker。split別total/accepted/rejected/acceptance_rate、採用音声時間、RTF、I/O、bin別duration不合格等 |

学習時間の指標は `train_ja_hours` / `train_en_hours` です。従来の `ja_hours` / `en_hours` は全splitの採用合計として残します。

benchmarkは `input.selection.json` にselection method、seed、quota、元manifest SHA256、選択IDを保存し、release summaryにも同梱します。ノートブックは1000件の集計から品質・length-bin分布・RTF・ASR/full wall・checkpoint/scan時間・GPU/peak VRAMを表示します。1000件とproductionの初期値は無効のままです。

## Performance-related defaults

```text
num_shards: 128
max_pairs_per_shard: 3000
checkpoint_every: 100 utterances
ASR batch_size: 8
```

Driveに既に保存済みの設定は自動上書きしません。既存設定が10なら `run.checkpoint_every` を100へ変更してください。Qwenの既存profile値は変更していません。

I/O指標は `checkpoint_writes`（保存試行数）、`checkpoint_write_seconds`、`directory_scan_count`、`directory_scan_seconds`。runの増分書き込み量は `current_run_written_bytes`（置換分を含む書き込み量で、純増容量ではない）です。baselineは保存先の全ファイルを含み、その後のWAV/checkpoint/report置換は旧ファイルとの差分で監視します。外部プロセスとの同時書き込みは非対応です。

## Production前に残っていること

以下はすべて未実施です。

- 実JParaCrawl export validation
- Colab T4 10 pair smoke
- Colab T4 100 pair smoke
- Colab T4 1000 pair benchmark
- 30万pair production

実データの品質・VRAM・I/O負担・CUを確認してから本番を判断してください。ローカル合成データのテストはこの確認を代替しません。`--allow-production`、`APPROVE_PRODUCTION=False`、`RUN_ALL_REMAINING=False` を維持しています。
