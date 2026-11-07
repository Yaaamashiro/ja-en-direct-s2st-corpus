# Kokoro + JParaCrawl 運用ガイド

[READMEに戻る](../README.md) · [Colabを開く](https://colab.research.google.com/github/Yaaamashiro/ja-en-direct-s2st-corpus/blob/main/notebooks/colab_kokoro_jparacrawl.ipynb)

## 入力と処理の流れ

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

## Colabの手順

専用ノートブックはNVIDIA L4を本番targetとし、`production-kokoro-jparacrawl-l4.yaml` を使います。CUDA必須・最低VRAM 20 GiB・BF16不要です。L4以外はtarget/actual GPUのwarningを表示し、T4はVRAM条件で停止します。従来の `production-kokoro-jparacrawl.yaml` は残しています。Kokoro float32とWhisper float16を順番にロードします。初期値はKokoro worker=1・ASR batch=8です。Qwen用の `S2ST_TTS_BATCH_SIZE` を引き継ぐと停止します。

最初に、実行したい処理に合わせてセルを選びます。

| 状況 | 実行順 | 設定 |
| --- | --- | --- |
| 初回 | **1 → 2 → 3** | L4、10件で動作を確認してから100件 |
| 速度測定と最終テスト | **4 → 5 → 6** | 測定を有効にし、採用した設定で1000件 |
| 本番開始・中断後の再開 | **1 → 2 → 5 → 7 → 8** | 既存の設定を確認し、まず1 shard、確認後に残り全部 |
| 全shard完了後の最終集計 | **1 → 2 → 9** | CPU可。2で `RELEASE_ONLY = True`、9で `WRITE_RELEASE = True` |

セル1はDrive接続・コード取得・依存関係の導入を行います。環境の残っている同じランタイムでは、完了済みの準備セルを繰り返す必要はありません。セル8の全件実行はshardの生成とQCまでを行います。セル9でreleaseを書き出します。

CLIから生成する場合は `--allow-production` が必要です。既存の `run-remaining-shards` の自動集計を分けるには、ノートブックと同じ `run-remaining-shards --skip-consolidate` を使います。

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml prepare
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 10
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 100
# 任意のbenchmark（結果を見てから）
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml smoke-test --pairs 1000
```

ノートブックでは `python` の代わりに専用venvの `PYTHON` を使い、設定はDrive上の `kokoro-jparacrawl-l4-config.yaml` です。10件・100件smokeはexport順の先頭N件を使います。1000件benchmarkはtrainからlength-bin 0/1/2/3/4を20/150/400/330/100件ずつ、seed=42とpair_idのSHA256 rankingで選び、最後に元input順に戻します。既定では区分ごとの必要件数が不足すると、入力パス・不足区分・train実件数を表示して停止します。`--benchmark-shortfall redistribute` を明示すると、不足区分を実件数まで減らし、不足分を他の区分へ元のquota比率と最大剰余法（同率はbin ID順）で再配分し、総数1000件を維持します。train全体の対象件数が不足する場合は再配分でも停止します。`--benchmark-seed` でseedを指定できます。本文やsplitは変更しません。選択方法・seed・quota・元manifestのSHA256・選択IDは `input.selection.json` とrelease summaryに記録します。10件・100件・1000件はそれぞれ `kokoro-l4-smoke/pairs-N/production/` に独立して保存します。以前の先頭1000件benchmarkを残している場合は新しいsmoke出力先を指定してください。

## L4 performance tuning

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

詳しい実装範囲と未実施項目は [L4実装レポート](kokoro-l4-report.md) を参照してください。

## Voiceと再現性

既定は日本語 `jf_alpha`、英語 `af_heart`、speed=1.0。日英で同一人物を表す設定ではありません。公式 `kokoro==0.9.4` とモデルrevision `f3ff3571791e39611d31c381e3a41a3af07b4987` を固定しています。voice tensorも同じrevisionから取得します。1つのKModelを日英KPipelineで共有し、空チャンクを除く全チャンクを順番に連結します。全チャンクが空ならgeneration failureです。

```bash
python -m s2st_corpus.cli --config configs/production-kokoro-jparacrawl-l4.yaml voice-samples --output-dir /content/drive/MyDrive/ja-en-direct-s2st-corpus/kokoro-voice-samples --en-voices af_heart af_bella am_adam
```

日本語はjf_alpha・jf_gongitsune・jf_nezumi・jf_tebukuro・jm_kumoを固定文で試聴できます。試聴は本番データとは別です。configのvoiceを確定してからsmokeを開始してください。同じ出力先でvoice・speed・モデル・入力・QC等を変更すると `provenance mismatch` で停止します。設定を元に戻すか、新しい出力先を指定してください。

## 保存先と再開

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

索引はColabでは `/content/s2st-input-index/` に保存します（`S2ST_INPUT_INDEX_DIR` で変更可能。未指定時はOSのローカル一時領域）。索引を別CLIプロセスへ引き継がず、再接続・新しいCLIでは元ファイルを検証して作り直します。各shardではmanifest・export summaryのsize/mtime等を確認し、途中変更があればキャッシュを黙って更新せず停止します。統合時はJSONL公開の直前に元manifest・summary・smoke subsetのSHA256を再確認します。単独の `run-shard` / `run-next-shard` と `run-remaining-shards --skip-consolidate` でも終了時に入力のSHA256を再確認します。

## 結果・速度の読み方

WhisperとQC閾値は従来どおり（JA CER≤0.10、EN WER≤0.10、0.7〜30秒）。Kokoroのcontent retryは0です。`[done]` はshardの処理終了を意味し、全ペア合格とは限りません。`alignment-transfer-s2ut` にはtrain/devを含む `accepted.jsonl` を渡し、trainerが `split` に従って学習・検証に分けます。

Kokoro productionの `checkpoint_every` は初期値100で、単位はpair数ではなく**日英合計のutterance数**です。例えば50 pairs×2言語で100 utterancesです。ASR batch_size=8とは独立し、100 utteranceごと、言語の処理終了時、phase終了時、捕捉した例外時、shard終了時に保存します。ASR batchの結果はRAMに保持し、8件ごとの全checkpoint書き換えを避けます。正常checkpointのWAVのみ再利用し、checkpointにない孤立WAVは従来どおり再生成します。既存Drive設定で値が10の場合は、設定の `run.checkpoint_every` を100へ変更してください。Qwenの既存config値は維持しています。

容量はCLIプロセス開始後、対象outputとcacheを最初に確認する際だけ全走査し、それ以降のatomic WAV/checkpoint/report置換は旧サイズとの差分を加算して上限を監視します。各shardの開始・終了で同じoutputを全走査しません。新しいCLIプロセスではbaselineを取得し直し、手動 `preflight` でも最新の容量を確認できます。この差分監視は同じ出力先に他プロセスが書き込まない前提です。

summaryの `checkpoint_writes` / `checkpoint_write_seconds`、`directory_scan_count` / `directory_scan_seconds` でI/O負担を確認できます。checkpoint_writesは保存試行数です。`current_run_written_bytes` は置換を含む書き込み量で、保存サイズの純増とは異なります。duration不合格は `duration_failures_by_length_bin` と `duration_failures_by_language_and_length_bin` にutterance単位で集計します。閾値30秒は変更していません。

`summary.json` は採用件数、corpus別件数、採用音声のja_hours/en_hours、入力と生成設定のprovenanceを含みます。`attempt_metrics` の各言語には全生成音声の時間・TTS推論時間・RTF・評価済み試行の平均CER/WERが入ります（平均値はcorpus全体の文字数/単語数重み付き指標ではありません）。生成失敗・空音声・duration落ち・ASR失敗も集計します。

RTF = TTS推論wall秒 / 生成音声秒。モデルload・WAV変換/保存はRTFから除き、`invocations` の `tts_load_seconds`、`asr_load_seconds`、各phase時間、`full_pipeline_wall_seconds` で別に確認します。GPUとPyTorchのpeak allocated VRAMも実行単位で保存します。pairs/secと音声時間/wall時間は成功・失敗した保存済みinvocationの累積wall時間を使います。**強制切断でfinallyまで実行されなかったinvocationはwall計測が残らず、速度が過大に見えることがあります。CU見積もりには切断なしの新しいbenchmarkを使ってください。** 実行ログのCU差分とwall時間から消費率を計算し、RTFだけからCUを断定しないでください。

本番完了後の `consolidate` は全shardと音声の整合性を確認します。`complete: true` / `missing_shards: []` を確認してください。途中集計だけなら `consolidate --allow-incomplete` を明示します。

外部manifestを使う実行では、`consolidate --audio-verification inventory` で、shard単位のファイル一覧と保存済みQCを照合して書き出せます。WAVの存在は確認し、内容・形式・SHA256の再検証は省きます。結果の `summary.json` に `audio_verification: inventory` を記録します。WAVの内容まで確認する場合は、既定の `full` を使います。集計はGPUを必要としません。

## 検証

```bash
python -m pytest -q
# 固定文の日英TTS単体チェック（入力exportもWhisperも使わない）
python scripts/check_kokoro_runtime.py --device cpu
```

実装時の実行範囲と未検証項目は [Kokoro実装レポート](kokoro-implementation-report.md) と [本番前修正レポート](kokoro-scale-report.md) を参照してください。JParaCrawl由来データの利用条件はKokoro重みのライセンスとは別です。[ATTRIBUTION.md](../ATTRIBUTION.md)に公式情報を記載しています。
