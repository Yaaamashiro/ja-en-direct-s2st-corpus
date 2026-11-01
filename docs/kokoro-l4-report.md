# L4 Kokoro並列化・Whisper tuning実装レポート

## Git

- branch: `feature/kokoro-tts`
- base commit: `2cf2814`
- commit: この文書を含むコミットを最終回答に記載。
- main merge・30万pair生成は実行していません。

## Parallel implementation

| 項目 | 実装 |
| --- | --- |
| Kokoro multiprocessing | worker_count>=2で専用pool。1は既存逐次baseline |
| start method | spawn固定。fork contextは明示的に拒否 |
| worker model loading | 親が固定assetを取得・存在検証。workerはlocal-onlyでKModelを1回loadし、JA/EN KPipelineとvoiceを共有 |
| task scheduling | bounded dynamic queue。key=(pair_id, language, attempt)、重複task/resultを拒否 |
| WAV writing | worker内で24 kHz→mono→soxr HQ→16 kHz PCM16 atomic write。waveform IPCなし |
| parent checkpoint | 完了metadataを正しいrecordへ追加し、100 utteranceごと・phase終了・例外時に保存。workerはmanifestを書かない |
| resume | 検証済みcheckpointのWAVは再利用。orphanは再生成。破損WAVを正常扱いしない |
| failure | worker例外/CUDA fatal/process crashで保存してCLI failure。自動worker再起動なし |
| phase boundary | pool終了・join後にWhisperをload。TTS/ASR overlapなし |

workerはTorch CPU threads=1。親のstorage budgetはworkerのatomic置換前後のbyte差分を受け取り、WAVをDrive上で再走査せず更新します。SIGTERMはColab上で親のpool cleanupへ接続し、終了時にchildをjoin/必要ならkillします。強制killまで含む全状況で最後のcheckpointが保存される保証はなく、未記録WAVはorphanとして再生成します。

worker_countは音声semantic identityから除外し、既存provenanceを変更しません。新規runには `initial_performance.kokoro_worker_count`、各invocationには実際のworker_count・ASR batchを保存します。worker_count変更だけで正常な既存音声を無効化しません。voice/revision/speed/input/QC等の不一致は従来どおり拒否し、ASR batchも従来のprovenance条件に残しています。

## L4 profile

```text
config: configs/production-kokoro-jparacrawl-l4.yaml
target GPU: L4
minimum VRAM: 20 GiB
CUDA required: true
BF16 required: false
Kokoro: float32
Whisper: float16
candidate workers: 1,2,4,6 (8 is optional, explicit only)
candidate ASR batches: 8,16,24,32
```

GPU名の完全一致は要求せず、別GPUならtarget/actualのwarningを表示します。T4は最低VRAM条件で拒否します。元のKokoro profileとQwen profileは削除・上書きしていません。L4用は設定・本番音声・smoke出力を別名にしています。

## Performance benchmark

新CLI：`benchmark-performance --stage tts|asr|all`。

- train 200 pairs、seed=42、length-bin quota=4/30/80/66/20。全候補で同じsource records/text。
- WAVとcheckpointはlocal SSD `/content/kokoro-l4-benchmark`。Drive音声出力は拒否。
- レポートはDriveの `kokoro-performance/l4-performance.json`、日時付きsnapshot、logs。
- 各候補を新しいspawn subprocessで実行。ASR OOMは候補の `status=oom` として記録し、process終了後に次へ進む。
- ASRはTTS tuningの最初の全件成功候補の音声を固定して使う。各ASR候補の前後にWAV SHA256を確認する。
- 親がKokoro/Whisperの固定revisionを取得し、候補側はoffline/local資源を使う。
- 同じ条件・音声で品質を比較し、metric/QCが変わるASR候補を推奨しない。TTSはtext/voiceと大きなduration差を検査する。実L4でのworker間ASR品質比較は未実施。
- throughput最大を優先。headroom不足・generation failure・品質差のある候補は除外。推奨可能な候補がなければnull。
- configを自動変更しない。production中のautotuneなし。performance-lockはoptionalのため未実装。

測定値はaudio seconds/hours、wall、RTF、pairs/utterances per second、generation/empty/ASR failure、OOM、GPU名、per-process allocated/reserved peak等。benchmark wallにはload/startup/local WAV I/Oを含みます。

PyTorchのworker peak合計は同時刻のGPU全体peakではなく参考値です。per-worker・合計参考・最大workerを分け、benchmark時だけ約5秒間隔でnvidia-smiのGPU全体使用量もサンプルします。headroom警告には観測使用量とreserved参考値を併用し、90%超を推奨から除外します。サンプル間の短いpeakは捕捉できないため、最終1000件full benchmarkも必要です。productionではGPU pollingを追加せず、gpu_utilizationはnullです。

## Notebook

```text
Drive / repo / deps / L4 config / validation / GPU / preload / voice
→ 10 pairs → 100 pairs
→ TTS tuning → ASR tuning → recommendation
→ manual config confirmation
→ 1000 pairs full benchmark
→ plan/status → production 1 shard → remaining shards → consolidate
```

`RUN_L4_TUNING=False`、`RUN_BENCHMARK_1000=False`、`PERFORMANCE_CONFIG_CONFIRMED=False`、`APPROVE_PRODUCTION=False`、`RUN_ALL_REMAINING=False` が初期値です。L4設定はDriveの `kokoro-jparacrawl-l4-config.yaml` に保存し、既存設定を自動上書きしません。

## Tests

- 既存97件を含むローカルpytest：**119 passed**。既存tar展開に関するDeprecationWarning 2件のみ。GitHub CIの結果ではありません。
- 実spawn＋FakeBackendで固定20 utterancesをworker=1/2/4で比較。全件1回、正しいpath/record、voice、16 kHz mono PCM16、duration、SHA256、atomic writeを確認。実Kokoro/GPUの品質比較ではありません。
- worker CUDA errorの注入、実process crash(exit 17)、通常のempty generation failure、親checkpoint保存、100 utterance間隔、逆順completion、resume skip、orphan再生成、親storage budget差分を検査。
- L4 profile / T4拒否 / 別GPU warning、throughputと90%headroomによる推奨、ASR品質差の除外を検査。
- mock候補を使った200件抽出・全候補iteration・同じsample/audio・OOM後の次候補・report JSON・config不変を確認。
- ASR batch=8/16/24/32で同じmock文字起こしから同じmetric/QCを得ることを確認。実Whisper/L4比較は未実施。
- 既存Kokoro runtime checkもCPUで実行成功。日英各1文、JA=3.875秒、EN=3.800秒、native/canonical形式を確認。`data/kokoro-l4-runtime-check/report.json`（Git対象外）。これは逐次TTS単体確認であり、L4性能値やparallel品質確認ではありません。既知のWindows phonemizer終了時cleanup警告あり、returncodeは0。

## Production defaults

```text
kokoro worker_count: 1
asr batch_size: 8
checkpoint_every: 100 utterances
num_shards: 128
max_pairs_per_shard: 3000
```

**L4実測前なので最適値は未決定です。** 4 workerやASR batch=24が最適と判断したものではありません。

## 未実施

- Colab L4 10 pair smoke
- Colab L4 100 pair smoke
- L4 worker tuning
- L4 ASR batch tuning
- 最終設定で1000 pair full benchmark（CER/WER・duration・RTF・VRAM・CU確認）
- production 1 shard
- 300k production

上記を実測してから本番へ進みます。input index、export text、train/dev split、voice/revision、QC閾値、0.7〜30秒のduration条件、WAV integrity、streaming release、canonical orderは維持しています。
