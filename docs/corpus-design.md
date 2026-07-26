# 日英合成音声ペアコーパス最終設計

更新日: 2026-07-23
方向: 日本語音声 → 英語音声
用途: Translatotron 2、S2UT、Cascade の比較
TTS: Qwen3-TTS 1.7B CustomVoice、日英同一プリセット話者

## 1. 最終構成

学習コーパスは次で固定する。

| データ | 役割 | 目標比率 |
|---|---|---:|
| Filtered JESC | 会話・口語・物語的表現 | 70% |
| KFTT | 説明文・長文・大きな語順変換 | 30% |

比率は文対数ではなく、採用後の**日本語音声時間**で管理する。

- JESC: 日本語約210時間
- KFTT: 日本語約90時間
- 完了条件: `min(total_ja_hours, total_en_hours) >= 300 h`
- 想定文対数: 約18万〜22万。最終値は2,000文パイロットの実測時間から決める。

外部評価には Business Scene Dialogue Corpus (BSD) を使う。BSDは学習に混ぜない。

```text
Training:
  Filtered JESC + KFTT

In-domain evaluation:
  JESC test
  KFTT test

External conversational evaluation:
  BSD evaluation
```

## 2. 研究上の位置づけ

このコーパスで検証するのは、音響条件を単一話者TTSで統制した場合に、近年のDirect S2STが日英翻訳と語順変換を学習できるかである。

主張できる範囲は次に限定する。

- 合成音声から合成音声へのオフライン文単位翻訳
- 単一話者・クリーン音声
- JESC、KFTT、BSDに含まれる文体
- データ量、文長、語順差に対するモデルの傾向

自然発話、複数話者、雑音、訛り、実環境への一般化は、この実験だけからは主張しない。

## 3. データソースと利用条件

### JESC

- 公式: https://nlp.stanford.edu/projects/jesc/
- 規模: raw 2,801,388文対、train 2,797,388、dev 2,000、test 2,000
- 内容: 映画・テレビ字幕由来の会話、口語、物語的表現
- ライセンス: CC BY-SA 4.0
- 使用法: trainから高品質部分だけを選び、dev/testは独立評価に使う

CC BY-SA 4.0は改変を許すが、帰属表示、変更表示、継承条件がある。

### KFTT

- 公式: https://www.phontron.com/kftt/
- 規模: train 約44万文対、dev 約1,166、test 約1,160
- 内容: 京都関連Wikipediaの説明文
- ライセンス: CC BY-SA 3.0
- 使用法: trainから時間・文長・語順差を層化して抽出し、公式dev/testを維持する

### BSD

- 公式: https://github.com/tsuruoka-lab/BSD
- 規模: train 20,000、dev 2,051、evaluation 2,120
- 内容: 人手で作成・翻訳された日英ビジネス会話
- ライセンス: CC BY-NC-SA
- 使用法: evaluation 2,120文を外部会話評価専用にする

JESC、KFTT、BSDはライセンスが異なるため、音声、マニフェスト、帰属情報をソース別に分離する。公開可否はソースごとに確認し、初期版は研究室内利用を前提とする。

## 4. split設計

公式splitを優先し、文単位のランダム再分割は行わない。

| split | 内容 |
|---|---|
| `train_jesc` | JESC公式trainから選別 |
| `train_kftt` | KFTT公式trainから選別 |
| `dev_jesc` | JESC公式devの採用可能例 |
| `dev_kftt` | KFTT公式devの採用可能例 |
| `test_jesc` | JESC公式testの人手確認済み例 |
| `test_kftt` | KFTT公式testの採用可能例 |
| `test_bsd` | BSD公式evaluationの採用可能例 |
| `test_balanced` | JESC/KFTTから同数を層化抽出した総合ビュー |

評価例と完全一致・近似一致する文は、学習側から除外する。JESCでは作品単位の独立性を前提にせず、公式splitに加えて正規化後の完全重複、MinHash近似重複、日英片側一致を検査する。

学習曲線用に、次の入れ子集合を固定seedで作る。

```text
train_50h ⊂ train_150h ⊂ train_300h ⊆ train_full
```

時間は日本語入力音声時間である。各集合でJESC/KFTT比、文長分布、語順差分布を可能な限り維持する。`train_full`は日英双方300時間以上になるまで追加した最終集合である。

## 5. テキスト表現

各言語について四つの表現を保持する。

```text
text_raw        # 配布物そのまま
text_norm       # 最小正規化後。重複判定・フィルタ用
tts_text        # Qwen3-TTSに実際に渡した文字列
text_eval_norm  # ASRとのCER/WER計算用
```

rawを上書きしない。すべての正規化、読み変換、手修正を別列と履歴に保存する。

最小正規化は次に固定する。

- Unicode `NFKC`
- 改行、タブ、連続空白を半角空白一つへ統一
- 前後空白を削除
- 制御文字、ゼロ幅文字を除去
- HTML/XML実体を復元
- 変換規則に `normalizer_version` を付ける

TTS入力では日本語の漢字・仮名、英語の大文字小文字、両言語の自然な句読点を維持する。URL、メールアドレス、数式、字幕タグ、読みが曖昧な記号列は、本文の一部を黙って削除せず文対ごと除外する。数字と略語は原則として保持し、パイロットで系統的誤読が確認された形式だけ、版管理された置換表で修正する。

## 6. JESCの選別

JESCは量が十分あるため、全件を使わず品質を優先する。

### 初期除外

- 空文、URL、HTML/XML、制御文字
- 話者名だけの行
- `(Laughter)`、`[Music]` 等の効果音だけの行
- 記号・数字だけの行
- 日本語8文字未満、英語4語未満
- 日本語160文字超、英語50語超
- 日英の長さ比が極端な文
- 言語判定に失敗した文
- 完全・近似重複
- 明らかな不適切アラインメント

### 対訳品質

多言語文埋め込みで日英類似度を計算する。ただし固定閾値で機械的に切らず、2,000文パイロットの人手判定から閾値を決める。

短い字幕に偏らないよう、採用済み日本語音声時間を次の目安で構成する。

| 文長層 | 目標 |
|---|---:|
| short | 20% |
| medium | 50% |
| long | 30% |

極端な暴力・差別・性的表現は、研究に不要な生成を避けるため、必要に応じて除外フラグを適用する。除外規則と件数は報告する。

## 7. KFTTの選別

KFTTは高品質な説明文として使うが、京都ドメインに偏るため約30%に制限する。

- 公式trainから抽出する
- 日本語8〜180文字、英語4〜60語を初期範囲とする
- URL、表、数式、読み上げ困難な記号列を除く
- 固有名詞、年号、寺社名の読みを重点検査する
- 短・中・長と語順差small/medium/largeを層化する
- 公式dev/testは学習に入れない

## 8. 日本語TTSテキストと読み検査

Qwen3-TTSには、正規化した日本語表記をそのまま渡す。カタカナ読みはTTS入力へ置換せず、読みの監査とASR-CER計算だけに使う。

```text
ja_text_raw
  → Unicode・空白・記号の正規化
  → 数字、年号、略語の読み上げ表記を決定
  → ja_tts_text
  → Qwen3-TTS

ja_tts_text
  → OpenJTalk系の固定辞書
  → ja_reading_kana（QC用）
```

保存項目:

```text
ja_text_raw
ja_text_norm
ja_tts_text
ja_reading_kana
ja_reading_engine
ja_reading_version
ja_reading_override
```

- raw、評価用正規化、TTS入力、QC用読みを混同しない
- train: ASRで読みを検査し、重大な誤読はTTS入力の修正、再生成、または除外
- dev/test: `ja_tts_text` と生成音声を全件人手確認
- 手修正はrawを変えず、overrideと変更履歴として保存

## 9. 日英同一プリセット話者

初期候補は `Ono_Anna` とする。公式説明では日本語母語のプリセット話者であり、全対応言語を発話できる一方、最高品質には各話者の母語利用が推奨されている。したがって英語側は本生成前の実測で採否を決める。

フル生成前にJESC/KFTTから文長、数字・固有名詞、語順差を層化した500文対を選び、同じ `Ono_Anna` で日英合計1,000音声を生成して次を確認する。

- 日本語読み正規化CER
- 英語WER
- 日本語・英語生成音声間のspeaker similarity
- 無音、反復、途中終了率
- 日英各50件以上の人手聴取
- 生成速度、VRAM、再生成率

暫定合格条件は、言語別のコーパス集計CER/WERが5%以下、CER/WERが20%を超える発話が5%以下、重大な無音・反復・言語誤りが1%以下とする。短文では一語の誤りで率が大きくなるため、最終判定は編集数と人手聴取も併用する。

`Ono_Anna` の英語が不合格の場合だけ、同じ500文対で `Aiden` と `Ryan` を日英双方に適用して比較する。明瞭度を第一、日英speaker similarityを第二基準として一人を選び、フル生成では一つのspeaker IDだけを使う。複数話者の音声を同じ学習集合に混ぜない。

## 10. Qwen3-TTS固定条件

- モデル: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- モデルrevision: `f00cf133b78d3c2c35857faba3b1be9b98c4f971`
- 公式実装: https://github.com/QwenLM/Qwen3-TTS
- モデル: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
- ライセンス表示: Apache-2.0。生成コーパスの公開条件は元テキストのライセンスも含めて別途確認する
- `generate_custom_voice` を使用する
- `language="Japanese"` / `language="English"` を明示し、`Auto` は使わない
- 初期speakerは日英とも `speaker="Ono_Anna"`
- `instruct=""` とし、感情、速度、音量、息、笑い等の指示は使用しない
- `torch.bfloat16`、非ストリーミング生成を基本とする
- `max_new_tokens=2048` とし、その他のsampling parameterは固定revisionの `generation_config.json` を使う
- `generation_config.json` のSHA-256をrelease metadataへ保存する
- モデルrevision、公式コードcommit、`qwen-tts`、PyTorch、CUDA、FlashAttentionの版を固定する
- seedは `SHA-256(pair_id + NUL + language + NUL + attempt)` の先頭8 byteを符号なし整数化し、63 bitに制限して決める
- 発話ごとにPython、NumPy、PyTorch、全CUDA deviceのseedを設定する
- canonical生成は `batch_size=1`、`workers_per_gpu=1` から開始する
- フル生成開始後にspeaker、設定、依存関係を変えない。変更時はcorpus major versionを上げる

## 11. 音声形式

本番releaseでは容量を100 GB未満に抑えるため、Qwen3-TTSが返すnative波形は一時データとして扱い、既定では永続保存しない。話者選定・pilotなど、音質監査が必要な小規模実験だけ`save_native=true`で保存する。

### canonical学習用

- 16 kHz
- mono
- signed PCM 16-bit WAV
- 決定論的リサンプリング
- 冒頭末尾の無音を除き、各端100 msを残す
- 話速、ピッチ、雑音、残響、時間伸縮を加えない
- 原則として音量正規化を行わない

音声健全性の初期条件:

- duration 0.7〜30.0秒
- 破損、NaN、全無音なし
- 異常なクリッピング、DC、音量なし
- 反復、途中終了、異常に長い無音なし

## 12. TTS品質管理

### ASR固定条件

品質管理、日本語Cascade ASR、生成英語音声の最終文字起こしは、すべて同じOpenAI Whisper checkpointに統一する。

```yaml
model_id: openai/whisper-large-v3-turbo
revision: 60be3615a4d667e1258e8ad29130467587c489aa
task: transcribe
do_sample: false
num_beams: 1
condition_on_prev_tokens: false
return_timestamps: false
chunking: false
```

- モデル: https://huggingface.co/openai/whisper-large-v3-turbo
- ライセンス表示: MIT
- 日本語では `language="japanese"`、英語では `language="english"` を明示する
- 言語自動判定と音声翻訳モードは使わない
- 16 kHz音声を入力し、30秒以下の文単位音声は分割しない
- モデルrevision、Transformers、PyTorch、推論設定を固定する
- QCと最終ASR-BLEUで同じASRを使うため、その依存性を研究の限界として記載する
- 参照英語TTS音声にも同じASRを適用し、ASR-BLEUとWERの実測上限を必ず報告する
- testの人手聴取と異常出力率を併記し、ASRスコアだけで結論を出さない

### 内容一致

- 日本語: 参照とASR結果を同じOpenJTalk系読み生成器へ通し、NFKC、カタカナ統一、句読点・空白除去後にCER
- 英語: 参照とASR結果へ同じWhisper `EnglishTextNormalizer`を適用後にWER
- 初期閾値: 日本語CER 10%、英語WER 10%
- 最終閾値: 500文対のバイリンガル話者試験と2,000文対パイロットの聴取結果で固定

再試行はシステム失敗と内容失敗を分ける。

- OOM、I/O、worker停止などのシステム失敗は同じseedで再実行し、`attempt`を増やさない
- CER/WER、無音、反復、途中終了などの内容失敗は `attempt=1` の別seedで一度だけ再生成する
- attempt 0だけ合格なら0、attempt 1だけ合格なら1を採用する
- 両方合格ならCER/WERが低い方、同値ならattempt 0を採用する
- 両方不合格なら除外する

WAVは一時ファイルへ書き、形式検査とSHA-256計算に成功してからatomic renameする。不合格例は削除せず、理由をfailure ledgerに残す。

ASR選別で難しい文だけが消えないよう、除外率を次の属性別に報告する。

- JESC/KFTT
- 文長
- 語順差
- 固有名詞・数字の有無
- 日本語読み修正の有無

### 人手確認

- JESC test: 対訳と日本語入力音声を全件確認
- KFTT test: 日本語入力音声を全件確認
- BSD evaluation: 対訳と日本語入力音声を全件確認
- dev: 各言語10%
- train: 1万件ごとに日英各100件

ASRは全工程で同じcheckpointを使うが、用途ごとの出力、正規化結果、評価結果は別ファイルに保存する。

## 13. 語順差アノテーション

日本語形態素、英語token、日英word alignmentから次を計算する。

```text
alignment_coverage
crossing_ratio
reorder_kendall
mean_position_shift
reorder_bin = small | medium | large
```

bin境界はtrainの有効アラインメント例の三分位点から決め、dev/test/BSDにも同じ境界を使う。alignment coverageが低い例は学習から除外せず、語順別分析だけから外す。

## 14. モデル比較

三方式は同一の文対、split、日本語入力音声、英語参照文を使う。

| 方式 | 学習対象 |
|---|---|
| Translatotron 2 | 日本語音声 → 英語音素・メル → 波形 |
| S2UT | 日本語音声 → 英語unit → unit vocoder |
| Cascade | 同じWhisper日本語ASR → 日英MT → 同じ英語TTS |

評価:

- ASR-BLEU
- ASR-chrF
- COMET（補助）
- 出力失敗率、無音率、反復率
- 文長別スコア
- 語順差別スコア
- JESC/KFTT/BSD別スコア
- CascadeのASR CER
- Oracle-ASR Cascade
- 参照英語TTSをASRに通した評価上限

主結果は `train_300h` または `train_full` で比較し、50/150/300時間で学習曲線を示す。

## 15. マニフェスト

正本はParquet、交換用にTSV/JSONLを出力する。

```text
pair_id
corpus
corpus_version
license_id
original_split
split
document_id
segment_id
translation_origin
ja_text_raw
en_text_raw
ja_text_norm
en_text_norm
ja_tts_text
en_tts_text
ja_text_eval_norm
en_text_eval_norm
normalizer_version
ja_reading_kana
ja_reading_engine
ja_reading_version
ja_reading_override
filter_flags
alignment_coverage
reorder_kendall
reorder_bin
tts_speaker_id
tts_language_ja
tts_language_en
tts_instruct_sha256
tts_model_id
tts_model_revision
tts_code_revision
tts_package_version
tts_generation_config_sha256
tts_seed_ja
tts_seed_en
tts_attempt_ja
tts_attempt_en
qc_asr_model_id
qc_asr_model_revision
qc_asr_decode_config_sha256
ja_wav_native     # 既定ではnull
en_wav_native     # 既定ではnull
ja_wav_16k
en_wav_16k
ja_duration
en_duration
ja_sample_rate_native
en_sample_rate_native
ja_sha256
en_sha256
ja_asr_text
en_asr_text
ja_cer_kana
en_wer
sim_ja_en
qc_status
qc_reasons
manual_review_status
created_at
```

`pair_id`はソース、元ID、raw text hashから決定論的に作る。

## 16. ディレクトリ

```text
data/
  raw/
    jesc/
    kftt/
    bsd/
  interim/
    text/
    readings/
    alignments/
    jobs/
  audio/
    16k/{ja,en}/<shard>/
  manifests/
    source/
    qc/
    releases/
  reports/
  provenance/
    licenses/
    model_cards/
    environment_locks/
configs/
docs/
scripts/
```

配布物、生成音声、モデル重みはGitに入れない。Gitでは設計、設定、スクリプト、統計、公開可能なprovenanceのみ管理する。

## 17. 段階的実行

### Gate 0: 権利・TTS

- JESC/KFTT/BSDの利用条件と公開範囲を記録
- Qwen3-TTSのモデルカード、ライセンス、model revisionを保存
- Whisperのモデルカード、ライセンス、model revisionを保存
- 公式コードcommitと実行環境lockを固定
- 保存先、バックアップ、計算環境を確定

### Gate 1: 5文 production smoke test

- JESC 3文、KFTT 2文、short/medium/long、日英を含む
- TTS入力正規化、QC用読み生成、TTS、再開、16 kHz化、ASR-QC、manifestまで通す

### Gate 2: 500文対 バイリンガル話者試験

- `Ono_Anna` の日英明瞭度と日英間話者類似度を確認
- 不合格時だけ `Aiden`、`Ryan` を同じ集合で比較
- Qwen3-TTS model、revision、commit、speaker、生成設定を固定

### Gate 3: 2,000文 pilot

- 採用率、CER/WER、平均時間、RTF、容量を実測
- JESC品質閾値、TTS QC閾値、文長閾値を固定
- 300時間に必要な文対数と計算時間を再見積り
- major errorを含む採用品が人手標本の5%を超える場合は停止

### Gate 4: 最初の10,000文

- ドメイン、文長、語順差、固有名詞別の失敗率を監査
- JESC/KFTTの採用時間比を確認

### Gate 5: full release

- 日英双方300時間以上
- JESC/KFTT日本語時間比 70/30 ±2ポイント
- split間完全重複ゼロ
- 近似重複監査済み
- testの人手確認済み
- 全音声にSHA-256、設定ハッシュ、QC結果あり
- 50/150/300時間集合が入れ子
- ソース別・文長別・語順差別統計を生成

## 18. 容量

16 kHz / mono / PCM16は1時間あたり約107.3 MiBである。300時間×2言語で約64.37 GiB、再生成10%を含め約70.81 GiBになる。本番パイプラインの音声・manifest出力hard limitは78 GiBとし、native WAVは保存しない。

別途、Dockerイメージに約16 GB、QwenとWhisperのHugging Face cacheに約7 GBを見込む。したがって同一ドライブで全てを保持すると約100 GBに近づく。容量が厳しい環境では、生成先だけを80 GB以上空いた外付けSSDへ置く。

生成時間は推測せず、2,000文pilotの実測RTFから算出する。

## 19. 実装順

```text
1. ingest
2. normalize
3. deduplicate
4. filter-and-score
5. prepare-tts-text-and-readings
6. annotate-reordering
7. select
8. synthesize
9. qc
10. build-subsets
11. release
12. audit
```

最初に実装するのは、100文について `text → TTS text + QC reading → TTS → 16 kHz → ASR-QC → manifest` が最後まで再開可能に動くパイプラインである。
