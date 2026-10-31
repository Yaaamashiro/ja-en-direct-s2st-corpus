# 入力manifest I/O修正レポート

branch: `feature/kokoro-tts`。base commit: `8bf0ca8`。本修正のcommitは最終回答に記載します。

## 変更

- `input_index.py`：プロセス内で共有するローカルSQLite入力索引を追加。ordinal / pair_id / shard / full JSONを保存し、元JSONLを読む際に同じraw byte列からSHA256を計算します。
- `external.py`：export検証結果とidentityをcache。`run_input_sha256` は索引作成時のdigestを再利用し、同じmanifestをもう一度hashしません。smokeと元recordの照合も索引へqueryします。
- `runner.py`：external profileのshard取得・完了shard再開・recheck・planを索引へ接続。legacy Qwen/JESC/KFTTのshard取得は従来APIを維持。
- `cli.py`：起動時にcacheを初期化。statusは索引から `SELECT pair_id ... WHERE shard=?` し、completed shardごとの元manifest全走査を解消。
- `release.py`：検証済み入力索引をローカルSQLでコピーし、QC結果を結合。元inputを再parseせず、ordinal順と全integrity検証を維持。
- ノートブック：索引rootを `/content/s2st-input-index` に指定。本番gateは変更なし。

起動後の最初の検証で元JSONLを1回読みます。検証用のJSONチェックはローカル索引上で行い、その結果を同じCLIの全shardで共有します。各shardではmanifestとsummaryのsize / mtime / ctime / inode等を確認し、変更検出時は停止します。snapshotを途中で黙って再作成しません。

`consolidate` は全QC結果の検証・releaseのtmp書き込み後、公開の直前に元manifestとsummaryをSHA256で再確認します。smokeではsubsetも確認します。単独の `run-shard` / `run-next-shard` CLIも終了時に同じ最終確認をします。新しいCLIプロセスでは索引を作り直し、前プロセスのローカル索引を信用しません。

通常の `run-remaining-shards` では元manifest全体へのアクセスは、索引化の1回と最終ハッシュ確認の1回です。128 shardで512回相当の読み込みを繰り返す経路をなくしました。RAM使用量は従来どおり1 shard程度で、索引はローカルディスクに保持します。CLI終了時には所有する一時索引を削除します。強制切断で一時ファイルが残っても、次のCLIから再利用しません。

## ローカル検証

pytest: **97 passed**。既存tar展開に関するDeprecationWarning 2件のみ。GitHub CIの実行結果ではありません。

新規12ケース：

- 128 shard取得＋identity確認＋planで元manifestのopenは1回。
- completed shard=100のstatusも元manifestのopenは1回。
- manifest / summaryの途中変更をそれぞれ拒否。
- file属性チェックでは検出できない変更を模擬し、最終SHA256で拒否。
- 最終確認を含めても元manifestの全読み込みは2回。
- smoke用のshard再分割・元record照合は既存ローカル索引を使用。
- CLI再起動相当のcache初期化で再検証・再索引化。
- shard index / 最大件数の制約を維持。
- CRLF・空行を含む元ファイルのraw SHA256を正しく計算。
- 索引構築中の元ファイル変更を拒否し、失敗した一時索引を清掃。
- 索引のみを再初期化しても既存検証digestとの不一致を拒否。

既存のcanonical order・split release・publish marker・WAV format/SHA256/duration・provenance・resume・層化benchmark・Qwen回帰テストも通っています。入力本文、train/dev split、voice、revision、QC閾値は変更していません。

## 未実施

実JParaCrawl export validation、Colab T4の10件→100件→1000件実測、30万件productionは未実施です。実データの品質・RTF・CUは本修正のテストから推定していません。
