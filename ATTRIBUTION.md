# Third-party attribution

リポジトリ内の自作ソースコードには [MIT License](LICENSE) が適用されます。
このライセンスは、以下の外部データ、モデル、モデル出力へ一括して適用
されるものではありません。

JESC・KFTTの配布物、生成音声、Qwen3-TTS・Whisperのモデル重みはGitへ
収録せず、実行時に公式配布元から取得します。

## Datasets

### JESC

- 名称：Japanese-English Subtitle Corpus
- 著者：Reid Pryzant, Youngjoo Chung, Dan Jurafsky, Denny Britz
- 公式ページ：https://nlp.stanford.edu/projects/jesc/
- 使用版：2019-05-12 official split
- ライセンス：CC BY-SA 4.0
- ライセンス本文：https://creativecommons.org/licenses/by-sa/4.0/
- 本パイプラインによる変更：正規化、フィルタ、重複除去、部分選別、
  TTS音声化、16 kHzへのリサンプリング、ASR品質検査

引用：

```bibtex
@article{pryzant_jesc_2018,
  author = {Pryzant, Reid and Chung, Youngjoo and Jurafsky, Dan and Britz, Denny},
  title = {JESC: Japanese-English Subtitle Corpus},
  journal = {Language Resources and Evaluation Conference (LREC)},
  year = {2018}
}
```

### KFTT

- 名称：Kyoto Free Translation Task
- 著者：Graham Neubig
- 公式ページ：https://www.phontron.com/kftt/
- 使用版：Data Only v1.0
- ライセンス：CC BY-SA 3.0
- ライセンス本文：https://creativecommons.org/licenses/by-sa/3.0/
- 本パイプラインによる変更：正規化、フィルタ、重複除去、部分選別、
  TTS音声化、16 kHzへのリサンプリング、ASR品質検査

引用：

```bibtex
@misc{neubig11kftt,
  author = {Graham Neubig},
  title = {The Kyoto Free Translation Task},
  howpublished = {http://www.phontron.com/kftt},
  year = {2011}
}
```

### JParaCrawl v3（external manifest profileの入力）

- 提供元：NTT。English–Japanese v3.0を外部リポジトリで選別・exportしたテキストを使用。
- 公式情報・利用規約：[JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
- 同ページの「対訳データ，単言語データおよび学習済みモデル利用に関する利用規約」は、情報解析を伴う研究開発目的の利用を定め、派生データにも同条件を適用しています。掲載条件では商用利用は認められておらず、商用利用は提供元への問い合わせが案内されています。CC BY等と推測して表記しないでください。
- 論文：[JParaCrawl v3.0: A Large-scale English-Japanese Parallel Corpus](https://aclanthology.org/2022.lrec-1.721/)（Makoto Morishita, Katsuki Chousa, Jun Suzuki, Masaaki Nagata; 2022）。
- 本リポジトリの処理：export検証、TTS音声化、16 kHzへのリサンプリング、ASR品質検査。再フィルタリング・再ランキングはしません。元のcorpus/split/length_bin等とexportファイルのSHA256を保持します。

## Models

### Kokoro-82M

- 提供者：hexgrad。モデル `hexgrad/Kokoro-82M`、v1.0。
- 固定revision：`f3ff3571791e39611d31c381e3a41a3af07b4987`。
- 重みのライセンス：Apache License 2.0（[固定revisionのモデルカード](https://huggingface.co/hexgrad/Kokoro-82M/blob/f3ff3571791e39611d31c381e3a41a3af07b4987/README.md)）。
- 公式ランタイム：[hexgrad/kokoro](https://github.com/hexgrad/kokoro)、使用package `kokoro==0.9.4`。
- 日英G2P：[hexgrad/misaki](https://github.com/hexgrad/misaki)、使用package `misaki==0.9.4`。UniDic・pyopenjtalk・eSpeak NG等の依存資源にもそれぞれの利用条件があります。
- モデルのライセンスを、入力のJParaCrawlテキストやその派生コーパス全体に一括適用しません。重みや依存資源はGitに収録しません。

### Qwen3-TTS

- 提供者：Qwen Team, Alibaba Cloud
- モデルID：`Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- 固定revision：`f00cf133b78d3c2c35857faba3b1be9b98c4f971`
- 用途：日本語・英語の合成音声生成
- ライセンス：Apache License 2.0
- 固定モデルページ：https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/f00cf133b78d3c2c35857faba3b1be9b98c4f971
- ソースコード：https://github.com/QwenLM/Qwen3-TTS
- ライセンス本文：https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE

引用：

```bibtex
@article{Qwen3-TTS,
  title = {Qwen3-TTS Technical Report},
  author = {Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang
    and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao
    and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang
    and Jin Xu and Jingren Zhou and Junyang Lin},
  journal = {arXiv preprint arXiv:2601.15621},
  year = {2026}
}
```

### Whisper

- 提供者：OpenAI
- モデルID：`openai/whisper-large-v3-turbo`
- 固定revision：`60be3615a4d667e1258e8ad29130467587c489aa`
- 用途：合成音声の日本語CER・英語WER品質検査
- ライセンス：MIT License
- 固定モデルページ：https://huggingface.co/openai/whisper-large-v3-turbo/tree/60be3615a4d667e1258e8ad29130467587c489aa
- ソースコード：https://github.com/openai/whisper
- ライセンス本文：https://github.com/openai/whisper/blob/main/LICENSE

引用：

```bibtex
@article{radford2022robust,
  title = {Robust Speech Recognition via Large-Scale Weak Supervision},
  author = {Alec Radford and Jong Wook Kim and Tao Xu and Greg Brockman
    and Christine McLeavey and Ilya Sutskever},
  journal = {arXiv preprint arXiv:2212.04356},
  year = {2022}
}
```

## Redistribution

- 生成コーパスを配布する場合は、このファイルを同梱し、各manifestの
  `corpus`、`source_id`、`corpus_version`、`license_id`を保持してください。
- JESC由来とKFTT由来の生成物を、リポジトリのMIT Licenseや単一の別
  ライセンスへ一括して変更しないでください。
- モデル重み、モデルcache、またはそれらを含むコンテナを配布する場合は、
  各モデルと同梱ソフトウェアのライセンス本文・著作権表示も保持してください。

この文書に記載した外部資源の条件は要約です。利用・再配布前に、各公式
ページとライセンス本文も確認してください。
