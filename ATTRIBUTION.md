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

## Models

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
