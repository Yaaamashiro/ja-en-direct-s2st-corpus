# Data attribution

リポジトリ内のソースコードには [Apache License 2.0](LICENSE) が適用されます。
この文書は、パイプラインが扱う外部データと生成物の条件を示します。

このパイプラインは次の公式配布物をダウンロードし、正規化・選別後に
合成音声へ変換します。配布物と生成物を再配布する場合は、各ライセンスの
帰属表示、変更表示、継承条件を確認してください。

## JESC

- 名称：Japanese-English Subtitle Corpus
- 著者：Reid Pryzant, Youngjoo Chung, Dan Jurafsky, Denny Britz
- 公式ページ：https://nlp.stanford.edu/projects/jesc/
- 使用版：2019-05-12 official split
- ライセンス：CC BY-SA 4.0
- 変更：正規化、フィルタ、重複除去、部分選別、TTS音声化

引用：

```bibtex
@article{pryzant_jesc_2018,
  author = {Pryzant, Reid and Chung, Youngjoo and Jurafsky, Dan and Britz, Denny},
  title = {JESC: Japanese-English Subtitle Corpus},
  journal = {Language Resources and Evaluation Conference (LREC)},
  year = {2018}
}
```

## KFTT

- 名称：Kyoto Free Translation Task
- 著者：Graham Neubig
- 公式ページ：https://www.phontron.com/kftt/
- 使用版：Data Only v1.0
- ライセンス：CC BY-SA 3.0
- 変更：正規化、フィルタ、重複除去、部分選別、TTS音声化

引用：

```bibtex
@misc{neubig11kftt,
  author = {Graham Neubig},
  title = {The Kyoto Free Translation Task},
  howpublished = {http://www.phontron.com/kftt},
  year = {2011}
}
```

このファイルに記載した外部データの条件は要約です。利用・再配布前に、
各公式ページのライセンス本文も確認してください。
