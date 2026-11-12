# Third-party attribution

The repository's source code is available under the [MIT License](LICENSE). Separate terms apply to the third-party datasets, models, and derived outputs listed below.

The pipeline obtains the JESC and KFTT archives and Qwen3-TTS and Whisper weights from their official distribution sources at runtime. Git tracks the code and configuration. Generated audio is stored in the run output.

## Datasets

### JESC

- **Name** — Japanese-English Subtitle Corpus
- **Authors** — Reid Pryzant, Youngjoo Chung, Dan Jurafsky, and Denny Britz
- **Official page** — <https://nlp.stanford.edu/projects/jesc/>
- **Version** — Official splits dated 2019-05-12
- **License** — CC BY-SA 4.0. [License text](https://creativecommons.org/licenses/by-sa/4.0/)
- **Pipeline changes** — Normalization, filtering, deduplication, subset selection, speech synthesis, resampling to 16 kHz, and ASR quality checks.

Citation

```bibtex
@article{pryzant_jesc_2018,
  author = {Pryzant, Reid and Chung, Youngjoo and Jurafsky, Dan and Britz, Denny},
  title = {JESC: Japanese-English Subtitle Corpus},
  journal = {Language Resources and Evaluation Conference (LREC)},
  year = {2018}
}
```

### KFTT

- **Name** — Kyoto Free Translation Task
- **Author** — Graham Neubig
- **Official page** — <https://www.phontron.com/kftt/>
- **Version** — Data Only v1.0
- **License** — CC BY-SA 3.0. [License text](https://creativecommons.org/licenses/by-sa/3.0/)
- **Pipeline changes** — Normalization, filtering, deduplication, subset selection, speech synthesis, resampling to 16 kHz, and ASR quality checks.

Citation

```bibtex
@misc{neubig11kftt,
  author = {Graham Neubig},
  title = {The Kyoto Free Translation Task},
  howpublished = {http://www.phontron.com/kftt},
  year = {2011}
}
```

### JParaCrawl v3 (external manifest input)

- **Provider** — NTT. The pipeline uses English–Japanese v3.0 text selected and exported by the upstream repository.
- **Terms** — The published terms cover research and development involving information analysis and apply the same conditions to derived data. The standard terms prohibit commercial use. Contact NTT about commercial use. Cite the official terms when describing the dataset's licensing. [JParaCrawl](https://www.kecl.ntt.co.jp/icl/lirg/jparacrawl/)
- **Paper** — Makoto Morishita, Katsuki Chousa, Jun Suzuki, and Masaaki Nagata (2022). [JParaCrawl v3.0: A Large-scale English-Japanese Parallel Corpus](https://aclanthology.org/2022.lrec-1.721/)
- **Pipeline changes** — Export validation, speech synthesis, resampling to 16 kHz, and ASR quality checks. The pipeline preserves the exported records, including their corpus, split, and length bin, and records the export's SHA-256 digest.

## Models

### Kokoro-82M

- **Provider** — hexgrad. Model `hexgrad/Kokoro-82M`, v1.0.
- **Pinned revision** — `f3ff3571791e39611d31c381e3a41a3af07b4987`.
- **Weight license** — Apache License 2.0. [Pinned model card](https://huggingface.co/hexgrad/Kokoro-82M/blob/f3ff3571791e39611d31c381e3a41a3af07b4987/README.md)
- **Official runtime** — The pipeline uses `kokoro==0.9.4`. [hexgrad/kokoro](https://github.com/hexgrad/kokoro)
- **Japanese and English G2P** — The pipeline uses `misaki==0.9.4`. UniDic, pyopenjtalk, and eSpeak NG have their own terms. [hexgrad/misaki](https://github.com/hexgrad/misaki)
- **Scope** — The model license covers the weights. The JParaCrawl input and derived corpus are governed by the data terms above. The runtime fetches the weights and dependencies separately.

### Qwen3-TTS

- **Provider** — Qwen Team, Alibaba Cloud
- **Model ID** — `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`
- **Pinned revision** — `f00cf133b78d3c2c35857faba3b1be9b98c4f971`
- **Use** — Japanese and English speech synthesis
- **License** — Apache License 2.0. [License text](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE)
- [Pinned model page](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice/tree/f00cf133b78d3c2c35857faba3b1be9b98c4f971)
- [Source code](https://github.com/QwenLM/Qwen3-TTS)

Citation

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

- **Provider** — OpenAI
- **Model ID** — `openai/whisper-large-v3-turbo`
- **Pinned revision** — `60be3615a4d667e1258e8ad29130467587c489aa`
- **Use** — Japanese CER and English WER quality checks for synthesized speech
- **License** — MIT License. [License text](https://github.com/openai/whisper/blob/main/LICENSE)
- [Pinned model page](https://huggingface.co/openai/whisper-large-v3-turbo/tree/60be3615a4d667e1258e8ad29130467587c489aa)
- [Source code](https://github.com/openai/whisper)

Citation

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

- When distributing the generated corpus, include this file and preserve `corpus`, `source_id`, `corpus_version`, and `license_id` in the manifests.
- Keep the applicable dataset terms for JESC-derived and KFTT-derived outputs.
- When distributing weights, a model cache, or a container that includes them, retain the licenses and copyright notices for the models and bundled software.

This file summarizes third-party terms. Consult the official pages and license texts before use or redistribution.
