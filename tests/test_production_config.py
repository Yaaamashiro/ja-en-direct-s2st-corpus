from pathlib import Path

from s2st_corpus.config import load_config


def test_production_profile_is_pinned_and_storage_bounded() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "production-qwen17b.yaml"
    )
    config = load_config(path)
    assert config.run.num_shards == 128
    assert config.run.maximum_output_gib < 100
    assert config.run.minimum_cache_free_gib == 8
    assert config.run.save_native is False
    assert config.tts.model_id.endswith("1.7B-CustomVoice")
    assert len(config.tts.revision) == 40
    assert config.tts.device == "cuda:0"
    assert config.tts.dtype == "bfloat16"
    assert config.tts.speaker == "Ono_Anna"
    assert config.asr.model_id == "openai/whisper-large-v3-turbo"
    assert config.device.require_bf16 is True
    assert config.sources["jesc"].version == "2019-05-12"
    assert config.sources["kftt"].version == "1.0"
    assert config.smoke.pair_count == 5
    assert config.smoke.output_dir != config.run.output_dir
