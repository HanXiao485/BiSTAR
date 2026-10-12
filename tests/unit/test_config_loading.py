import pytest

from src.common.config import ConfigError, load_bistar_config, load_codec_config, validate_experiment_config


def test_codec_configs_are_valid():
    assert load_codec_config("configs/codecs/h264.yaml")["codec"]["name"] == "h264"
    bistar = load_bistar_config("configs/codecs/bistar.yaml")["codec"]
    assert bistar["payload_encoding"] == {"mode": "jpeg", "quality": 50}


def test_unknown_critical_field_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"; path.write_text("codec: {name: x, implementation: x, pixel_format: x, encoder: {}, output: {}, surprise: true}")
    with pytest.raises(ConfigError): load_codec_config(path)


def test_unknown_experiment_field_is_rejected():
    with pytest.raises(ConfigError):
        validate_experiment_config({"experiment": {"name":"x","seed":0,"output_dir":"x"}, "dataset": {}, "rd": {}, "surprise": True}, "rd")
