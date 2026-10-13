from pathlib import Path
from typing import Any, Dict

import yaml


class ConfigError(ValueError):
    pass


def load_yaml(path: Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError("Configuration does not exist: {}".format(path))
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigError("Top-level YAML value must be a mapping: {}".format(path))
    return data


def require(mapping: Dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ConfigError("Missing required fields: {}".format(", ".join(missing)))


def load_codec_config(path: Path) -> Dict[str, Any]:
    data = load_yaml(path)
    require(data, "codec")
    codec = data["codec"]
    require(codec, "name", "implementation", "pixel_format", "encoder", "output")
    require(codec["encoder"], "codec_name", "preset", "mode", "gop_size", "b_frames")
    require(codec["output"], "container", "extension")
    allowed = {"name", "implementation", "pixel_format", "encoder", "output"}
    unknown = set(codec) - allowed
    if unknown:
        raise ConfigError("Unknown critical codec fields: {}".format(sorted(unknown)))
    return data


def load_bistar_config(path: Path) -> Dict[str, Any]:
    data = load_yaml(path)
    require(data, "codec")
    codec = data["codec"]
    require(codec, "name", "implementation", "patch_size", "selector", "payload_encoding", "reconstruction")
    require(codec["payload_encoding"], "mode", "quality")
    if codec["payload_encoding"]["mode"] not in ("png", "jpeg"):
        raise ConfigError("BiSTAR payload_encoding.mode must be png or jpeg")
    quality = int(codec["payload_encoding"]["quality"])
    if quality < 0 or quality > 100:
        raise ConfigError("BiSTAR payload_encoding.quality must be between 0 and 100")
    require(codec["reconstruction"], "type")
    if codec["reconstruction"]["type"] not in ("previous_frame", "interpolation"):
        raise ConfigError("BiSTAR reconstruction.type must be previous_frame or interpolation")
    return data


def resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def validate_experiment_config(data: Dict[str, Any], kind: str) -> Dict[str, Any]:
    shared = {"experiment", "dataset"}
    allowed = {
        "rd": shared | {"rd", "methods", "bistar", "quality_metrics", "metrics", "runtime", "logging", "smoke"},
        "matched_bitrate": shared | {"matched_bitrate", "methods", "bistar", "bistar_atlas", "metrics", "runtime", "logging"},
        "ablation": shared | {"ablation", "bistar", "bistar_atlas", "metrics", "runtime", "logging"},
        "realtime": {"experiment", "network", "methods", "runtime", "logging"},
    }
    if kind not in allowed: raise ConfigError("Unknown experiment kind: {}".format(kind))
    unknown = set(data) - allowed[kind]
    if unknown: raise ConfigError("Unknown experiment fields: {}".format(sorted(unknown)))
    require(data, "experiment")
    require(data["experiment"], "name", "seed", "output_dir")
    if kind != "realtime": require(data, "dataset")
    required_section = kind if kind != "rd" else "rd"
    if kind in ("rd", "matched_bitrate", "ablation"): require(data, required_section)
    return data
