import json
import shutil

from src.common.config import load_yaml, resolve_path, validate_experiment_config
from src.experiments.atlas_runner import load_atlas_config
from .runner import PROJECT_ROOT, prepare_run


def run_realtime(config_path):
    config = validate_experiment_config(load_yaml(config_path), "realtime")
    output = prepare_run(config, resolve_path(config["experiment"]["output_dir"], PROJECT_ROOT))
    tools = {name: bool(shutil.which(name)) for name in ("gst-launch-1.0", "tc")}
    network = config.get("network", {})
    if network.get("enabled"):
        raise RuntimeError("Network shaping must be launched explicitly with setup_netem.sh and always paired with reset_netem.sh")
    atlas_method = config.get("methods", {}).get("bistar_atlas", {})
    atlas = None
    if atlas_method.get("enabled"):
        atlas = load_atlas_config(resolve_path(atlas_method["config"], PROJECT_ROOT))["codec"]
    status = {"status": "ready" if all(tools.values()) else "dependency_missing", "tools": tools,
              "evaluation_mode": config["experiment"].get("evaluation_mode", "causal"),
              "bistar_atlas": {
                  "enabled": bool(atlas),
                  "gop_size": atlas["gop_size"] if atlas else None,
                  "mask_encoding": atlas["control"]["mask_encoding"] if atlas else None,
                  "send_mask_once_per_gop": atlas["control"]["send_once_per_gop"] if atlas else None,
                  "encoder": atlas["encoder"]["codec_name"] if atlas else None,
                  "b_frames": atlas["encoder"]["b_frames"] if atlas else None,
                  "patch_budget_strategy": (
                      "fixed" if atlas and atlas["selector"].get("patch_budget_mode", True)
                      else "adaptive_gop" if atlas else None
                  ),
                  "min_visible_patches": atlas["selector"].get("min_visible_patches") if atlas else None,
                  "max_visible_patches": atlas["selector"].get("max_visible_patches") if atlas else None,
              }}
    (output / "summary.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
