import json
import shutil

from src.common.config import load_yaml, resolve_path, validate_experiment_config
from .runner import PROJECT_ROOT, prepare_run


def run_realtime(config_path):
    config = validate_experiment_config(load_yaml(config_path), "realtime")
    output = prepare_run(config, resolve_path(config["experiment"]["output_dir"], PROJECT_ROOT))
    tools = {name: bool(shutil.which(name)) for name in ("gst-launch-1.0", "tc")}
    network = config.get("network", {})
    if network.get("enabled"):
        raise RuntimeError("Network shaping must be launched explicitly with setup_netem.sh and always paired with reset_netem.sh")
    status = {"status": "ready" if all(tools.values()) else "dependency_missing", "tools": tools,
              "evaluation_mode": config["experiment"].get("evaluation_mode", "causal")}
    (output / "summary.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return status
