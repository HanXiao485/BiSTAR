import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from .hashing import sha256_file


def _command(args):
    result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip() if result.returncode == 0 else None


def git_state(root: Path) -> Dict[str, Any]:
    commit = _command(["git", "-C", str(root), "rev-parse", "HEAD"])
    status = _command(["git", "-C", str(root), "status", "--porcelain"])
    vm = _command(["git", "-C", str(root / "third_party/VideoMAE"), "rev-parse", "HEAD"])
    return {"project_commit": commit, "project_dirty": bool(status) if status is not None else None, "videomae_commit": vm}


def environment_info(root: Path) -> Dict[str, Any]:
    version = subprocess.run(["ffmpeg", "-hide_banner", "-version"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    buildconf = subprocess.run(["ffmpeg", "-hide_banner", "-buildconf"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True).stdout
    env_dir = root / "outputs/environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    conf_path = env_dir / "ffmpeg_buildconf.txt"
    conf_path.write_text(buildconf, encoding="utf-8")
    try:
        import torch
        torch_version, cuda, gpu = torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        torch_version, cuda, gpu = None, None, None
    return {"python": sys.version, "platform": platform.platform(), "torch": torch_version, "cuda": cuda, "gpu": gpu,
            "ffmpeg_version": version.splitlines()[0], "ffmpeg_buildconf_sha256": sha256_file(conf_path),
            "libx264_available": "libx264" in encoders, "libx265_available": "libx265" in encoders}


def write_run_metadata(root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "git_state.json").write_text(json.dumps(git_state(root), indent=2), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(environment_info(root), indent=2), encoding="utf-8")
    for name in ("stdout.log", "stderr.log"):
        (output_dir / name).touch()
