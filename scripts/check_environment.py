#!/usr/bin/env python3
import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
REQUIRED = ["numpy", "yaml", "cv2", "pandas", "matplotlib", "skimage", "pydantic", "pytest", "torch", "torchvision", "timm", "lpips", "decord"]


def main():
    checks = {"python_3_10_or_newer": sys.version_info >= (3, 10), "ffmpeg": bool(shutil.which("ffmpeg")), "ffprobe": bool(shutil.which("ffprobe"))}
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout if checks["ffmpeg"] else ""
    checks.update({"libx264": "libx264" in encoders, "libx265": "libx265" in encoders})
    for module in REQUIRED:
        try: importlib.import_module(module); checks["python:" + module] = True
        except Exception: checks["python:" + module] = False
    checks["videomae_source"] = (ROOT / "third_party/VideoMAE/modeling_pretrain.py").is_file()
    checks["videomae_checkpoint"] = (ROOT / "checkpoints/videomae_base.pth").is_file()
    if checks["videomae_checkpoint"]:
        from src.common.config import load_yaml
        from src.common.hashing import sha256_file
        expected = load_yaml(ROOT / "configs/models/videomae.yaml")["model"]["checkpoint"]["sha256"]
        checks["videomae_checkpoint_hash"] = sha256_file(ROOT / "checkpoints/videomae_base.pth") == expected
    try:
        from src.bistar.reconstruction.videomae_adapter import VideoMAEAdapter
        from src.common.config import load_yaml
        VideoMAEAdapter(load_yaml(ROOT / "configs/models/videomae.yaml")).load_model()
        checks["videomae_model_load"] = True
    except Exception as error:
        checks["videomae_model_load"] = False
        checks["videomae_model_error"] = str(error)
    required_ok = all(value for key, value in checks.items() if key != "videomae_model_error")
    report = {"ok": required_ok, "checks": checks}
    output = ROOT / "outputs/environment/check_environment.json"; output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    from src.common.logging import environment_info
    (output.parent / "environment.json").write_text(json.dumps(environment_info(ROOT), indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if required_ok else 1


if __name__ == "__main__": raise SystemExit(main())
