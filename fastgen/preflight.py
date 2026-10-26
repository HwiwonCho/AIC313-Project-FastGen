"""Fail early on server dependencies/GPU/FID assets, before any long training."""
import time
from pathlib import Path

import torch

from .data import read_manifest, safe_image
from .engine import build_model
from .metrics import peer_evaluation_precision
from .runtime import MemoryGuard, atomic_json, environment_record


def preflight(task):
    started = time.perf_counter()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
    if task["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("This server does not support the configured BF16 training")
    device = torch.device(task["device"])
    guard = MemoryGuard(device, task["vram_limit_bytes"])
    manifest = read_manifest(task["manifest"], task["split_dir"])
    for row in manifest["train_rows"]:
        safe_image(task["data_root"], row["path"])
    with torch.device("meta"):
        model = build_model(task["architecture"])
    count = sum(p.numel() for p in model.parameters())
    if count != 95_524_611:
        raise RuntimeError("Unexpected full model parameter count")
    del model
    # Only the provided evaluation Inception is downloaded/loaded if missing.
    # Doing this now avoids discovering an unavailable asset after 3000 updates.
    from cleanfid.fid import build_feature_extractor
    with torch.no_grad(), peer_evaluation_precision():
        feature = build_feature_extractor("clean", device=device, use_dataparallel=False)
        features = feature(torch.zeros(1, 3, 299, 299, device=device))
        if not bool(torch.isfinite(features).all()):
            raise RuntimeError("Provided FID Inception failed its startup check")
    result = {"phase": "passed", "parameters_each_model": count,
              "train_files": manifest["train_files"], "official_validation_accessed": False,
              "environment": environment_record(device), "memory": guard.check(),
              "wall_seconds": time.perf_counter()-started,
              "scope": "Dependencies, one GPU, BF16 and FID asset; actual training peak is measured by each trainer."}
    atomic_json(Path(task["run_dir"])/"preflight.json", result)
    return result
