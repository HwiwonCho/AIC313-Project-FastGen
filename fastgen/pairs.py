"""Frozen self-trained teacher pairs, generated in a separate GPU process."""
import json
import math
import time
from pathlib import Path

import torch

from .data import read_manifest
from .engine import load_ema
from .metrics import peer_evaluation_precision
from .runtime import MemoryGuard, atomic_json, atomic_save, digest, sha256
from .sampling import sample_images


def generate_pairs(task):
    started = time.perf_counter()
    directory = Path(task["output"])
    directory.mkdir(parents=True, exist_ok=True)
    manifest = read_manifest(task["manifest"], task["split_dir"])
    protocol = {"version": 1, "source_sha256": sha256(task["source"]),
                "manifest_sha256": digest(manifest), "method": task["method"],
                "classes": len(manifest["mapping"]), "pairs_per_class": task["pairs_per_class"],
                "batch_size": task["batch_size"], "shard_size": task["shard_size"],
                "seed": task["seed"], "precision": task["precision"],
                "storage": "fp16", "quantize_z_before_teacher": True,
                "matmul_tf32": False, "cudnn_tf32": True,
                "synthetic_pair_augmentation": False, "code_sha256": task["code_sha256"]}
    protocol_path = directory / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Pair protocol/source changed; use a new directory")
    else:
        if any(directory.iterdir()):
            raise ValueError("Pair directory contains unrecognized files")
        atomic_json(protocol_path, protocol)
    signature = digest(protocol)
    if (directory / "index.json").exists():
        index = json.loads((directory / "index.json").read_text())
        if index["protocol"] != protocol or any(sha256(directory/r["file"]) != r["sha256"] for r in index["shards"]):
            raise ValueError("Existing completed pair pool is inconsistent")
        return index
    device = torch.device(task["device"])
    guard = MemoryGuard(device, task["vram_limit_bytes"])
    model, saved = load_ema(task["source"], device)
    if saved["identity"]["manifest_sha256"] != protocol["manifest_sha256"]:
        raise ValueError("Teacher and pair manifest differ")
    del saved
    count = protocol["classes"] * protocol["pairs_per_class"]
    shards = []
    for shard in range(math.ceil(count / task["shard_size"])):
        name = f"shard_{shard:05d}.pt"
        path = directory / name
        size = min(task["shard_size"], count-shard*task["shard_size"])
        labels = torch.arange(shard*task["shard_size"], shard*task["shard_size"]+size) % protocol["classes"]
        if path.exists():
            stored = torch.load(path, map_location="cpu", weights_only=True)
            if (stored["protocol_sha256"] != signature or not torch.equal(stored["category"], labels)
                or tuple(stored["z"].shape) != (size, 3, 64, 64)
                or stored["x"].shape != stored["z"].shape
                or stored["z"].dtype != torch.float16 or stored["x"].dtype != torch.float16
                or not bool(torch.isfinite(stored["z"]).all()) or not bool(torch.isfinite(stored["x"]).all())):
                raise ValueError("Incomplete/corrupted pair shard")
        else:
            cpu_rng = torch.Generator().manual_seed(task["seed"] + shard * 2)
            extra = torch.Generator(device=device).manual_seed(task["seed"] + shard * 2 + 1)
            # Quantization happens BEFORE teacher inference, preserving the pair.
            z = torch.randn(size, 3, 64, 64, generator=cpu_rng).half()
            targets = []
            for start in range(0, size, task["batch_size"]):
                cats = labels[start:start+task["batch_size"]].to(device)
                noise = z[start:start+task["batch_size"]].to(device=device, dtype=torch.float32)
                with peer_evaluation_precision():
                    target = sample_images(model, noise.shape, cats, device=device,
                               method=task["method"], noise=noise, extra_generator=extra,
                               precision=task["precision"])
                targets.append(target.cpu().half())
                guard.check()
            stored = {"z": z, "x": torch.cat(targets), "category": labels,
                      "protocol_sha256": signature}
            atomic_save(path, stored)
        shards.append({"file": name, "count": size, "sha256": sha256(path)})
        print(json.dumps({"pair_shard": shard+1, "total_shards": math.ceil(count/task["shard_size"]),
                          "elapsed_seconds": time.perf_counter()-started, **guard.check()}), flush=True)
        del stored
    index = {"protocol": protocol, "count": count, "shard_size": task["shard_size"], "shards": shards}
    atomic_json(directory / "index.json", index)
    atomic_json(directory / "summary.json", {"wall_seconds": time.perf_counter()-started,
                "memory": guard.check(), "source_sha256": protocol["source_sha256"],
                "official_validation_accessed": False})
    return index
