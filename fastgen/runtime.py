"""Portable records, deterministic state and single-device resource accounting."""
import contextlib
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def append_event(path, **value):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps({"unix": time.time(), **value}, allow_nan=False) + "\n")


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["cpu"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


@contextlib.contextmanager
def preserved_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


class MemoryGuard:
    """Check the lifetime peak; never reset it midway through a training stage."""
    def __init__(self, device, limit=20_000_000_000, margin=0.90):
        self.device, self.limit = torch.device(device), limit
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.set_device(self.device)
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(total, limit) * margin / total, self.device)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
        self.started = time.perf_counter()

    def snapshot(self):
        if self.device.type != "cuda":
            return {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "cuda_measured": False}
        allocated = torch.cuda.max_memory_allocated(self.device)
        reserved = torch.cuda.max_memory_reserved(self.device)
        return {"peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved, "cuda_measured": True}

    def check(self):
        result = self.snapshot()
        if result["peak_allocated_bytes"] >= self.limit:
            raise RuntimeError(f"Training VRAM limit reached: {result['peak_allocated_bytes']} >= {self.limit}")
        return result


def environment_record(device):
    import torchvision
    from importlib.metadata import version, PackageNotFoundError
    packages = {}
    for name in ("torch", "torchvision", "Pillow", "clean-fid", "numpy", "scipy", "tqdm"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "torchvision": torchvision.__version__,
              "cuda_runtime": torch.version.cuda, "device": str(device), "packages": packages,
              "determinism": "Same-device seeded RNG and complete checkpoint restore; cross-device bitwise equality is not promised."}
    if torch.device(device).type == "cuda":
        p = torch.cuda.get_device_properties(device)
        result.update(gpu=p.name, total_gpu_memory=p.total_memory)
        try:
            result["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            result["driver"] = None
    return result


def plot_curve(path, rows):
    canvas = Image.new("RGB", (1000, 400), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((15, 10), "Training loss (logged updates); FID is recorded separately", fill="black")
    points = [(r["step"], r["loss"]) for r in rows if "loss" in r]
    if points:
        lo, hi = min(y for _, y in points), max(y for _, y in points)
        xs = max(1, points[-1][0] - points[0][0])
        pixels = [(40 + 920 * (x-points[0][0])/xs, 360-320*(y-lo)/max(hi-lo, 1e-9)) for x,y in points]
        if len(pixels) > 1:
            draw.line(pixels, fill="navy", width=2)
        draw.text((45, 375), f"step {points[0][0]} to {points[-1][0]}; loss range {lo:.5g} to {hi:.5g}", fill="black")
    canvas.save(path)
