"""Source integrity, frozen configuration and repeatable run locking."""
import ast
import contextlib
import os
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def check_official():
    from fastgen.runtime import sha256
    record = read_json(ROOT / "official_files.json")
    for name in record.get("initial_starter_paths", []):
        if not (ROOT / name).is_file():
            raise ValueError(f"Initial starter path is missing: {name}")
    for name, expected in record["sha256"].items():
        if sha256(ROOT / name) != expected:
            raise ValueError(f"Official immutable file changed: {name}")
    model_source = (ROOT / "model.py").read_text()
    tree = ast.parse(model_source)
    base = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    import hashlib
    if hashlib.sha256(ast.get_source_segment(model_source, base).encode()).hexdigest() != record["base_Model_source_sha256"]:
        raise ValueError("The official base Model class was changed")


def code_inventory():
    from fastgen.runtime import sha256
    result = {}
    for pattern in ("*.py", "*.sh", "fastgen/*.py", "networks/*.py", "src/*.py", "tests/*.py", "configs/*.json",
                    "tools/**/*.py", "requirements.txt", "README.md", "TRAINING.md", "CITATIONS.md",
                    ".orca/.gitignore", "assets/*", "official_files.json", "network_provenance.json"):
        for path in ROOT.glob(pattern):
            result[path.relative_to(ROOT).as_posix()] = sha256(path)
    return result


@contextlib.contextmanager
def run_lock(directory):
    lock_path = directory / ".pipeline.lock"
    with lock_path.open("a+b") as f:
        f.seek(0)
        if os.name == "nt":
            import msvcrt
            if f.read(1) == b"":
                f.write(b"0")
                f.flush()
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise RuntimeError("Another pipeline owns this output directory")
        else:
            import fcntl
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError("Another pipeline owns this output directory")
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def validate_config(cfg):
    from fastgen.engine import DEFAULT_ARCHITECTURE
    if cfg["architecture"] != DEFAULT_ARCHITECTURE:
        raise ValueError("The submission pipeline fixes the verified 95.52M architecture")
    if cfg["vram_limit_bytes"] > 20_000_000_000 or cfg["vram_limit_bytes"] <= 0:
        raise ValueError("VRAM limit must be positive and at most 20,000,000,000 bytes")
    if cfg["microbatch"] not in (4, 8, 16) or cfg["accumulation"] < 1 or cfg["microbatch"] * cfg["accumulation"] != 16:
        raise ValueError("Use microbatch 4/8/16 with effective batch 16")
    pilot_micro = cfg["pilot"].get("microbatch", cfg["microbatch"])
    pilot_accum = cfg["pilot"].get("accumulation", cfg["accumulation"])
    if pilot_micro not in (4, 8, 16) or pilot_accum < 1 or pilot_micro * pilot_accum != 16:
        raise ValueError("Pilot execution must also have effective batch16")
    if not 1 <= cfg["pilot"]["steps_each"] <= 3000:
        raise ValueError("Deadline preset caps the two-condition pilot at 3000 updates each")
    if cfg["fixed_augmentation_if_pilot_disabled"] not in ("flip", "spatial"):
        raise ValueError("Invalid fixed augmentation")
    for name in ("teacher", "reflow", "distill"):
        stage = cfg[name]
        if stage["steps"] < 1 or stage["horizon"] < 1 or stage["lr"] <= 0 or stage["warmup"] < 0 or stage["eval_every"] < 1:
            raise ValueError(f"Invalid {name} settings")
    if cfg["save_every"] < 1 or cfg["log_every"] < 1 or cfg["eval_batch_size"] < 1:
        raise ValueError("Positive logging, saving and evaluation intervals required")
    if cfg["evaluation"] != {"protocol": "yunseo_scaled_a80bbd0_train755_defaults_v1", "seed": 27182, "precision": "fp32", "workers": 4} or cfg["eval_batch_size"] != 8 or cfg["holdout_per_class"] != 5:
        raise ValueError("Peer FID protocol is fixed; do not silently change seed/batch/precision/reference size")
    execution = cfg.get("execution", {"soflow_prefetch_batches": 0})
    if set(execution) != {"soflow_prefetch_batches"} or execution["soflow_prefetch_batches"] not in (0, 2):
        raise ValueError("Use disabled or bounded two-batch SoFlow prefetch")
    if cfg["precision"] not in ("fp32", "bf16"):
        raise ValueError("Invalid training precision")
    if cfg["pairs"]["precision"] != "fp32" or cfg["pairs"]["method"] != "solution4":
        raise ValueError("Default distillation needs a deterministic FP32 four-call teacher")
    if any(cfg["pairs"][k] < 1 for k in ("pairs_per_class", "shard_size", "batch_size")):
        raise ValueError("Pair counts and sizes must be positive")


