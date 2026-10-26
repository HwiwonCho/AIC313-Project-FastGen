"""Audited execution-only continuation; never warm-start from EMA or rewind to best.

All numerical checkpoint fields are preserved byte for byte. Only the source
inventory and execution metadata may change, and a receipt proves full-state
equality before the training worker opens the new checkpoint.
"""
import copy
import ctypes
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch

from .project import read_json, run_lock
from .runtime import append_event, atomic_json, atomic_save, digest, sha256


def fingerprint(value):
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            t = item.detach().cpu().contiguous()
            h.update(repr((str(t.dtype), list(t.shape))).encode())
            h.update(ctypes.string_at(t.data_ptr(), t.numel()*t.element_size()))
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=lambda k: (type(k).__name__, repr(k))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            h.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        else:
            h.update(repr((type(item).__name__, item)).encode())
    visit(value)
    return h.hexdigest()


def learning_identity(identity):
    result = copy.deepcopy(identity)
    result.pop("code_sha256", None)
    result.pop("execution", None)
    return result


def migrate_checkpoint(source, task, destination, receipt_path):
    from .data import read_manifest
    from .engine import stage_identity
    source, destination = Path(source), Path(destination)
    old_hash = sha256(source)
    saved = torch.load(source, map_location="cpu", weights_only=False)
    required = {"model", "ema", "optimizer", "rng", "batch_state", "step", "curve", "active_seconds"}
    if saved.get("format") != "fastgen_e2e_v1" or not required <= saved.keys():
        raise ValueError("Continuation requires this pipeline's complete training checkpoint")
    if saved["signature"] != digest(saved["identity"]):
        raise ValueError("Source checkpoint identity/signature disagree")
    new_identity = stage_identity(task, read_manifest(task["manifest"], task["split_dir"]))
    if learning_identity(saved["identity"]) != learning_identity(new_identity):
        raise ValueError("Continuation would change the learning/data/FID recipe")
    state = {key: value for key, value in saved.items() if key not in ("identity", "signature")}
    before = {key: fingerprint(value) for key, value in state.items()}
    old_identity = saved["identity"]
    saved["identity"], saved["signature"] = new_identity, digest(new_identity)
    if destination.exists():
        existing = torch.load(destination, map_location="cpu", weights_only=False)
        if existing["signature"] != saved["signature"] or any(
                fingerprint(existing[key]) != expected for key, expected in before.items()):
            raise ValueError("Existing continuation checkpoint differs from the source full state")
        del existing
    else:
        atomic_save(destination, saved)
    restored = torch.load(destination, map_location="cpu", weights_only=False)
    after = {key: fingerprint(restored[key]) for key in state}
    if before != after or sha256(source) != old_hash:
        raise ValueError("Full-state preservation verification failed")
    receipt = {"source": str(source), "source_sha256": old_hash,
               "output_sha256_at_adoption": sha256(destination), "step": saved["step"],
               "allowed_identity_changes": ["code_sha256", "execution"],
               "previous_identity": old_identity, "identity": new_identity,
               "full_state_fingerprints_before": before, "full_state_fingerprints_after": after,
               "full_state_equal": True, "learning_updates_during_adoption": 0}
    atomic_json(destination.parent/"identity.json", new_identity)
    atomic_json(receipt_path, receipt)
    return receipt


def _checkpoint_link_or_copy(source, target):
    if Path(source).suffix == ".pt":
        try:
            os.link(source, target)  # Only completed, immutable pilot checkpoints.
            return target
        except OSError:
            pass
    return shutil.copy2(source, target)


def prepare_continuation(source, target, identity):
    """Adopt completed pilots and history once; source worker must be stopped."""
    source, target = Path(source).resolve(strict=True), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Continuation needs a separate sibling output directory")
    marker = target/"continuation.json"
    if marker.exists():
        record = read_json(marker)
        if record["source_run"] != str(source):
            raise ValueError("Continuation source changed")
        return record
    with run_lock(source):
        old = read_json(source/"pipeline_identity.json")
        old_cfg, new_cfg = copy.deepcopy(old["config"]), copy.deepcopy(identity["config"])
        old_cfg.pop("execution", None)
        new_cfg.pop("execution", None)
        if old_cfg != new_cfg or any(old[k] != identity[k] for k in ("data_root", "split_dir")):
            raise ValueError("Only execution settings may change during continuation")
        summary = read_json(source/"teacher/summary.json")
        if summary["phase"] != "stopped" or not 0 < summary["step"] < summary["target_steps"]:
            raise ValueError("Source teacher must have cooperatively stopped before completion")
        if any((source/name).exists() for name in ("pairs_teacher", "reflow", "distill")):
            raise ValueError("This continuation is restricted to the unfinished teacher stage")
        for name in ("pilot_flip", "pilot_spatial"):
            if old_cfg["pilot"]["enabled"]:
                if read_json(source/name/"summary.json")["phase"] != "complete":
                    raise ValueError("Both augmentation pilots must already be complete")
                shutil.copytree(source/name, target/name, copy_function=_checkpoint_link_or_copy, dirs_exist_ok=True)
        history = target/"adopted_history"/"previous_run"
        history.mkdir(parents=True, exist_ok=True)
        for name in ("source_snapshot", "adopted_history"):
            if (source/name).exists():
                shutil.copytree(source/name, history/name, dirs_exist_ok=True)
        for name in ("pipeline_identity.json", "status.json", "preflight.json", "augmentation_selection.json", "execution.jsonl"):
            if (source/name).exists():
                shutil.copy2(source/name, history/name)
        teacher = target/"teacher"
        shutil.copytree(source/"teacher", teacher, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("*.pt", "*.tmp", "identity.json"))
        # Accumulated costs include the old execution, not just this session.
        old_execution = (source/"execution.jsonl").read_text()
        new_execution = (target/"execution.jsonl").read_text() if (target/"execution.jsonl").exists() else ""
        (target/"execution.jsonl").write_text(old_execution+new_execution)
        record = {"source_run": str(source), "source_checkpoint_sha256": sha256(source/"teacher/current.pt"),
                  "step": summary["step"], "previous_code": old["code"], "new_code": identity["code"],
                  "changed_source_files": sorted(name for name in set(old["code"]) | set(identity["code"])
                                                 if old["code"].get(name) != identity["code"].get(name)),
                  "learning_configuration_equal": True, "pilot_training_repeated": False}
        atomic_json(marker, record)
        atomic_json(target/"adopted_history/continuation.json", record)
        append_event(target/"execution.jsonl", event="full_state_continuation_prepared", step=record["step"],
                     source_run=str(source))
        return record


def continue_teacher(source, task, target):
    receipt = target/"adopted_history/teacher_full_state_continuation.json"
    with run_lock(source):
        record = read_json(target/"continuation.json")
        if sha256(Path(source)/"teacher/current.pt") != record["source_checkpoint_sha256"]:
            raise ValueError("Stopped source checkpoint changed since preparation")
        if not receipt.exists():
            migrate_checkpoint(Path(source)/"teacher/current.pt", task, target/"teacher/current.pt", receipt)
        if (Path(source)/"teacher/best.pt").exists() and not (target/"adopted_history/teacher_best_continuation.json").exists():
            migrate_checkpoint(Path(source)/"teacher/best.pt", task, target/"teacher/best.pt",
                               target/"adopted_history/teacher_best_continuation.json")
