"""Scratch SoFlow and EMA-warm-started distillation, without legacy launchers."""
import copy
import json
import math
import shutil
import signal
import time
from pathlib import Path

import torch

from networks.soflow_model import SoFlow64, update_ema
from .data import PairBatches, TrainBatches, read_manifest
from .metrics import preview, train_proxy
from .runtime import (MemoryGuard, append_event, atomic_json, atomic_save, digest,
                      environment_record, plot_curve, restore_rng, rng_state, seed_all, sha256)
from .sampling import precision_context

DEFAULT_ARCHITECTURE = {"num_categories": 151, "base_channels": 128,
                        "channel_mult": [1, 2, 2, 4], "num_blocks": 3}
SOFLOW_RECIPE = {"fm_ratio": .75, "p": 1., "eps": .01, "cfg_scale": 3.,
                 "cfg_decay_time": .8, "mix_ratio": .25, "null_probability": .1,
                 "TC_end_ratio": .002}


def build_model(architecture=None):
    model = SoFlow64(**(architecture or DEFAULT_ARCHITECTURE))
    count = sum(p.numel() for p in model.parameters())
    if count > 100_000_000:
        raise ValueError(f"Model exceeds 100M parameters: {count}")
    return model


def learning_rate(settings, completed):
    horizon, warmup = settings["horizon"], settings["warmup"]
    if completed < warmup:
        multiplier = (completed + 1) / max(1, warmup)
    else:
        p = min(1., max(0., (completed - warmup) / max(1, horizon - warmup)))
        multiplier = .1 + .9 * .5 * (1 + math.cos(math.pi * p))
    return settings["lr"] * multiplier


def soflow_objective(model, images, labels, step, settings):
    size, device = len(images), images.device
    fm = size * 3 // 4
    if size % 4 or fm == 0:
        raise ValueError("SoFlow microbatch must be a positive multiple of four")
    tc = size - fm
    cfg = torch.rand(size, device=device) < SOFLOW_RECIPE["null_probability"]
    t_fm = torch.sigmoid(torch.randn(fm, device=device) - .2)
    t_tc = torch.sigmoid(torch.randn(tc, device=device) * .8 + .2).clamp(1e-3, 1-1e-3)
    proposal = torch.sigmoid(torch.randn(tc, device=device) * .8 - 1)
    destination = torch.minimum(proposal, t_tc - 1e-4).clamp_min(0)
    middle = model.middle_time(t_tc, destination, step, settings["horizon"], end_ratio=SOFLOW_RECIPE["TC_end_ratio"])
    middle = torch.maximum(middle, destination)
    options = {key: SOFLOW_RECIPE[key] for key in ("p", "eps", "cfg_scale", "cfg_decay_time", "mix_ratio")}
    loss_fm, mse_fm = model.velocity_loss(images[:fm], t_fm, labels[:fm], cfg[:fm], **options)
    loss_tc, mse_tc = model.consistency_loss(images[fm:], t_tc, middle, destination,
                                           labels[fm:], cfg[fm:], **options)
    return .75 * loss_fm + .25 * loss_tc, {"fm_mse": float(mse_fm.detach()),
                                         "tc_mse": float(mse_tc.detach())}


def paired_objective(model, z, target, labels, kind):
    if kind == "distill":
        t = torch.ones(len(z), device=z.device, dtype=torch.float32)
        prediction = model.solution(z, t, torch.zeros_like(t), labels)
        loss = (prediction.float() - target).square().mean()
    elif kind == "reflow":
        t = torch.rand(len(z), device=z.device)
        xt = (1 - t[:, None, None, None]) * target + t[:, None, None, None] * z
        loss = (model.predict_velocity(xt, t, labels).float() - (z - target)).square().mean()
    else:
        raise ValueError("Unknown pair objective")
    return loss, {"endpoint_mse" if kind == "distill" else "reflow_mse": float(loss.detach())}


def load_ema(path, device="cpu"):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(saved["architecture"])
    model.load_state_dict(saved["ema"], strict=True)
    return model.to(device).eval().requires_grad_(False), saved


def stage_identity(task, manifest):
    settings = task["settings"]
    source_hash = sha256(task["source"]) if task.get("source") else None
    pool_hash = sha256(Path(task["pairs"]) / "index.json") if task.get("pairs") else None
    identity = {"settings": settings, "architecture": task["architecture"],
                "learning_recipe": {"soflow": SOFLOW_RECIPE if settings["kind"]=="soflow" else None,
                                    "optimizer": "AdamW", "betas": [.9,.99], "weight_decay": 0,
                                    "gradient_clip": 10, "ema_target": .9999,
                                    "pair_sampling": "size_weighted_shard_per_microbatch_then_uniform_rows" if task.get("pairs") else None},
                "training_precision": task["precision"],
                "evaluation": {"batch_size": task["eval_batch_size"], **task["evaluation"]},
                "manifest_sha256": digest(manifest), "source_sha256": source_hash,
                "pairs_index_sha256": pool_hash, "code_sha256": task["code_sha256"]}
    identity["execution"] = task.get("execution", {"soflow_prefetch_batches": 0})
    return identity


def train_stage(task, *, proxy_function=None):
    """One child process owns one GPU stage and its current/best checkpoints."""
    run = Path(task["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    settings = task["settings"]
    if settings["steps"] <= 0 or settings["microbatch"] <= 0 or settings["accumulation"] <= 0:
        raise ValueError("Positive update and batch counts required")
    if settings["kind"] == "soflow" and settings["microbatch"] % 4:
        raise ValueError("SoFlow microbatch must be divisible by 4")
    manifest = read_manifest(task["manifest"], task["split_dir"])
    identity = stage_identity(task, manifest)
    source_hash = identity["source_sha256"]
    pool_hash = identity["pairs_index_sha256"]
    signature = digest(identity)
    if (run / "identity.json").exists():
        if json.loads((run / "identity.json").read_text()) != identity:
            raise ValueError("Stage configuration/source/data/code changed; use a fresh run directory")
    else:
        atomic_json(run / "identity.json", identity)
    seed_all(settings["seed"])
    device = torch.device(task["device"])
    guard = MemoryGuard(device, task["vram_limit_bytes"])
    environment = environment_record(device)
    atomic_json(run / "environment.json", environment)
    atomic_json(run / "sessions" / (str(time.time_ns())+"_environment.json"), environment)
    model = build_model(task["architecture"]).to(device)
    if task.get("source"):
        source = torch.load(task["source"], map_location="cpu", weights_only=False)
        if source["identity"]["manifest_sha256"] != digest(manifest):
            raise ValueError("Warm-start manifest differs")
        model.load_state_dict(source["ema"], strict=True)
        del source
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=(.9, .99), weight_decay=0)
    if settings["kind"] == "soflow":
        batches = TrainBatches(task["data_root"], manifest, pilot=settings["pilot"],
                               augmentation=settings["augmentation"], seed=settings["seed"],
                               unique=settings["unique_sampling"])
    else:
        batches = PairBatches(task["pairs"], settings["seed"])
        protocol = batches.index["protocol"]
        if protocol["source_sha256"] != source_hash or protocol["manifest_sha256"] != digest(manifest):
            raise ValueError("Pair pool does not belong to the exact warm-start source")
    if settings["kind"] == "soflow" and task.get("execution", {}).get("soflow_prefetch_batches", 0):
        from .prefetch import PrefetchBatches
        batches = PrefetchBatches(batches, settings["microbatch"], capacity=2)
    step, best, active_seconds = 0, None, 0.
    current, best_path = run / "current.pt", run / "best.pt"
    curve = []
    if current.exists():
        saved = torch.load(current, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Checkpoint signature differs")
        model.load_state_dict(saved["model"], strict=True)
        ema.load_state_dict(saved["ema"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        batches.load_state_dict(saved["batch_state"])
        restore_rng(saved["rng"])
        step, best, active_seconds = saved["step"], saved["best_fid"], saved["active_seconds"]
        curve = saved["curve"]
        if best is not None:
            header = torch.load(best_path, map_location="meta", weights_only=False) if best_path.exists() else None
            if header is None or header["best_fid"] != best:
                # Recover a crash between the atomic current save and best copy.
                # Only the actual newly scored current is eligible for repair.
                if (saved.get("metrics") or {}).get("FID_train_proxy") != best:
                    raise ValueError("Current/best scores disagree without a completed scored current")
                tmp = best_path.with_suffix(".tmp")
                shutil.copyfile(current, tmp)
                tmp.replace(best_path)
                append_event(run/"events.jsonl", event="recovered_interrupted_best_adoption", step=step, score=best)
            elif header["signature"] != signature:
                raise ValueError("Best checkpoint belongs to another stage")
            del header
        del saved
    stopped = False
    previous_handlers = {}
    def stop_handler(_signum, _frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM) + ((signal.SIGBREAK,) if hasattr(signal,"SIGBREAK") else ()):
        previous_handlers[sig] = signal.signal(sig, stop_handler)
    session_start = time.perf_counter()
    initial_active = active_seconds
    memory_previous = {}
    if (run / "summary.json").exists():
        memory_previous = json.loads((run / "summary.json").read_text()).get("memory", {})
    append_event(run / "events.jsonl", event="session_start", step=step, signature=signature,
                 source_sha256=source_hash, initialization="checkpoint_resume" if current.exists() else
                 ("scratch" if source_hash is None else "own_scratch_lineage_EMA"),
                 reference_overlaps_training=not settings["pilot"])

    def save_checkpoint(metrics=None):
        nonlocal active_seconds
        if shutil.disk_usage(run).free < task["minimum_free_bytes"]:
            raise RuntimeError("Insufficient free disk space for atomic checkpoint preservation")
        active_seconds = initial_active + time.perf_counter() - session_start
        atomic_save(current, {"format": "fastgen_e2e_v1", "signature": signature,
                    "identity": identity, "architecture": task["architecture"], "step": step,
                    "model": model.state_dict(), "ema": ema.state_dict(),
                    "optimizer": optimizer.state_dict(), "rng": rng_state(),
                    "batch_state": batches.state_dict(), "best_fid": best, "curve": curve,
                    "active_seconds": active_seconds, "metrics": metrics})

    def write_summary(phase):
        memory = guard.snapshot() if phase == "failed" else guard.check()
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            memory[key] = max(memory[key], memory_previous.get(key, 0))
        value = {"phase": phase, "step": step, "target_steps": settings["steps"],
                 "best_train_proxy_fid": best, "active_wall_seconds": initial_active + time.perf_counter()-session_start,
                 "memory": memory, "parameters_each_model": sum(p.numel() for p in model.parameters()),
                 "source_sha256": source_hash, "signature": signature,
                 "reference_overlaps_training": not settings["pilot"], "official_validation_accessed": False}
        atomic_json(run / "summary.json", value)
        plot_curve(run / "loss_curve.png", curve)
        return value

    try:
        while step < settings["steps"] and not stopped:
            began = time.perf_counter()
            guard.check()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(settings, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            rows = []
            for _ in range(settings["accumulation"]):
                with precision_context(device, task["precision"]):
                    batch = batches.batch(settings["microbatch"], device)
                    if settings["kind"] == "soflow":
                        loss, row = soflow_objective(model, *batch, step, settings)
                    else:
                        loss, row = paired_objective(model, *batch, settings["kind"])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("Nonfinite training objective")
                (loss / settings["accumulation"]).backward()
                rows.append({"loss": float(loss.detach()), **row})
                guard.check()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10, error_if_nonfinite=True)
            optimizer.step()
            update_ema(ema, model, min(.9999, (1+step)/(10+step)))
            step += 1
            row = {"step": step, "lr": lr, "gradient_norm": float(norm),
                   "update_seconds": time.perf_counter()-began,
                   **{k: sum(r[k] for r in rows)/len(rows) for k in rows[0]}}
            if step == 1 or step % settings["log_every"] == 0 or step == settings["steps"]:
                curve.append(row)
                append_event(run / "metrics.jsonl", **row)
                print(json.dumps({"stage": run.name, **row, **guard.check()}), flush=True)
                write_summary("training")
            if step in (1, 1000) or step % settings["eval_every"] == 0 or step == settings["steps"]:
                preview(ema, run / "samples" / f"step_{step:07d}", device=device,
                        method=settings["sampling_method"], precision=task["precision"])
            if step % settings["eval_every"] == 0 or step == settings["steps"]:
                evaluation_start = time.perf_counter()
                # Each attempt gets a fresh directory; failed attempts are retained.
                evaluation = run / "evaluations" / f"step_{step:07d}_{time.time_ns()}"
                metrics = (proxy_function or train_proxy)(ema, manifest, task["data_root"], evaluation,
                          device=device, method=settings["sampling_method"],
                          batch_size=task["eval_batch_size"], precision=task["evaluation"]["precision"],
                          seed=task["evaluation"]["seed"], workers=task["evaluation"]["workers"],
                          reference_overlaps_training=not settings["pilot"])
                score = metrics["FID_train_proxy"]
                if not math.isfinite(score):
                    raise FloatingPointError("Nonfinite training proxy FID")
                improved = best is None or score < best
                if improved:
                    best = score
                save_checkpoint(metrics)
                if improved:
                    tmp = best_path.with_suffix(".tmp")
                    shutil.copyfile(current, tmp)
                    tmp.replace(best_path)
                append_event(run / "events.jsonl", event="train_proxy", step=step, score=score,
                             strict_best=improved, wall_seconds=time.perf_counter()-evaluation_start,
                             reference_overlaps_training=not settings["pilot"])
            elif step % settings["save_every"] == 0:
                save_checkpoint()
        if stopped:
            save_checkpoint()
            append_event(run / "events.jsonl", event="cooperative_stop", step=step)
        return write_summary("stopped" if stopped else "complete")
    except BaseException as error:
        append_event(run / "events.jsonl", event="failed", step=step, error=repr(error), memory=guard.snapshot())
        write_summary("failed")
        raise
    finally:
        if hasattr(batches, "close"):
            batches.close()
        append_event(run / "events.jsonl", event="session_end", step=step,
                     session_wall_seconds=time.perf_counter()-session_start)
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
