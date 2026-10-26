"""A finite, sequential pilot -> scratch teacher -> distilled student run."""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


from .project import check_official, code_inventory, read_json, run_lock, validate_config
from .submission import bundle_submission, evaluate_submission, write_facts


def child(action, task, directory):
    from fastgen.runtime import append_event, atomic_json
    path = directory / "tasks" / (task["name"] + ".json")
    if path.exists() and read_json(path) != task:
        raise ValueError("Existing task changed")
    atomic_json(path, task)
    started = time.perf_counter()
    append_event(directory / "execution.jsonl", event="child_start", name=task["name"], action=action)
    atomic_json(directory/"status.json", {"phase": "running", "stage": task["name"], "unix": time.time()})
    process = subprocess.Popen([sys.executable, "-B", str(ROOT / "pipeline.py"), action, "--task", str(path)], cwd=ROOT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=="nt" else 0)
    try:
        try:
            code = process.wait()
        except KeyboardInterrupt:
            # The worker handles SIGTERM at a complete optimizer-update boundary.
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                process.terminate()
            code = process.wait()
            raise KeyboardInterrupt from None
        if code:
            raise RuntimeError(f"Stage {task['name']} stopped/failed with exit code {code}; downstream stages were not run")
    finally:
        append_event(directory / "execution.jsonl", event="child_end", name=task["name"],
                     action=action, returncode=process.returncode, process_wall_seconds=time.perf_counter()-started)
        atomic_json(directory/"status.json", {"phase": "running" if process.returncode==0 else "stopped_or_failed",
                    "stage": task["name"], "last_returncode": process.returncode, "unix": time.time()})


def execute_pipeline(args):
    from fastgen.data import prepare_manifest
    from fastgen.runtime import append_event, atomic_json, digest
    cfg = read_json(args.config)
    validate_config(cfg)
    check_official()
    directory, data_root = Path(args.run_root).resolve(), Path(args.data_root).resolve(strict=True)
    split_dir = Path(args.split_dir).resolve(strict=True)
    directory.mkdir(parents=True, exist_ok=True)
    if not str(cfg["device"]).startswith("cuda"):
        raise ValueError("The real pipeline requires CUDA; synthetic tests call individual CPU stages")
    if shutil.disk_usage(directory).free < cfg["minimum_free_bytes"]:
        raise RuntimeError("Not enough free space for this pipeline")
    with run_lock(directory):
        session_started=time.perf_counter()
        append_event(directory/"execution.jsonl", event="pipeline_session_start")
        try:
            inventory = code_inventory()
            identity = {"config": cfg, "code": inventory, "data_root": str(data_root), "split_dir": str(split_dir)}
            identity_path = directory / "pipeline_identity.json"
            if identity_path.exists() and read_json(identity_path) != identity:
                raise ValueError("Pipeline code/config/data path changed; use a fresh run root")
            atomic_json(identity_path, identity)
            for name in inventory:
                dest = directory / "source_snapshot" / name
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ROOT / name, dest)
            continuation = getattr(args, "continue_from", None)
            if continuation is not None:
                from .continuation import prepare_continuation
                prepare_continuation(continuation, directory, identity)
            elif (directory/"continuation.json").exists():
                continuation = Path(read_json(directory/"continuation.json")["source_run"])
            manifest_path = directory / "data" / "manifest.json"
            manifest = prepare_manifest(data_root, split_dir, manifest_path,
                                        holdout_per_class=cfg["holdout_per_class"], seed=cfg["seed"])
            common = {"manifest": str(manifest_path), "data_root": str(data_root), "split_dir": str(split_dir),
                      "device": cfg["device"], "precision": cfg["precision"], "architecture": cfg["architecture"],
                      "execution": cfg.get("execution", {"soflow_prefetch_batches": 0}),
                      "vram_limit_bytes": cfg["vram_limit_bytes"], "minimum_free_bytes": cfg["minimum_free_bytes"],
                      "eval_batch_size": cfg["eval_batch_size"], "evaluation": cfg["evaluation"], "code_sha256": digest(inventory)}
            child("_preflight", {**common, "name": "preflight", "run_dir": str(directory)}, directory)

            def train(name, section, *, pilot=False, augmentation="flip", source=None, pairs=None):
                settings = {**section, "seed": cfg["seed"], "pilot": pilot, "augmentation": augmentation,
                            "unique_sampling": cfg["unique_sampling"],
                            "microbatch": cfg["pilot"].get("microbatch", cfg["microbatch"]) if pilot else cfg["microbatch"],
                            "accumulation": cfg["pilot"].get("accumulation", cfg["accumulation"]) if pilot else cfg["accumulation"], "save_every": cfg["save_every"], "log_every": cfg["log_every"]}
                settings.pop("enabled", None)
                task = {**common, "name": name, "run_dir": str(directory / name), "settings": settings,
                        "source": str(source) if source else None, "pairs": str(pairs) if pairs else None}
                summary = directory / name / "summary.json"
                if name == "teacher" and continuation is not None:
                    from .continuation import continue_teacher
                    continue_teacher(continuation, task, directory)
                if not (summary.exists() and read_json(summary)["phase"] == "complete"):
                    child("_train", task, directory)
                return directory / name / "best.pt"

            augmentation = cfg["fixed_augmentation_if_pilot_disabled"]
            selection = {"pilot_enabled": cfg["pilot"]["enabled"], "augmentation": augmentation}
            if cfg["pilot"]["enabled"]:
                pilot = {"kind": "soflow", "steps": cfg["pilot"]["steps_each"], "lr": cfg["teacher"]["lr"],
                         "warmup": 1000, "horizon": cfg["teacher"]["horizon"],
                         "eval_every": cfg["pilot"]["steps_each"], "sampling_method": "renoise4"}
                for aug in ("flip", "spatial"):
                    train("pilot_"+aug, pilot, pilot=True, augmentation=aug)
                scores = {aug: read_json(directory / ("pilot_"+aug) / "summary.json")["best_train_proxy_fid"]
                          for aug in ("flip", "spatial")}
                augmentation = "spatial" if scores["spatial"] <= scores["flip"]-cfg["pilot"]["minimum_fid_gain_for_spatial"] else "flip"
                selection = {"pilot_enabled": True, "scores": scores, "augmentation": augmentation,
                             "rule": "Spatial needs the predeclared absolute FID improvement; otherwise retain flip.",
                             "not_statistical_significance": True, "not_a_long_run_quality_guarantee": True,
                             "official_validation_accessed": False}
            atomic_json(directory / "augmentation_selection.json", selection)
            teacher = train("teacher", {**cfg["teacher"], "kind": "soflow", "sampling_method": "renoise4"}, augmentation=augmentation)

            def pairs(name, source, method):
                output = directory / name
                child("_pairs", {**common, **cfg["pairs"], "name": name,
                      "output": str(output), "source": str(source), "method": method}, directory)
                return output

            pool = pairs("pairs_teacher", teacher, cfg["pairs"]["method"])
            student_source = teacher
            if cfg["reflow"]["enabled"]:
                student_source = train("reflow", {**cfg["reflow"], "kind": "reflow", "sampling_method": "euler4"},
                                       source=teacher, pairs=pool)
                pool = pairs("pairs_reflow", student_source, "euler32")
            distilled = train("distill", {**cfg["distill"], "kind": "distill", "sampling_method": "endpoint1"},
                              source=student_source, pairs=pool)
            for source, mode in [(teacher, "few_nfe"), (distilled, "one_nfe")]:
                child("_export", {"name": "export_"+mode, "source": str(source), "mode": mode,
                      "output": str(directory / "submission/checkpoints" / (mode+".ckpt"))}, directory)
            bundle_submission(directory)
            if cfg["evaluate_official_at_end"]:
                evaluate_submission(directory, data_root, split_dir, cfg)
        except BaseException as error:
            atomic_json(directory/"status.json", {"phase": "interrupted" if isinstance(error,KeyboardInterrupt) else "failed", "error": repr(error), "unix": time.time()})
            raise
        finally:
            append_event(directory/"execution.jsonl", event="pipeline_session_end", session_wall_seconds=time.perf_counter()-session_started)
            write_facts(directory)
        bundle_submission(directory)
        atomic_json(directory/"status.json", {"phase": "complete", "unix": time.time()})
