"""Launch one fresh FastGen run and save its provenance beside its outputs.

Run from anywhere, using a new path below ``runs/``. This wrapper never resumes
or replaces an existing run. Checkpoints and images remain outside Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
SCRIPTS = {
    "vae": "train.py",
    "gan": "train_gan.py",
    "soflow": "train_soflow_v2.py",
}
DATA = ROOT / "data" / "pokemon-generation-one-22k"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def source_hashes() -> dict[str, str]:
    paths = sorted(ROOT.glob("*.py")) + sorted((ROOT / "src").rglob("*.py"))
    result = {}
    for path in paths:
        value = sha256(path)
        if value:
            result[str(path.relative_to(ROOT))] = value
    return result


def package_versions() -> dict[str, str | None]:
    packages = (
        "torch", "torchvision", "Pillow", "numpy", "scipy", "clean-fid",
        "tqdm", "kagglehub",
    )
    result = {}
    for package in packages:
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = None
    return result


def gpu_info() -> dict[str, object]:
    info: dict[str, object] = {"available": False, "names": [], "torch_cuda": None}
    try:
        import torch

        info["torch_cuda"] = torch.version.cuda
        info["available"] = torch.cuda.is_available()
        if info["available"]:
            info["names"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as error:
        info["inspection_error"] = f"{type(error).__name__}: {error}"
    return info


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=sorted(SCRIPTS))
    parser.add_argument("--run-dir", type=Path, required=True, help="Unused path under runs/")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-channels", type=int, default=128, help="SoFlow only")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers must be nonnegative")
    if args.model == "soflow" and args.seed != 42:
        parser.error("train_soflow_v2.py fixes seed=42; another seed requires a script change")
    if args.model != "soflow" and args.base_channels != 128:
        parser.error("--base-channels applies only to SoFlow")
    run_dir = (ROOT / args.run_dir).resolve()
    if run_dir == RUNS.resolve() or not run_dir.is_relative_to(RUNS.resolve()):
        parser.error("--run-dir must be a new directory below runs/")
    if run_dir.exists():
        parser.error(f"run directory already exists: {run_dir}")
    args.run_dir = run_dir
    return args


def command_for(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable, "-u", str(ROOT / SCRIPTS[args.model]),
        "--run-dir", str(args.run_dir),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
    ]
    if args.model == "soflow":
        command.extend(["--num-workers", str(args.workers)])
        command.extend(["--base-channels", str(args.base_channels)])
    else:
        command.extend(["--workers", str(args.workers), "--seed", str(args.seed)])
    return command


def artifact(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def measurements(run_dir: Path, log_path: Path, model: str) -> dict:
    result: dict[str, object] = {
        "parameters": None,
        "generator_parameters": None,
        "discriminator_parameters": None,
        "peak_vram_gb": None,
        "completed_epochs": None,
    }
    if log_path.is_file():
        # Parameter counts are printed before the first training epoch.
        prefix = log_path.open("rb").read(512 * 1024).decode("utf-8", errors="replace")
        patterns = {
            "parameters": r"(?m)^parameters:\s*([\d,]+)",
            "generator_parameters": r"(?m)^generator parameters:\s*([\d,]+)",
            "discriminator_parameters": r"(?m)^discriminator parameters:\s*([\d,]+)",
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, prefix)
            if match:
                result[key] = int(match.group(1).replace(",", ""))
    loss_path = run_dir / "losses.csv"
    if loss_path.is_file():
        try:
            with loss_path.open(newline="", encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    if row.get("epoch"):
                        result["completed_epochs"] = int(row["epoch"])
                    if row.get("peak_vram_gb"):
                        value = float(row["peak_vram_gb"])
                        result["peak_vram_gb"] = max(result["peak_vram_gb"] or 0, value)
        except (OSError, ValueError) as error:
            result["parse_error"] = f"{type(error).__name__}: {error}"
    if model == "gan" and result["generator_parameters"] is not None:
        result["parameters"] = result["generator_parameters"]
    return result


def try_report(run_dir: Path) -> dict:
    if not (run_dir / "losses.csv").is_file():
        return {"attempted": False, "reason": "losses.csv unavailable"}
    report = subprocess.run(
        [sys.executable, str(ROOT / "experiment_report.py"), str(run_dir)],
        cwd=ROOT, capture_output=True, text=True, timeout=120, check=False,
    )
    return {
        "attempted": True,
        "exit_code": report.returncode,
        "output": (report.stdout + report.stderr)[-4000:],
    }


def main() -> int:
    args = parse_args()
    run_dir: Path = args.run_dir
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir.with_name(run_dir.name + ".launch.lock")
    log_path = run_dir.with_name(run_dir.name + ".train.log.in_progress")
    manifest_path = run_dir.with_name(run_dir.name + ".run_manifest.in_progress.json")
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise SystemExit(f"run is already reserved: {lock_path}") from error
    os.close(fd)
    lock_path.write_text(f"pid={os.getpid()}\nstarted_utc={utc_now()}\n", encoding="utf-8")

    command = command_for(args)
    manifest = {
        "schema_version": 1,
        "status": "preparing",
        "model": args.model,
        "run_dir": str(run_dir),
        "started_utc": utc_now(),
        "ended_utc": None,
        "wall_seconds": None,
        "exit_code": None,
        "command": command,
        "command_shell": shlex.join(command),
        "working_directory": str(ROOT),
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "seed": args.seed,
            "base_channels": args.base_channels if args.model == "soflow" else None,
        },
        "git": {
            "head": git_output("rev-parse", "HEAD"),
            "branch": git_output("branch", "--show-current"),
            "status_porcelain": git_output("status", "--porcelain=v1", "--untracked-files=normal"),
        },
        "source_sha256": source_hashes(),
        "requirements_sha256": sha256(ROOT / "requirements.txt"),
        "data_sha256": {
            "train_split.txt": sha256(DATA / "train_split.txt"),
            "category_to_id.json": sha256(DATA / "category_to_id.json"),
        },
        "environment": {
            "python": sys.version,
            "python_executable": sys.executable,
            "package_versions": package_versions(),
            "gpu": gpu_info(),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
    }
    start_clock = time.perf_counter()
    process = None
    try:
        atomic_json(manifest_path, manifest)
        print(f"Launching: {manifest['command_shell']}", flush=True)
        print(f"Live log until run directory exists: {log_path}", flush=True)
        with log_path.open("xb") as log_file:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            manifest["status"] = "running"
            manifest["pid"] = process.pid
            atomic_json(manifest_path, manifest)
            assert process.stdout is not None
            while True:
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    break
                log_file.write(chunk)
                log_file.flush()
                try:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                except (AttributeError, BrokenPipeError):
                    pass
                if run_dir.is_dir() and log_path.exists():
                    log_path.replace(run_dir / "train.log")
                    manifest_path.replace(run_dir / "run_manifest.json")
                    log_path = run_dir / "train.log"
                    manifest_path = run_dir / "run_manifest.json"
            process.wait()
    except KeyboardInterrupt:
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        manifest["interrupted"] = True
    except Exception as error:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        manifest["wrapper_error"] = f"{type(error).__name__}: {error}"
    finally:
        manifest["ended_utc"] = utc_now()
        manifest["wall_seconds"] = round(time.perf_counter() - start_clock, 3)
        manifest["exit_code"] = process.returncode if process is not None else None
        manifest["status"] = "completed" if manifest["exit_code"] == 0 else "failed"
        if run_dir.is_dir():
            if log_path != run_dir / "train.log" and log_path.exists():
                log_path.replace(run_dir / "train.log")
                log_path = run_dir / "train.log"
            manifest_path = run_dir / "run_manifest.json"
        if manifest["exit_code"] == 0 and run_dir.is_dir():
            try:
                manifest["report"] = try_report(run_dir)
            except Exception as error:
                manifest["report"] = {
                    "attempted": True,
                    "error": f"{type(error).__name__}: {error}",
                }
        if run_dir.is_dir():
            manifest["measurements"] = measurements(run_dir, log_path, args.model)
            manifest["artifacts"] = {
                name: artifact(run_dir / name)
                for name in (
                    "latest.ckpt", "losses.csv", "loss_curves.png", "train.log",
                    "experiment_summary.generated.md",
                )
            }
            manifest["samples"] = sorted(path.name for path in (run_dir / "samples").glob("*.png"))
            manifest["snapshots"] = sorted(
                (
                    {"name": path.name, "bytes": path.stat().st_size}
                    for path in (run_dir / "snapshots").glob("*.pt")
                ),
                key=lambda item: item["name"],
            )
        atomic_json(manifest_path, manifest)
        lock_path.unlink(missing_ok=True)
    print(f"Run status: {manifest['status']}; manifest: {manifest_path}", flush=True)
    return manifest["exit_code"] if manifest["exit_code"] is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
