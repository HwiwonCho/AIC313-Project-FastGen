"""Copy compact evidence from one finished FastGen run into experiments/.

A run must have a successful wrapper manifest, complete epoch CSV/log, and a
checkpoint matching the manifest. This script never copies model weights, full
logs, generated evaluation images, or the dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
EXPERIMENTS = ROOT / "experiments"
MODELS = {"vae": "vae", "gan": "gan", "soflow_v2": "soflow"}
MAX_BYTES = {
    "run_manifest.json": 2_000_000,
    "run.json": 2_000_000,
    "requirements.at_run.txt": 1_000_000,
    "losses.csv": 5_000_000,
    "loss_curves.png": 10_000_000,
    "sample.png": 10_000_000,
    "fid_result.json": 2_000_000,
    "fid_result_nfe1.json": 2_000_000,
    "fid_result_nfe4.json": 2_000_000,
}
EPOCH_LINE = re.compile(r"^Epoch\s+(\d+)/(\d+)\s+\|")
SAMPLE_EPOCH = re.compile(r"epoch_(\d+)")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SETTINGS = (
    "device:", "gpu:", "train images:", "categories:", "batches per epoch:",
    "sample rows:", "parameters:", "generator parameters:",
    "discriminator parameters:",
)


class CurateError(Exception):
    pass


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def small_file(path: Path, limit: int) -> None:
    if not path.is_file():
        raise CurateError(f"required file is missing: {path}")
    size = path.stat().st_size
    if size > limit:
        raise CurateError(f"file exceeds {limit:,} bytes: {path} ({size:,})")


def read_json(path: Path, limit: int) -> dict:
    small_file(path, limit)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CurateError(f"invalid JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise CurateError(f"expected a JSON object: {path}")
    return value


def run_identity(path: Path) -> tuple[Path, str, str]:
    try:
        run_dir = path.resolve(strict=True)
        relative = run_dir.relative_to(RUNS.resolve(strict=True))
    except (OSError, ValueError) as error:
        raise CurateError("run directory must exist below this project's runs/") from error
    if not run_dir.is_dir() or len(relative.parts) != 2:
        raise CurateError("expected runs/MODEL/RUN_ID")
    model, run_id = relative.parts
    if model not in MODELS or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise CurateError("unsupported model folder or unsafe run ID")
    return run_dir, model, run_id


def sanitized_log(log_path: Path, expected_epochs: int) -> str:
    if not log_path.is_file():
        raise CurateError(f"training log is missing: {log_path}")
    settings: list[str] = []
    epochs: dict[int, str] = {}
    with log_path.open("r", encoding="utf-8", errors="replace", newline=None) as handle:
        for raw in handle:
            line = ANSI_ESCAPE.sub("", raw).strip()
            line = "".join(char if 32 <= ord(char) <= 126 else " " for char in line)
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            match = EPOCH_LINE.match(line)
            if match:
                epoch, total = (int(value) for value in match.groups())
                if total != expected_epochs or epoch in epochs:
                    raise CurateError(f"duplicate or mismatched epoch log line: {line[:160]}")
                epochs[epoch] = line[:600]
            elif line.startswith(SETTINGS) and len(settings) < 30:
                settings.append(line[:300])
    if sorted(epochs) != list(range(1, expected_epochs + 1)):
        missing = sorted(set(range(1, expected_epochs + 1)) - set(epochs))
        raise CurateError(f"training log has {len(epochs)}/{expected_epochs} epoch summaries; missing {missing[:12]}")
    return (
        "# Selected settings and every epoch summary from train.log\n"
        "# Batch tqdm output was excluded; full train.log remains in runs/.\n\n"
        + "\n".join(settings)
        + "\n\n"
        + "\n".join(epochs[epoch] for epoch in range(1, expected_epochs + 1))
        + "\n"
    )


def csv_last_row(path: Path, expected_epochs: int) -> dict:
    small_file(path, MAX_BYTES["losses.csv"])
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    try:
        observed = [int(row["epoch"]) for row in rows]
    except (KeyError, TypeError, ValueError) as error:
        raise CurateError("losses.csv must have integer epoch values") from error
    if observed != list(range(1, expected_epochs + 1)):
        raise CurateError(f"losses.csv has {len(observed)}/{expected_epochs} sequential epochs")
    return rows[-1]


def sample_file(run_dir: Path, checkpoint_epoch: int) -> tuple[Path, int]:
    candidates = []
    for path in (run_dir / "samples").glob("*.png"):
        match = SAMPLE_EPOCH.search(path.stem)
        if match:
            epoch = int(match.group(1))
            if epoch <= checkpoint_epoch:
                candidates.append((epoch, int(path.stem.endswith("_ema")), path))
    if not candidates:
        raise CurateError("no representative sample at or before the checkpoint epoch")
    epoch, _, path = max(candidates, key=lambda entry: (entry[0], entry[1], entry[2].name))
    small_file(path, MAX_BYTES["sample.png"])
    return path, epoch


def checkpoint_epoch(path: Path) -> int:
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
        del checkpoint
        return int(epoch)
    except Exception as error:
        raise CurateError(f"could not read checkpoint epoch: {type(error).__name__}: {error}") from error


def validate_fid(
    path: Path | None, model: str, checkpoint_hash: str, epoch: int,
    expected_nfe: int,
) -> dict | None:
    if path is None:
        return None
    result = read_json(path, MAX_BYTES["fid_result.json"])
    if result.get("model") != MODELS[model]:
        raise CurateError("FID result model does not match the training run")
    if result.get("checkpoint_sha256") != checkpoint_hash or result.get("checkpoint_epoch") != epoch:
        raise CurateError("FID result refers to a different checkpoint")
    if result.get("status") != "self_computed_fid_not_ta_official":
        raise CurateError("FID result has an unsupported protocol/status")
    if result.get("generated_image_count") != 3020 or result.get("reference_image_count") != 3020:
        raise CurateError("FID result does not contain 3,020 generated and reference images")
    if result.get("category_count") != 151 or result.get("samples_per_category") != 20:
        raise CurateError("FID result has a different category/sample protocol")
    if result.get("seed") != 20260929:
        raise CurateError("FID result has a different evaluation seed")
    val_split = ROOT / "data" / "pokemon-generation-one-22k" / "val_split.txt"
    if not val_split.is_file():
        raise CurateError(f"fixed validation split is missing: {val_split}")
    if result.get("reference_split_sha256") != hash_file(val_split):
        raise CurateError("FID result used a different validation split")
    actual_nfe = result.get("soflow_nfe")
    if model == "soflow_v2":
        if actual_nfe != expected_nfe:
            raise CurateError(f"SoFlow FID NFE {actual_nfe!r} differs from {expected_nfe}")
    elif expected_nfe != 1 or actual_nfe is not None:
        raise CurateError("VAE/GAN FID should use the single-forward protocol")
    if not isinstance(result.get("fid"), (int, float)) or not math.isfinite(result["fid"]):
        raise CurateError("FID result has no numeric fid")
    return result


def curate(args: argparse.Namespace) -> dict:
    run_dir, model, run_id = run_identity(args.run_dir)
    manifest = read_json(run_dir / "run_manifest.json", MAX_BYTES["run_manifest.json"])
    if manifest.get("model") != MODELS[model] or manifest.get("run_dir") != str(run_dir):
        raise CurateError("manifest does not identify this run directory/model")
    if manifest.get("status") != "completed" or manifest.get("exit_code") != 0:
        raise CurateError(f"run is not completed successfully (status={manifest.get('status')!r})")
    if (run_dir.parent / f"{run_id}.launch.lock").exists():
        raise CurateError("run launcher still holds a lock")
    if not manifest.get("ended_utc") or manifest.get("wall_seconds") is None:
        raise CurateError("completed manifest lacks end time or wall duration")
    expected_epochs = int(manifest["config"]["epochs"])
    if expected_epochs < 1:
        raise CurateError("invalid requested epoch count")
    csv_last_row(run_dir / "losses.csv", expected_epochs)
    log_summary = sanitized_log(run_dir / "train.log", expected_epochs)

    checkpoint = run_dir / "latest.ckpt"
    if not checkpoint.is_file():
        raise CurateError("latest.ckpt is missing")
    artifact = (manifest.get("artifacts") or {}).get("latest.ckpt") or {}
    expected_hash = artifact.get("sha256")
    if not expected_hash or artifact.get("bytes") != checkpoint.stat().st_size:
        raise CurateError("checkpoint size/hash metadata is missing or mismatched")
    actual_hash = hash_file(checkpoint)
    if actual_hash != expected_hash:
        raise CurateError("checkpoint SHA-256 changed since training finished")
    epoch = checkpoint_epoch(checkpoint)
    if epoch != expected_epochs:
        raise CurateError(f"checkpoint epoch {epoch} differs from completed epoch {expected_epochs}")
    sample, sample_epoch = sample_file(run_dir, epoch)

    required = {
        "run_manifest.json": run_dir / "run_manifest.json",
        "losses.csv": run_dir / "losses.csv",
        "loss_curves.png": run_dir / "loss_curves.png",
        "sample.png": sample,
    }
    optional = {
        "run.json": run_dir / "run.json",
        "requirements.at_run.txt": run_dir / "requirements.at_run.txt",
    }
    for name, path in required.items():
        small_file(path, MAX_BYTES[name])
    for name, path in optional.items():
        if path.is_file():
            small_file(path, MAX_BYTES[name])
            required[name] = path
    if "run.json" in required:
        read_json(required["run.json"], MAX_BYTES["run.json"])
    fid_path = args.fid_result
    if fid_path is None and (run_dir / "fid_result.json").is_file():
        fid_path = run_dir / "fid_result.json"
    if args.fid_result_few is not None and model != "soflow_v2":
        raise CurateError("--fid-result-few applies only to SoFlow v2")
    if args.fid_result_few is not None and fid_path is None:
        raise CurateError("SoFlow NFE 4 result also requires --fid-result for NFE 1")
    fid_one = validate_fid(fid_path, model, actual_hash, epoch, expected_nfe=1)
    fid_four = validate_fid(
        args.fid_result_few, model, actual_hash, epoch, expected_nfe=4
    )
    if fid_one and fid_four and fid_one.get("weight_key") != fid_four.get("weight_key"):
        raise CurateError("SoFlow NFE 1 and NFE 4 used different model weights")
    if fid_path is not None:
        name = "fid_result_nfe1.json" if model == "soflow_v2" else "fid_result.json"
        required[name] = fid_path
    if args.fid_result_few is not None:
        required["fid_result_nfe4.json"] = args.fid_result_few
    dest = EXPERIMENTS / model / run_id
    if dest.exists():
        raise CurateError(f"destination already exists; refusing overwrite: {dest}")
    plan = {
        "run": str(run_dir),
        "destination": str(dest),
        "dry_run": args.dry_run,
        "epochs": epoch,
        "checkpoint_sha256": actual_hash,
        "sample_epoch": sample_epoch,
        "files": sorted([*required, "train_log_summary.txt"]),
        "fid": {
            "nfe1": fid_one["fid"] if fid_one else None,
            "nfe4": fid_four["fid"] if fid_four else None,
        },
    }
    if args.dry_run:
        return plan

    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".curate-", dir=dest.parent))
    try:
        for name, source in required.items():
            shutil.copy2(source, staging / name)
        (staging / "train_log_summary.txt").write_text(log_summary, encoding="utf-8")
        staging.rename(dest)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--fid-result", type=Path, help="Optional NFE 1 FID JSON (also VAE/GAN)")
    parser.add_argument("--fid-result-few", type=Path, help="Optional SoFlow NFE 4 FID JSON")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the copy plan only")
    args = parser.parse_args()
    try:
        print(json.dumps(curate(args), ensure_ascii=False, indent=2))
        return 0
    except (CurateError, KeyError, ValueError, TypeError) as error:
        print(f"Cannot curate: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


