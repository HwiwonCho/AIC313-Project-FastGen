"""Self-contained export, packaging, final evaluation and write-up records."""
import shutil
import os
import subprocess
import sys
import time
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


from .project import code_inventory, read_json


def export_ema(source, output, mode):
    import torch
    from fastgen.engine import DEFAULT_ARCHITECTURE
    from fastgen.runtime import atomic_json, atomic_save, sha256
    source, output = Path(source), Path(output)
    source_hash = sha256(source)
    provenance = {"source_sha256": source_hash, "mode": mode}
    metadata = output.with_suffix(".provenance.json")
    if metadata.exists():
        if read_json(metadata)["source"] != provenance:
            raise ValueError("Existing exported checkpoint belongs to a different source")
        if not output.exists() or read_json(metadata)["checkpoint_sha256"] != sha256(output):
            raise ValueError("Exported checkpoint changed")
        return
    saved = torch.load(source, map_location="cpu", weights_only=False)
    expected_kind = "distill" if mode == "one_nfe" else "soflow"
    if saved["architecture"] != DEFAULT_ARCHITECTURE or saved["identity"]["settings"]["kind"] != expected_kind:
        raise ValueError("Export source architecture/objective does not match the submission model")
    state = {"backbone."+k: v for k,v in saved["ema"].items()}
    if output.exists():
        # Finish an interrupted export only after proving exact EMA equality.
        existing = torch.load(output, map_location="cpu", weights_only=True)
        existing = existing.get("state_dict", existing)
        if set(existing) != set(state) or not all(torch.equal(existing[k], state[k]) for k in state):
            raise ValueError("Unfinished export does not match this source EMA")
        del existing
    else:
        atomic_save(output, {"state_dict": state})
    del state
    del saved
    from model import Model
    model = Model.load_checkpoint(str(output), mode, device="cpu")
    if model.count_parameters() != 95_524_611:
        raise ValueError("Exported model parameter count differs")
    atomic_json(metadata, {"source": provenance, "source_path": str(source),
                           "checkpoint_sha256": sha256(output), "parameters": model.count_parameters(),
                           "strict_official_base_loader_passed": True})


def bundle_submission(directory):
    from fastgen.runtime import atomic_json, sha256
    submission = directory / "submission"
    names = code_inventory()
    # Include every training stage and fixed split; source snapshot is the run's actual version.
    names.update({str(p.relative_to(ROOT)).replace("\\", "/"): None
                  for p in (ROOT / "data/pokemon-generation-one-22k").iterdir() if p.is_file()})
    for name in names:
        target = submission / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copytree(directory / "source_snapshot", submission / "records/source_snapshot", dirs_exist_ok=True)
    for name in ("pipeline_identity.json", "augmentation_selection.json", "execution.jsonl", "preflight.json",
                 "writeup_facts.json", "frozen_before_official_evaluation.json"):
        if (directory / name).exists():
            shutil.copyfile(directory / name, submission / "records" / name)
    for stage in ("pilot_flip", "pilot_spatial", "teacher", "reflow", "distill"):
        src = directory / stage
        if not src.exists():
            continue
        dest = submission / "records" / stage
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("identity.json", "summary.json", "environment.json", "events.jsonl", "metrics.jsonl", "loss_curve.png"):
            if (src/name).exists():
                shutil.copyfile(src/name, dest/name)
        # Early samples are immutable and included individually, as required.
        if (src/"samples/step_0000001").exists():
            shutil.copytree(src/"samples/step_0000001", dest/"early_samples", dirs_exist_ok=True)
        if (src/"sessions").exists():
            shutil.copytree(src/"sessions", dest/"sessions", dirs_exist_ok=True)
    for pool in ("pairs_teacher", "pairs_reflow"):
        for name in ("protocol.json", "index.json", "summary.json"):
            source = directory/pool/name
            if source.exists():
                target = submission/"records"/pool/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    if (directory/"adopted_history").exists():
        shutil.copytree(directory/"adopted_history", submission/"records/adopted_history", dirs_exist_ok=True)
    evaluation = directory/"official_evaluation"
    if evaluation.exists():
        for source in evaluation.rglob("*"):
            if source.is_file() and source.suffix in (".json", ".log"):
                target = submission/"records/official_evaluation"/source.relative_to(evaluation)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
    atomic_json(submission / "submission_manifest.json", {"sha256": {
        p.relative_to(submission).as_posix(): sha256(p) for p in submission.rglob("*") if p.is_file()
        and p.name != "submission_manifest.json"}, "official_validation_used_for_selection": False})


def evaluate_submission(directory, data_root, split_dir, cfg):
    from .orchestration import child
    from fastgen.runtime import atomic_json, sha256
    freeze = {mode: sha256(directory / "submission/checkpoints" / (mode+".ckpt"))
              for mode in ("one_nfe", "few_nfe")}
    atomic_json(directory / "frozen_before_official_evaluation.json", freeze)
    # Child processes alone import the evaluator and open validation images.
    for mode in ("one_nfe", "few_nfe"):
        task = {"name": "official_"+mode, "mode": mode, "run_root": str(directory),
                "data_root": str(data_root), "split_dir": str(split_dir), "device": cfg["device"],
                "batch_size": cfg["final_evaluation_batch_size"], "seed": cfg["final_evaluation_seed"],
                "checkpoint_sha256": freeze[mode]}
        child("_evaluate", task, directory)


def official_evaluation(task):
    import re
    import torch
    from PIL import Image
    from fastgen.data import safe_image
    from fastgen.runtime import atomic_json, seed_all, sha256
    directory = Path(task["run_root"])
    submission = directory / "submission"
    checkpoint = submission / "checkpoints" / (task["mode"]+".ckpt")
    if sha256(checkpoint) != task["checkpoint_sha256"]:
        raise ValueError("Frozen submission checkpoint changed")
    result_path = directory / "official_evaluation" / (task["mode"]+".json")
    if result_path.exists():
        if read_json(result_path)["checkpoint_sha256"] != task["checkpoint_sha256"]:
            raise ValueError("Previous final evaluation used another checkpoint")
        return
    # A fresh process must resolve every import from the submission itself.
    # In-process checking can accidentally reuse modules from the source tree.
    isolated_env = {k:v for k,v in os.environ.items() if k != "PYTHONPATH"}
    subprocess.run([sys.executable, "-B", "submission_check.py", ".", "--device", task["device"]],
                   cwd=submission, env=isolated_env, check=True)
    # The official downloader normalizes images in place. Give it a private
    # opaque-RGB validation view so it cannot rewrite the original dataset.
    view = directory / "official_evaluation/data_view"
    view.mkdir(parents=True, exist_ok=True)
    paths = [p.strip() for p in (Path(task["split_dir"])/"val_split.txt").read_text().splitlines() if p.strip()]
    for relative in paths:
        target = view / relative
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(safe_image(task["data_root"], relative)) as image:
                rgba = image.convert("RGBA")
                rgb = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
                rgb.save(target, format="PNG")  # Lossless even if the original suffix is .jpg.
    (view / ".transparency_composited_on_white").touch()
    attempt = directory / "official_evaluation" / (task["mode"]+"_"+str(time.time_ns()))
    attempt.mkdir()
    # Run the exact provided evaluate.py with a fixed seed, from the packaged folder.
    code = "import runpy,torch;torch.manual_seed("+str(task["seed"])+");torch.cuda.manual_seed_all("+str(task["seed"])+");runpy.run_path('evaluate.py',run_name='__main__')"
    command = [sys.executable, "-B", "-c", code, "--model_checkpoint", str(checkpoint),
               "--evaluate_mode", task["mode"], "--data_root", str(view),
               "--split_dir", task["split_dir"], "--reference_dir", str(attempt/"reference"),
               "--output_dir", str(attempt/"generated"), "--batch_size", str(task["batch_size"]),
               "--device", task["device"]]
    started = time.perf_counter()
    with (attempt/"console.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=submission, env=isolated_env, stdout=log, stderr=subprocess.STDOUT)
    output = (attempt/"console.log").read_text(encoding="utf-8")
    match = re.search(re.escape(task["mode"])+r" FID: ([\d.eE+-]+)", output)
    if result.returncode or not match:
        raise RuntimeError(f"Provided evaluator failed; see {attempt/'console.log'}")
    if sha256(checkpoint) != task["checkpoint_sha256"]:
        raise ValueError("Checkpoint changed during evaluation")
    atomic_json(result_path, {"mode": task["mode"], "FID_printed_by_official_evaluator": float(match.group(1)),
                "precision_of_printed_FID": "four decimal places; see original console log",
                "checkpoint_sha256": task["checkpoint_sha256"], "seed": task["seed"],
                "batch_size": task["batch_size"], "samples": len(paths),
                "wall_seconds": time.perf_counter()-started, "console": str(attempt/"console.log"),
                "TA_measured": False, "used_for_training_or_selection": False})


def write_facts(directory):
    from fastgen.runtime import atomic_json
    records = [json.loads(x) for x in (directory/"execution.jsonl").read_text().splitlines()]
    facts = {"child_process_wall_seconds_all_attempts": sum(r.get("process_wall_seconds", 0) for r in records),
             "pipeline_session_wall_seconds_all_attempts": sum(r.get("session_wall_seconds",0) for r in records if r.get("event")=="pipeline_session_end"),
             "timing_scope": "Sum of sequential child lifetimes, including evaluation/checkpoint work; do not add nested stage active times.",
             "stages": {name: read_json(directory/name/"summary.json") for name in
                        ("pilot_flip", "pilot_spatial", "teacher", "reflow", "distill") if (directory/name/"summary.json").exists()},
             "evaluations": {name: read_json(directory/"official_evaluation"/(name+".json")) for name in
                            ("one_nfe", "few_nfe") if (directory/"official_evaluation"/(name+".json")).exists()}}
    atomic_json(directory/"writeup_facts.json", facts)
    if (directory/"submission/records").exists():
        shutil.copyfile(directory/"writeup_facts.json", directory/"submission/records/writeup_facts.json")


