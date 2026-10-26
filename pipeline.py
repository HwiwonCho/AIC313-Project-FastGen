"""FastGen pipeline entry point. See README.md and configs/deadline.json."""
import argparse
import json
from pathlib import Path

from fastgen.project import ROOT, check_official, code_inventory, read_json, validate_config
from fastgen.orchestration import execute_pipeline
from fastgen.submission import export_ema, official_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run", "check", "_preflight", "_train", "_pairs", "_export", "_evaluate"])
    parser.add_argument("--config", type=Path, default=ROOT/"configs/deadline.json")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split-dir", type=Path, default=ROOT/"data/pokemon-generation-one-22k")
    parser.add_argument("--run-root", type=Path, default=ROOT/"runs/deadline")
    parser.add_argument("--task", type=Path)
    parser.add_argument("--continue-from", type=Path,
                        help="Continue a stopped teacher with full optimizer/RNG state; execution-only changes allowed")
    args = parser.parse_args()
    if args.command == "plan":
        check_official()
        cfg = read_json(args.config)
        validate_config(cfg)
        print(json.dumps({"configuration": cfg, "ordered_stages": ["pilot_flip", "pilot_spatial", "teacher_from_scratch",
                    "frozen_teacher_pairs", "optional_reflow_DISABLED_by_default", "warm_started_1NFE_distillation",
                    "strict_export", "NFE_and_parameter_checks", "provided_evaluator_1NFE_and_4NFE"],
                    "training_started": False, "server_actions": False}, indent=2))
    elif args.command == "check":
        check_official()
        print("Official fixed files and base Model class match the downloaded originals.")
    elif args.command == "run":
        if args.data_root is None:
            parser.error("--data-root is required")
        execute_pipeline(args)
    elif args.task is None:
        parser.error("Internal stage needs --task")
    elif args.command == "_preflight":
        from fastgen.preflight import preflight
        preflight(read_json(args.task))
    elif args.command == "_train":
        from fastgen.engine import train_stage
        if train_stage(read_json(args.task))["phase"] != "complete":
            raise SystemExit(75)
    elif args.command == "_pairs":
        from fastgen.pairs import generate_pairs
        generate_pairs(read_json(args.task))
    elif args.command == "_export":
        task = read_json(args.task)
        export_ema(task["source"], task["output"], task["mode"])
    elif args.command == "_evaluate":
        official_evaluation(read_json(args.task))


if __name__ == "__main__":
    main()
