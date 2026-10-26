"""Load packaged classes, check full parameter counts and actual backbone calls."""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch


def check_submission(directory, device="cpu"):
    directory = Path(directory).resolve()
    if torch.device(device).type == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    # The checker itself runs in a fresh child process in the full pipeline.
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("packaged_model", directory/"model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for mode, expected in (("one_nfe", 1), ("few_nfe", 4)):
        model = module.Model.load_checkpoint(str(directory/"checkpoints"/(mode+".ckpt")), mode, device=device)
        if model.count_parameters() != 95_524_611:
            raise AssertionError("Wrong submission parameter count")
        count = []
        hook = model.backbone.network.register_forward_hook(lambda *unused: count.append(1))
        try:
            image = model.sample((1,3,64,64), category=torch.tensor([0], device=device), device=device)
        finally:
            hook.remove()
        if len(count) != expected or image.shape != (1,3,64,64) or not bool(torch.isfinite(image).all()) or not bool((image.abs() <= 1).all()):
            raise AssertionError("Actual NFE/output contract failed")
        print(f"{mode}: strict load, 95,524,611 parameters, {len(count)} calls, finite RGB64")
        del model
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    check_submission(a.directory, a.device)
