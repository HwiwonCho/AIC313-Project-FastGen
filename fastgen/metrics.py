"""Training selection only reads references selected from the TRAIN manifest."""
from contextlib import contextmanager
from pathlib import Path

import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import save_image

from .data import image64, safe_image
from .runtime import atomic_json, digest, preserved_rng, sha256
from .sampling import sample_images


@contextmanager
def peer_evaluation_precision():
    """Fresh PyTorch defaults used by the inspected Yunseo runner.

    Our training enables TF32 matmul; do not let it leak into generation or
    Clean-FID feature extraction. Always restore training settings afterwards.
    """
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = previous


@contextmanager
def counted_backbone(model):
    calls = {"batches": 0, "image_forwards": 0}
    def count(_network, inputs, _output):
        calls["batches"] += 1
        calls["image_forwards"] += len(inputs[0])
    hook = model.network.register_forward_hook(count)
    try:
        yield calls
    finally:
        hook.remove()


@torch.no_grad()
def preview(model, directory, *, device, method, precision, seed=12345):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with preserved_rng():
        generator = torch.Generator(device=device).manual_seed(seed)
        extra = torch.Generator(device=device).manual_seed(seed + 1)
        labels = torch.arange(8, device=device) % model.num_categories
        images = sample_images(model, (8, 3, 64, 64), labels, device=device, method=method,
                               generator=generator, extra_generator=extra, precision=precision)
        save_image((images + 1) / 2, directory / "grid.png", nrow=4)
        for i, image in enumerate(images):
            to_pil_image((image.cpu() + 1) / 2).save(directory / f"sample_{i:02d}.png")


@torch.no_grad()
def train_proxy(model, manifest, root, directory, *, device, method, batch_size,
                precision="fp32", seed=27182, workers=4,
                reference_overlaps_training=True, fid_function=None):
    directory = Path(directory)
    generated, reference = directory / "generated", directory / "reference"
    generated.mkdir(parents=True, exist_ok=False)
    reference.mkdir()
    references = manifest["references"]
    for i, row in enumerate(references):
        # Match Yunseo's scaled_train.evaluate_proxy tensor round-trip exactly.
        from torchvision.transforms import functional as TF
        tensor = TF.to_tensor(image64(safe_image(root, row["path"]))).mul(2).sub(1)
        TF.to_pil_image(((tensor+1)/2).clamp(0,1)).convert("RGB").save(reference / f"{i:06d}.png")
    labels = torch.tensor([row["category"] for row in references], dtype=torch.long)
    with preserved_rng(), peer_evaluation_precision(), counted_backbone(model) as actual_calls:
        generator = torch.Generator(device=device).manual_seed(seed)
        extra = torch.Generator(device=device).manual_seed(20261008)
        for start in range(0, len(labels), batch_size):
            cats = labels[start:start + batch_size].to(device)
            images = sample_images(model, (len(cats), 3, 64, 64), cats, device=device,
                                   method=method, generator=generator, extra_generator=extra,
                                   precision=precision)
            for i, image in enumerate(images):
                to_pil_image(((image.cpu() + 1) / 2).clamp(0,1)).convert("RGB").save(generated / f"{start+i:06d}.png")
        if fid_function is None:
            from cleanfid.fid import compute_fid
            fid_function = compute_fid
        score = float(fid_function(str(generated), str(reference), device=torch.device(device),
                                   batch_size=batch_size, num_workers=workers))
    nfe = {"endpoint1": 1, "renoise4": 4, "solution4": 4, "euler4": 4, "euler32": 32}[method]
    if actual_calls["image_forwards"] != len(labels)*nfe:
        raise RuntimeError("Observed model calls do not match the declared FID NFE")
    result = {"FID_train_proxy": score, "samples": len(labels), "seed": seed,
              "method": method, "reference_groups_sha256": digest(references),
              "official_validation_accessed": False, "not_a_TA_score": True,
              "protocol": "yunseo_scaled_a80bbd0_train755_defaults_v1",
              "reference_is_training_data": reference_overlaps_training,
              "generation_precision": precision, "batch_size": batch_size, "workers": workers,
              "actual_nfe": nfe, "actual_backbone_calls": actual_calls,
              "matmul_tf32": False, "cudnn_tf32": True, "cudnn_benchmark": False,
              "class_order": "0..150, five consecutive images per class",
              "reference_png_sha256": digest([sha256(p) for p in sorted(reference.glob('*.png'))]),
              "clean_fid_mode": "clean (library default)", "extra_renoise_seed": 20261008 if method=='renoise4' else None}
    from importlib.metadata import version
    result["metric_packages"] = {name: version(name) for name in ("torch", "torchvision", "Pillow", "clean-fid", "numpy", "scipy")}
    atomic_json(directory / "result.json", result)
    return result
