"""Self-computed FID for a fixed VAE, GAN, or SoFlow training checkpoint.

This uses the starter evaluator's image export and clean-fid functions, but it
does not certify the submission ModelOneNFE/ModelFewNFE interface. Never use
validation FID to choose checkpoints, architectures, or hyperparameters.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn

from dataset import PokemonDataModule
from evaluate import compute_fid, generate_samples, prepare_reference_set
from gan_model import ConditionalGenerator
from model import VAE
from soflow_model import SoFlow64


EXPECTED_CATEGORIES = 151
SAMPLES_PER_CATEGORY = 20
EXPECTED_IMAGES = EXPECTED_CATEGORIES * SAMPLES_PER_CATEGORY
MAX_MODEL_PARAMETERS = 100_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SamplingAdapter(nn.Module):
    """Expose the existing training model through evaluate.generate_samples()."""

    def __init__(self, model, kind, nfe):
        super().__init__()
        self.model = model
        self.kind = kind
        self.nfe = nfe

    @torch.inference_mode()
    def sample(self, shape, *, device, category):
        count, channels, height, width = shape
        if (channels, height, width) != (3, 64, 64):
            raise ValueError(f"Expected (N, 3, 64, 64), got {shape}")
        if category is None or category.shape != (count,):
            raise ValueError("A category ID is required for every sample")
        if self.kind == "vae":
            mean, logvar = self.model.prior(category)
            epsilon = torch.randn(count, self.model.latent_dim, device=device)
            latent = mean + torch.exp(0.5 * logvar) * epsilon
            return self.model.decode(latent, category)
        if self.kind == "gan":
            latent = torch.randn(count, self.model.latent_dim, device=device)
            return self.model(latent, category)
        noise = torch.randn(shape, device=device)
        return self.model.sample(noise, category, nfe=self.nfe)


def load_candidate(args, num_categories):
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a training checkpoint dictionary")

    if args.model == "vae":
        if checkpoint.get("model_type") != "vae" or "model" not in checkpoint:
            raise ValueError("This is not a fresh VAE training checkpoint")
        state = checkpoint["model"]
        model = VAE(
            latent_dim=state["prior_mean.weight"].shape[1],
            num_categories=num_categories,
            category_dim=state["category_embedding.weight"].shape[1],
        )
        weight_key = "model"
    elif args.model == "gan":
        if checkpoint.get("model_type") != "gan" or "generator" not in checkpoint:
            raise ValueError("This is not a fresh GAN training checkpoint")
        state = checkpoint["generator"]
        model = ConditionalGenerator(
            latent_dim=checkpoint["config"]["latent_dim"],
            num_categories=num_categories,
            category_dim=state["category_embedding.weight"].shape[1],
        )
        weight_key = "generator"
    else:
        weight_key = "ema_model" if args.soflow_weights == "ema" else "model"
        if weight_key not in checkpoint:
            raise ValueError(f"SoFlow checkpoint has no {weight_key!r} state")
        state = checkpoint[weight_key]
        model = SoFlow64(
            num_categories=num_categories,
            base_channels=args.soflow_base_channels,
        )

    epoch = checkpoint.get("epoch")
    model.load_state_dict(state, strict=True)
    del checkpoint, state
    return SamplingAdapter(model, args.model, args.soflow_nfe), weight_key, epoch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=("vae", "gan", "soflow"))
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Explicit, preselected latest.ckpt; never auto-select by FID")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New, nonexistent directory for this evaluation")
    parser.add_argument("--data-root", type=Path,
                        default=Path("data/pokemon-generation-one-22k"))
    parser.add_argument("--split-dir", type=Path,
                        default=Path("data/pokemon-generation-one-22k"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--soflow-nfe", type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument("--soflow-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--soflow-base-channels", type=int, default=128)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0 or args.soflow_base_channels < 1:
        parser.error("batch-size and soflow-base-channels must be positive; workers >= 0")
    if args.model != "soflow" and args.soflow_nfe != 1:
        parser.error("--soflow-nfe > 1 applies only to SoFlow")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {args.checkpoint}")
    if args.output_dir.exists():
        parser.error(f"Output directory must not exist: {args.output_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    for name in ("train_split.txt", "val_split.txt", "category_to_id.json"):
        if not (args.split_dir / name).is_file():
            parser.error(f"Missing provided split/mapping file: {args.split_dir / name}")
    return args


def main():
    args = parse_args()
    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    checkpoint_sha256_before = sha256(args.checkpoint)

    data_module = PokemonDataModule(
        data_root=args.data_root,
        split_dir=args.split_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    mapping = data_module.category_to_id
    val_counts = Counter(data_module.val_dataset.categories)
    if len(mapping) != EXPECTED_CATEGORIES or sorted(mapping.values()) != list(range(EXPECTED_CATEGORIES)):
        raise ValueError("The fixed category mapping is not 151 contiguous IDs")
    if len(data_module.val_dataset) != EXPECTED_IMAGES or set(val_counts.values()) != {SAMPLES_PER_CATEGORY}:
        raise ValueError("The validation split must contain 20 images for each of 151 categories")

    sampler, weight_key, checkpoint_epoch = load_candidate(args, len(mapping))
    parameter_count = sum(p.numel() for p in sampler.parameters())
    if parameter_count >= MAX_MODEL_PARAMETERS:
        raise ValueError(f"Sampling model has {parameter_count:,} parameters (limit is fewer than 100M)")
    sampler = sampler.to(args.device).eval()

    # Seed after model construction; reset again after reference export.
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    reference_dir = args.output_dir / "reference"
    generated_dir = args.output_dir / "generated"
    prepare_reference_set(data_module, reference_dir)
    if len(list(reference_dir.glob("*.png"))) != EXPECTED_IMAGES:
        raise RuntimeError("Reference export did not produce exactly 3,020 PNGs")

    # Validation DataLoader setup consumes RNG; seed sampling independently.
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    categories = torch.arange(EXPECTED_CATEGORIES, dtype=torch.long).repeat_interleave(SAMPLES_PER_CATEGORY)
    generate_samples(
        sampler,
        generated_dir,
        categories,
        batch_size=args.batch_size,
        device=args.device,
    )
    if len(list(generated_dir.glob("*.png"))) != EXPECTED_IMAGES:
        raise RuntimeError("Generation did not produce exactly 3,020 PNGs")

    score = compute_fid(
        generated_dir,
        reference_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
    if sha256(args.checkpoint) != checkpoint_sha256_before:
        raise RuntimeError("Checkpoint changed during evaluation; FID result was not saved")
    ended = datetime.now(timezone.utc)
    result = {
        "status": "self_computed_fid_not_ta_official",
        "use": "reporting_only_not_for_checkpoint_or_hyperparameter_selection",
        "official_submission_interface_verified": False,
        "model": args.model,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_sha256": checkpoint_sha256_before,
        "weight_key": weight_key,
        "parameter_count": parameter_count,
        "soflow_nfe": args.soflow_nfe if args.model == "soflow" else None,
        "seed": args.seed,
        "argv": sys.argv,
        "data_root": str(args.data_root.resolve()),
        "reference_split": str((args.split_dir / "val_split.txt").resolve()),
        "reference_split_sha256": sha256(args.split_dir / "val_split.txt"),
        "category_mapping_sha256": sha256(args.split_dir / "category_to_id.json"),
        "evaluation_code_sha256": sha256(Path(__file__).with_name("evaluate.py")),
        "self_eval_code_sha256": sha256(__file__),
        "reference_image_count": EXPECTED_IMAGES,
        "generated_image_count": EXPECTED_IMAGES,
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "category_count": EXPECTED_CATEGORIES,
        "fid": score,
        "clean_fid_version": importlib.metadata.version("clean-fid"),
        "torch_version": torch.__version__,
        "device": str(args.device),
        "batch_size": args.batch_size,
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "wall_seconds": time.perf_counter() - start_clock,
    }
    (args.output_dir / "fid_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
