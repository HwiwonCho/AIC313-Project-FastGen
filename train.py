import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import PokemonDataModule
from model import VAE


def vae_loss(
    recons,
    images,
    mean,
    logvar,
    prior_mean,
    prior_logvar,
    beta=1.0,
):
    reconstruction_loss = F.l1_loss(
        recons,
        images,
        reduction="none",
    )
    reconstruction_loss = reconstruction_loss.flatten(start_dim=1).sum(dim=1).mean()

    variance_ratio = torch.exp(logvar - prior_logvar)
    squared_mean_distance = (mean - prior_mean).pow(2) * torch.exp(-prior_logvar)
    kl_loss = 0.5 * (
        prior_logvar - logvar + variance_ratio + squared_mean_distance - 1
    )
    kl_loss = kl_loss.sum(dim=1).mean()

    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss


def save_generated_images(
    model,
    fixed_epsilon,
    fixed_categories,
    epoch,
    output_dir="runs/conditional_v4_samples",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        prior_mean, prior_logvar = model.prior(fixed_categories)
        fixed_z = prior_mean + torch.exp(0.5 * prior_logvar) * fixed_epsilon
        generated = model.decode(fixed_z, fixed_categories)

    save_image(
        generated,
        output_dir / f"epoch_{epoch:03d}.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )


def save_reconstructions(
    model,
    fixed_images,
    fixed_categories,
    epoch,
    output_dir="runs/conditional_v4_reconstructions",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        mean, _ = model.encode(fixed_images, fixed_categories)
        reconstructed = model.decode(mean, fixed_categories)

    comparison = torch.cat([fixed_images, reconstructed], dim=0)
    save_image(
        comparison,
        output_dir / f"epoch_{epoch:03d}.png",
        nrow=fixed_images.shape[0],
        normalize=True,
        value_range=(-1, 1),
    )


def train_one_epoch(model, train_loader, optimizer, device, beta):
    model.train()
    totals = {
        "total_loss": 0.0,
        "reconstruction_loss": 0.0,
        "kl_loss": 0.0,
    }
    total_images = 0
    progress = tqdm(train_loader, desc="Training")

    for images, categories in progress:
        images = images.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)
        batch_size = images.shape[0]

        recons, mean, logvar = model(images, categories)
        prior_mean, prior_logvar = model.prior(categories)
        loss, reconstruction_loss, kl_loss = vae_loss(
            recons,
            images,
            mean,
            logvar,
            prior_mean,
            prior_logvar,
            beta=beta,
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = loss.item()
        reconstruction_value = reconstruction_loss.item()
        kl_value = kl_loss.item()
        totals["total_loss"] += loss_value * batch_size
        totals["reconstruction_loss"] += reconstruction_value * batch_size
        totals["kl_loss"] += kl_value * batch_size
        total_images += batch_size

        progress.set_postfix(
            loss=f"{loss_value:.1f}",
            recon=f"{reconstruction_value:.1f}",
            kl=f"{kl_value:.1f}",
            beta=f"{beta:.3f}",
        )

    return {name: value / total_images for name, value in totals.items()}


LOSS_FIELDS = [
    "epoch",
    "beta",
    "total_loss",
    "reconstruction_loss",
    "kl_loss",
    "epoch_seconds",
    "elapsed_seconds",
    "peak_vram_gb",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional Pokemon VAE")
    parser.add_argument("--run-dir", type=Path, help="New run directory under runs/")
    parser.add_argument("--resume", action="store_true", help="Resume run-dir/latest.ckpt")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        parser.error("epochs and batch-size must be positive; workers cannot be negative")
    if args.resume and args.run_dir is None:
        parser.error("--resume requires --run-dir")
    return args


def write_losses(path, history):
    temporary_path = path.with_name(path.name + ".tmp")
    with temporary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=LOSS_FIELDS)
        writer.writeheader()
        writer.writerows(history)
    temporary_path.replace(path)


def save_checkpoint(path, model, optimizer, epoch, history, fixed_inputs, config):
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model_type": "vae",
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
            "fixed_inputs": fixed_inputs,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary_path,
    )
    temporary_path.replace(path)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = args.run_dir or Path("runs/vae") / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{os.getpid()}"
    )
    if args.resume:
        if not (run_dir / "latest.ckpt").is_file():
            raise FileNotFoundError(f"No checkpoint at {run_dir / 'latest.ckpt'}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("run directory:", run_dir.resolve())

    data_module = PokemonDataModule(
        batch_size=args.batch_size,
        num_workers=args.workers,
        return_category=True,
    )
    train_loader = data_module.train_dataloader()

    model = VAE(
        latent_dim=128,
        num_categories=len(data_module.category_to_id),
        category_dim=64,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    sample_category_names = ("Pikachu", "Charmander", "Squirtle", "Bulbasaur")
    variations_per_category = 4
    variation_z = torch.randn(
        variations_per_category,
        model.latent_dim,
        device=device,
    )
    fixed_epsilon = variation_z.repeat(len(sample_category_names), 1)
    fixed_categories = torch.tensor(
        [
            data_module.category_to_id[name]
            for name in sample_category_names
            for _ in range(variations_per_category)
        ],
        dtype=torch.long,
        device=device,
    )

    fixed_images, fixed_reconstruction_categories = next(iter(train_loader))
    fixed_images = fixed_images[:8].to(device, non_blocking=True)
    fixed_reconstruction_categories = fixed_reconstruction_categories[:8].to(
        device,
        non_blocking=True,
    )

    print("train images:", len(data_module.train_dataset))
    print("categories:", len(data_module.category_to_id))
    print("batches per epoch:", len(train_loader))
    print("sample rows:", sample_category_names)
    initial_parameter_counts = {
        "model": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    print("parameters:", initial_parameter_counts["model"])

    num_epochs = args.epochs
    warmup_epochs = 10
    min_beta = 0.25
    max_beta = 1.0
    config = {
        "model_type": "vae",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "learning_rate": 2e-4,
        "latent_dim": model.latent_dim,
        "num_categories": len(data_module.category_to_id),
        "warmup_epochs": warmup_epochs,
        "min_beta": min_beta,
        "max_beta": max_beta,
        "train_images": len(data_module.train_dataset),
        "train_split": "data/pokemon-generation-one-22k/train_split.txt",
        "initial_parameter_counts": initial_parameter_counts,
    }
    fixed_inputs = {
        "epsilon": fixed_epsilon,
        "categories": fixed_categories,
        "images": fixed_images,
        "reconstruction_categories": fixed_reconstruction_categories,
    }
    history = []
    start_epoch = 0
    elapsed_seconds = 0.0
    if args.resume:
        checkpoint = torch.load(run_dir / "latest.ckpt", map_location=device, weights_only=True)
        if checkpoint.get("model_type") != "vae":
            raise ValueError("Checkpoint is not a VAE run")
        saved_config = checkpoint["config"]
        for key in ("batch_size", "latent_dim", "num_categories", "warmup_epochs", "min_beta", "max_beta"):
            if saved_config[key] != config[key]:
                raise ValueError(f"Cannot resume with a different {key}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        start_epoch = int(checkpoint["epoch"])
        if len(history) != start_epoch:
            raise ValueError("Checkpoint history does not match its epoch")
        elapsed_seconds = history[-1]["elapsed_seconds"] if history else 0.0
        fixed_inputs = checkpoint["fixed_inputs"]
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng_state"]:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        write_losses(run_dir / "losses.csv", history)
        print(f"resumed after epoch {start_epoch}")
    else:
        with (run_dir / "run.json").open("w") as file:
            json.dump(
                {
                    **config,
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "torch_version": torch.__version__,
                },
                file,
                indent=2,
            )

    for epoch in range(start_epoch, num_epochs):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_start = time.perf_counter()
        warmup_fraction = min(1.0, epoch / warmup_epochs)
        beta = min_beta + (max_beta - min_beta) * warmup_fraction
        result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            beta=beta,
        )
        result["epoch"] = epoch + 1
        result["beta"] = beta
        result["epoch_seconds"] = time.perf_counter() - epoch_start
        elapsed_seconds += result["epoch_seconds"]
        result["elapsed_seconds"] = elapsed_seconds
        result["peak_vram_gb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        )
        history.append(result)

        recon_per_pixel = result["reconstruction_loss"] / (3 * 64 * 64)
        kl_per_dimension = result["kl_loss"] / model.latent_dim
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss={result['total_loss']:.2f} | "
            f"recon={result['reconstruction_loss']:.2f} | "
            f"kl={result['kl_loss']:.2f} | "
            f"beta={beta:.3f} | "
            f"recon/pixel={recon_per_pixel:.4f} | "
            f"kl/dim={kl_per_dimension:.4f}"
        )

        if (epoch + 1) % 5 == 0:
            save_generated_images(
                model,
                fixed_inputs["epsilon"],
                fixed_inputs["categories"],
                epoch + 1,
                output_dir=run_dir / "samples",
            )
            save_reconstructions(
                model,
                fixed_inputs["images"],
                fixed_inputs["reconstruction_categories"],
                epoch + 1,
                output_dir=run_dir / "reconstructions",
            )

        write_losses(run_dir / "losses.csv", history)
        save_checkpoint(
            run_dir / "latest.ckpt",
            model,
            optimizer,
            epoch + 1,
            history,
            fixed_inputs,
            config,
        )

    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.setdefault("initial_parameter_counts", initial_parameter_counts)
    metadata["final_parameter_counts"] = {
        "model": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }
    metadata["completed_epochs"] = max(start_epoch, num_epochs)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    temporary_path = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2))
    temporary_path.replace(metadata_path)


if __name__ == "__main__":
    main()
