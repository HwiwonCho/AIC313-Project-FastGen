import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.nn import functional as F
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import PokemonDataModule
from gan_model import ConditionalGenerator, ProjectionDiscriminator


def save_samples(
    generator,
    fixed_z,
    fixed_categories,
    epoch,
    output_dir="runs/conditional_gan_v3_samples",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        samples = generator(fixed_z, fixed_categories)
    save_image(
        samples,
        output_dir / f"epoch_{epoch:03d}.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )


def train_one_epoch(
    generator,
    discriminator,
    train_loader,
    generator_optimizer,
    discriminator_optimizer,
    device,
    diversity_weight=0.2,
):
    generator.train()
    discriminator.train()
    generator_total = 0.0
    discriminator_total = 0.0
    real_score_total = 0.0
    fake_score_total = 0.0
    total_images = 0
    progress = tqdm(train_loader, desc="GAN training")

    for real_images, categories in progress:
        real_images = real_images.to(device, non_blocking=True)
        categories = categories.to(device, non_blocking=True)
        batch_size = real_images.shape[0]

        z = torch.randn(batch_size, generator.latent_dim, device=device)
        with torch.no_grad():
            fake_images = generator(z, categories)

        real_scores = discriminator(real_images, categories)
        fake_scores = discriminator(fake_images, categories)
        discriminator_loss = (
            F.relu(1 - real_scores).mean()
            + F.relu(1 + fake_scores).mean()
        )

        discriminator_optimizer.zero_grad(set_to_none=True)
        discriminator_loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        discriminator_optimizer.step()

        z = torch.randn(batch_size, generator.latent_dim, device=device)
        second_z = torch.randn(batch_size, generator.latent_dim, device=device)
        generated_images = generator(z, categories)
        second_generated_images = generator(second_z, categories)
        generator_scores = discriminator(generated_images, categories)
        adversarial_loss = -generator_scores.mean()
        image_distance = (generated_images - second_generated_images).abs().mean()
        latent_distance = (z - second_z).abs().mean()
        diversity_ratio = image_distance / (latent_distance + 1e-6)
        generator_loss = adversarial_loss - diversity_weight * diversity_ratio

        generator_optimizer.zero_grad(set_to_none=True)
        generator_loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 5.0)
        generator_optimizer.step()

        generator_value = generator_loss.item()
        discriminator_value = discriminator_loss.item()
        real_score_value = real_scores.mean().item()
        fake_score_value = fake_scores.mean().item()
        generator_total += generator_value * batch_size
        discriminator_total += discriminator_value * batch_size
        real_score_total += real_score_value * batch_size
        fake_score_total += fake_score_value * batch_size
        total_images += batch_size

        progress.set_postfix(
            generator=f"{generator_value:.3f}",
            discriminator=f"{discriminator_value:.3f}",
            real=f"{real_score_value:.3f}",
            fake=f"{fake_score_value:.3f}",
        )

    return {
        "generator_loss": generator_total / total_images,
        "discriminator_loss": discriminator_total / total_images,
        "real_score": real_score_total / total_images,
        "fake_score": fake_score_total / total_images,
    }


LOSS_FIELDS = [
    "epoch",
    "generator_loss",
    "discriminator_loss",
    "real_score",
    "fake_score",
    "epoch_seconds",
    "elapsed_seconds",
    "peak_vram_gb",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a conditional Pokemon GAN")
    parser.add_argument("--run-dir", type=Path, help="New run directory under runs/")
    parser.add_argument("--resume", action="store_true", help="Resume run-dir/latest.ckpt")
    parser.add_argument("--epochs", type=int, default=50, help="Total epochs")
    parser.add_argument("--batch-size", type=int, default=64)
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


def save_checkpoint(
    path,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    epoch,
    history,
    fixed_inputs,
    config,
):
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model_type": "gan",
            "epoch": epoch,
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "generator_optimizer": generator_optimizer.state_dict(),
            "discriminator_optimizer": discriminator_optimizer.state_dict(),
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

    run_dir = args.run_dir or Path("runs/gan") / (
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
    num_categories = len(data_module.category_to_id)

    generator = ConditionalGenerator(
        latent_dim=128,
        num_categories=num_categories,
        category_dim=128,
    ).to(device)
    discriminator = ProjectionDiscriminator(num_categories=num_categories).to(device)

    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=2e-4,
        betas=(0.0, 0.9),
    )
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=2e-4,
        betas=(0.0, 0.9),
    )

    sample_category_names = ("Pikachu", "Charmander", "Squirtle", "Bulbasaur")
    variations_per_category = 4
    variation_z = torch.randn(
        variations_per_category,
        generator.latent_dim,
        device=device,
    )
    fixed_z = variation_z.repeat(len(sample_category_names), 1)
    fixed_categories = torch.tensor(
        [
            data_module.category_to_id[name]
            for name in sample_category_names
            for _ in range(variations_per_category)
        ],
        dtype=torch.long,
        device=device,
    )

    print("train images:", len(data_module.train_dataset))
    print("categories:", num_categories)
    print("batches per epoch:", len(train_loader))
    print("sample rows:", sample_category_names)
    initial_parameter_counts = {
        "generator": sum(p.numel() for p in generator.parameters()),
        "discriminator": sum(p.numel() for p in discriminator.parameters()),
    }
    initial_parameter_counts["total"] = (
        initial_parameter_counts["generator"] + initial_parameter_counts["discriminator"]
    )
    print("generator parameters:", initial_parameter_counts["generator"])
    print("discriminator parameters:", initial_parameter_counts["discriminator"])

    num_epochs = args.epochs
    config = {
        "model_type": "gan",
        "batch_size": args.batch_size,
        "seed": args.seed,
        "generator_learning_rate": 2e-4,
        "discriminator_learning_rate": 2e-4,
        "latent_dim": generator.latent_dim,
        "num_categories": num_categories,
        "diversity_weight": 0.2,
        "train_images": len(data_module.train_dataset),
        "train_split": "data/pokemon-generation-one-22k/train_split.txt",
        "initial_parameter_counts": initial_parameter_counts,
    }
    fixed_inputs = {"z": fixed_z, "categories": fixed_categories}
    history = []
    start_epoch = 0
    elapsed_seconds = 0.0
    if args.resume:
        checkpoint = torch.load(run_dir / "latest.ckpt", map_location=device, weights_only=True)
        if checkpoint.get("model_type") != "gan":
            raise ValueError("Checkpoint is not a GAN run")
        saved_config = checkpoint["config"]
        for key in ("batch_size", "latent_dim", "num_categories", "diversity_weight"):
            if saved_config[key] != config[key]:
                raise ValueError(f"Cannot resume with a different {key}")
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
        discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
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
        result = train_one_epoch(
            generator,
            discriminator,
            train_loader,
            generator_optimizer,
            discriminator_optimizer,
            device,
        )
        result["epoch"] = epoch + 1
        result["epoch_seconds"] = time.perf_counter() - epoch_start
        elapsed_seconds += result["epoch_seconds"]
        result["elapsed_seconds"] = elapsed_seconds
        result["peak_vram_gb"] = (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        )
        history.append(result)

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"G={result['generator_loss']:.3f} | "
            f"D={result['discriminator_loss']:.3f} | "
            f"real={result['real_score']:.3f} | "
            f"fake={result['fake_score']:.3f}"
        )

        if (epoch + 1) % 5 == 0:
            save_samples(
                generator,
                fixed_inputs["z"],
                fixed_inputs["categories"],
                epoch + 1,
                output_dir=run_dir / "samples",
            )

        write_losses(run_dir / "losses.csv", history)
        save_checkpoint(
            run_dir / "latest.ckpt",
            generator,
            discriminator,
            generator_optimizer,
            discriminator_optimizer,
            epoch + 1,
            history,
            fixed_inputs,
            config,
        )

    metadata_path = run_dir / "run.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.setdefault("initial_parameter_counts", initial_parameter_counts)
    metadata["final_parameter_counts"] = {
        "generator": sum(p.numel() for p in generator.parameters()),
        "discriminator": sum(p.numel() for p in discriminator.parameters()),
    }
    metadata["final_parameter_counts"]["total"] = (
        metadata["final_parameter_counts"]["generator"]
        + metadata["final_parameter_counts"]["discriminator"]
    )
    metadata["completed_epochs"] = max(start_epoch, num_epochs)
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    temporary_path = metadata_path.with_name(metadata_path.name + ".tmp")
    temporary_path.write_text(json.dumps(metadata, indent=2))
    temporary_path.replace(metadata_path)


if __name__ == "__main__":
    main()
