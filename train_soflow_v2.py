"""Train the larger, class-balanced SoFlow v2 experiment from scratch."""

import argparse
import copy
import csv
import math
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import PokemonDataModule
from soflow_model import SoFlow64, update_ema


def sample_times(batch_size, device):
    velocity_time = torch.sigmoid(
        torch.randn(batch_size, device=device) * 1.0 - 0.2
    )
    destination_proposal = torch.sigmoid(
        torch.randn(batch_size, device=device) * 0.8 - 1.0
    )
    consistency_time = torch.sigmoid(
        torch.randn(batch_size, device=device) * 0.8 + 0.2
    ).clamp(min=1e-4)
    destination_time = torch.minimum(
        destination_proposal,
        consistency_time - 1e-4,
    )
    return velocity_time, consistency_time, destination_time


def ema_decay(step, target=0.9999):
    """Track quickly at first, then gradually approach the long-run EMA."""
    return min(target, (step + 1) / (step + 10))


def learning_rate_multiplier(step, total_steps, warmup_steps, min_ratio=0.1):
    if step < warmup_steps:
        return (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_ratio + (1 - min_ratio) * cosine


def make_balanced_sampler(dataset):
    counts = Counter(dataset.category_ids)
    weights = torch.tensor(
        [1.0 / counts[category] for category in dataset.category_ids],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
    )


@torch.inference_mode()
def save_samples(model, noise, categories, path):
    model.eval()
    samples = model.sample(noise, categories, nfe=1)
    save_image(
        samples,
        path,
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )


def write_history(history, path):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def save_checkpoint(
    path,
    model,
    ema_model,
    optimizer,
    scheduler,
    epoch,
    global_step,
    history,
):
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
        temporary_path,
    )
    temporary_path.replace(path)


def train_one_epoch(
    model,
    ema_model,
    loader,
    optimizer,
    scheduler,
    device,
    global_step,
    total_steps,
):
    model.train()
    totals = {
        "loss": 0.0,
        "velocity_loss": 0.0,
        "consistency_loss": 0.0,
        "velocity_mse": 0.0,
        "consistency_mse": 0.0,
    }
    seen = 0
    progress = tqdm(loader, desc="SoFlow v2")

    for images, categories in progress:
        images = images.to(
            device,
            non_blocking=True,
            memory_format=torch.channels_last,
        )
        categories = categories.to(device, non_blocking=True)
        batch_size = images.shape[0]
        velocity_count = max(1, int(0.75 * batch_size))
        consistency_count = batch_size - velocity_count
        velocity_time, consistency_time, destination_time = sample_times(
            batch_size,
            device,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            cfg_mask = torch.rand(velocity_count, device=device) < 0.1
            velocity_loss, velocity_mse = model.velocity_loss(
                images[:velocity_count],
                velocity_time[:velocity_count],
                categories[:velocity_count],
                cfg_mask,
            )

            time = consistency_time[velocity_count:]
            destination = destination_time[velocity_count:]
            middle = model.middle_time(
                time,
                destination,
                global_step,
                total_steps,
            )
            cfg_mask = torch.rand(consistency_count, device=device) < 0.1
            consistency_loss, consistency_mse = model.consistency_loss(
                images[velocity_count:],
                time,
                middle,
                destination,
                categories[velocity_count:],
                cfg_mask,
            )

            loss = (
                velocity_loss * velocity_count
                + consistency_loss * consistency_count
            ) / batch_size

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            10.0,
        )
        optimizer.step()
        scheduler.step()
        global_step += 1
        decay = ema_decay(global_step)
        update_ema(ema_model, model, decay=decay)

        for name, value in (
            ("loss", loss),
            ("velocity_loss", velocity_loss),
            ("consistency_loss", consistency_loss),
            ("velocity_mse", velocity_mse),
            ("consistency_mse", consistency_mse),
        ):
            totals[name] += float(value.detach()) * batch_size
        seen += batch_size
        progress.set_postfix(
            loss=f"{float(loss.detach()):.4f}",
            mse=f"{float(velocity_mse.detach()):.3f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
            ema=f"{decay:.6f}",
            grad=f"{float(gradient_norm):.2f}",
        )

    return {name: value / seen for name, value in totals.items()}, global_step


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/soflow_v2")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--base-channels", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name())
        torch.cuda.reset_peak_memory_stats()

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ]
    )
    data_module = PokemonDataModule(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        return_category=True,
        transform=transform,
    )
    sampler = make_balanced_sampler(data_module.train_dataset)
    loader = data_module.train_dataloader(sampler=sampler)
    num_categories = len(data_module.category_to_id)

    model = SoFlow64(
        num_categories=num_categories,
        base_channels=args.base_channels,
    ).to(device, memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count > 100_000_000:
        raise ValueError(f"parameter limit exceeded: {parameter_count:,}")

    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )
    total_steps = args.epochs * len(loader)
    warmup_steps = 5 * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step,
            total_steps,
            warmup_steps,
        ),
    )

    run_dir = Path(args.run_dir)
    sample_dir = run_dir / "samples"
    snapshot_dir = run_dir / "snapshots"
    checkpoint_path = run_dir / "latest.ckpt"
    loss_path = run_dir / "losses.csv"
    sample_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    history = []
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        history = checkpoint.get("history", [])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        else:
            print("WARNING: checkpoint lacks CPU RNG state; resumed sampling may differ")
        if device.type == "cuda":
            if checkpoint.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
            else:
                print("WARNING: checkpoint lacks CUDA RNG state; resumed sampling may differ")
        print(f"resumed at epoch {start_epoch}, step {global_step}")

    sample_names = ("Pikachu", "Charmander", "Squirtle", "Bulbasaur")
    variations = 4
    generator = torch.Generator().manual_seed(20260922)
    fixed_noise = torch.randn(
        len(sample_names) * variations,
        3,
        64,
        64,
        generator=generator,
    ).to(device, memory_format=torch.channels_last)
    fixed_categories = torch.tensor(
        [
            data_module.category_to_id[name]
            for name in sample_names
            for _ in range(variations)
        ],
        device=device,
    )

    print("train images:", len(data_module.train_dataset))
    print("batches per epoch:", len(loader))
    print("parameters:", f"{parameter_count:,}")
    print("sample rows:", sample_names)

    for epoch in range(start_epoch, args.epochs):
        result, global_step = train_one_epoch(
            model,
            ema_model,
            loader,
            optimizer,
            scheduler,
            device,
            global_step,
            total_steps,
        )
        peak_vram = (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda"
            else 0.0
        )
        row = {
            "epoch": epoch + 1,
            **result,
            "learning_rate": scheduler.get_last_lr()[0],
            "ema_decay": ema_decay(global_step),
            "peak_vram_gb": peak_vram,
        }
        history.append(row)
        write_history(history, loss_path)
        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"loss={result['loss']:.4f} | "
            f"velocity_mse={result['velocity_mse']:.4f} | "
            f"consistency_mse={result['consistency_mse']:.4f} | "
            f"peak_vram={peak_vram:.2f} GB"
        )

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            save_samples(
                model,
                fixed_noise,
                fixed_categories,
                sample_dir / f"epoch_{epoch + 1:03d}_raw.png",
            )
            save_samples(
                ema_model,
                fixed_noise,
                fixed_categories,
                sample_dir / f"epoch_{epoch + 1:03d}_ema.png",
            )
            save_checkpoint(
                checkpoint_path,
                model,
                ema_model,
                optimizer,
                scheduler,
                epoch + 1,
                global_step,
                history,
            )

        if (epoch + 1) % 25 == 0:
            torch.save(
                {
                    "state_dict": ema_model.state_dict(),
                    "epoch": epoch + 1,
                    "parameters": parameter_count,
                },
                snapshot_dir / f"ema_epoch_{epoch + 1:03d}.pt",
            )


if __name__ == "__main__":
    main()
