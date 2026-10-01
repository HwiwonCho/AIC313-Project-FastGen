"""Train the class-conditional 64x64 SoFlow experiment from scratch."""

import copy
import csv
from pathlib import Path

import torch
from torchvision.utils import save_image
from tqdm import tqdm

from dataset import PokemonDataModule
from soflow_model import SoFlow64, update_ema


RUN_DIR = Path("runs/soflow")
SAMPLE_DIR = RUN_DIR / "samples"
CHECKPOINT_PATH = RUN_DIR / "latest.ckpt"
LOSS_PATH = RUN_DIR / "losses.csv"


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


@torch.no_grad()
def save_samples(model, fixed_noise, fixed_categories, epoch):
    model.eval()
    samples = model.sample(fixed_noise, fixed_categories, nfe=1)
    save_image(
        samples,
        SAMPLE_DIR / f"epoch_{epoch:03d}.png",
        nrow=4,
        normalize=True,
        value_range=(-1, 1),
    )


def save_checkpoint(model, ema_model, optimizer, epoch, global_step, history):
    torch.save(
        {
            "model": model.state_dict(),
            "ema_model": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "history": history,
        },
        CHECKPOINT_PATH,
    )


def write_history(history):
    with LOSS_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "epoch",
                "loss",
                "velocity_loss",
                "consistency_loss",
                "velocity_mse",
                "consistency_mse",
                "peak_vram_gb",
            ),
        )
        writer.writeheader()
        writer.writerows(history)


def train_one_epoch(
    model,
    ema_model,
    loader,
    optimizer,
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
    progress = tqdm(loader, desc="SoFlow training")

    for images, categories in progress:
        images = images.to(device, non_blocking=True)
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

            if consistency_count:
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
            else:
                consistency_loss = images.new_zeros(())
                consistency_mse = images.new_zeros(())

            loss = (
                velocity_loss * velocity_count
                + consistency_loss * consistency_count
            ) / batch_size

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        update_ema(ema_model, model)
        global_step += 1

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
            velocity=f"{float(velocity_loss.detach()):.4f}",
            consistency=f"{float(consistency_loss.detach()):.4f}",
        )

    return {name: value / seen for name, value in totals.items()}, global_step


def main():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    if device.type == "cuda":
        print("gpu:", torch.cuda.get_device_name())
        torch.cuda.reset_peak_memory_stats()

    data_module = PokemonDataModule(
        batch_size=32,
        num_workers=4,
        return_category=True,
    )
    loader = data_module.train_dataloader()
    num_categories = len(data_module.category_to_id)
    model = SoFlow64(num_categories=num_categories).to(device)
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )

    num_epochs = 200
    total_steps = num_epochs * len(loader)
    start_epoch = 0
    global_step = 0
    history = []
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_PATH.exists():
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model"])
        ema_model.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        history = checkpoint.get("history", [])
        print(f"resumed at epoch {start_epoch}, step {global_step}")

    sample_names = ("Pikachu", "Charmander", "Squirtle", "Bulbasaur")
    variations = 4
    fixed_noise = torch.randn(
        len(sample_names) * variations,
        3,
        64,
        64,
        device=device,
    )
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
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("sample rows:", sample_names)

    for epoch in range(start_epoch, num_epochs):
        result, global_step = train_one_epoch(
            model,
            ema_model,
            loader,
            optimizer,
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
            "peak_vram_gb": peak_vram,
        }
        history.append(row)
        write_history(history)
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"loss={result['loss']:.4f} | "
            f"velocity={result['velocity_loss']:.4f} | "
            f"consistency={result['consistency_loss']:.4f} | "
            f"peak_vram={peak_vram:.2f} GB"
        )

        if (epoch + 1) % 5 == 0 or epoch == start_epoch:
            save_samples(ema_model, fixed_noise, fixed_categories, epoch + 1)
            save_checkpoint(
                model,
                ema_model,
                optimizer,
                epoch + 1,
                global_step,
                history,
            )


if __name__ == "__main__":
    main()
