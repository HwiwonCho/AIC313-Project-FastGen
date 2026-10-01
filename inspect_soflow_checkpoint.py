"""Render raw-model and EMA samples from the latest SoFlow checkpoint."""

import json
from pathlib import Path

import torch
from torchvision.utils import save_image

from soflow_model import SoFlow64


RUN_DIR = Path("runs/soflow")
CHECKPOINT_PATH = RUN_DIR / "latest.ckpt"
OUTPUT_DIR = RUN_DIR / "diagnostics"
MAPPING_PATH = Path("data/pokemon-generation-one-22k/category_to_id.json")


@torch.inference_mode()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with MAPPING_PATH.open() as file:
        category_to_id = json.load(file)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    epoch = int(checkpoint["epoch"])
    names = ("Pikachu", "Charmander", "Squirtle", "Bulbasaur")
    variations = 4
    categories = torch.tensor(
        [category_to_id[name] for name in names for _ in range(variations)],
        device=device,
    )
    generator = torch.Generator(device=device).manual_seed(20260922)
    noise = torch.randn(
        len(categories), 3, 64, 64, device=device, generator=generator
    )

    model = SoFlow64(num_categories=len(category_to_id)).to(device).eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, state_key in (("raw", "model"), ("ema", "ema_model")):
        model.load_state_dict(checkpoint[state_key])
        samples = model.sample(noise.clone(), categories, nfe=1)
        path = OUTPUT_DIR / f"epoch_{epoch:03d}_{label}_nfe1.png"
        save_image(
            samples,
            path,
            nrow=4,
            normalize=True,
            value_range=(-1, 1),
        )
        print(path)


if __name__ == "__main__":
    main()
