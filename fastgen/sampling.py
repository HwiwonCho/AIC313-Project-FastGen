"""Noise is t=1 and data is t=0 throughout this package.

Linear re-noising follows the SoFlow inference formula; see CITATIONS.md.
No teacher, discriminator, decoder or unconditional forward runs here.
"""
from contextlib import nullcontext

import torch


def precision_context(device, precision="bf16"):
    device = torch.device(device)
    if precision not in ("bf16", "fp32"):
        raise ValueError("Precision must be bf16 or fp32")
    if precision == "bf16" and device.type == "cuda":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable; select fp32 before starting the run")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cuda", enabled=False) if device.type == "cuda" else nullcontext()


@torch.no_grad()
def sample_images(model, shape, category, *, device, method="renoise4", noise=None,
                  generator=None, extra_generator=None, precision="fp32"):
    device = torch.device(device)
    shape = tuple(shape)
    if len(shape) != 4 or shape[1:] != (3, 64, 64) or shape[0] < 1:
        raise ValueError("Expected a nonempty [B,3,64,64] shape")
    if category is None:
        raise ValueError("Class labels are required")
    category = category.to(device=device, dtype=torch.long)
    if category.shape != (shape[0],) or not bool(((category >= 0) & (category < model.num_categories)).all()):
        raise ValueError("Invalid class labels")
    if model.training:
        raise ValueError("Sampling requires eval mode")
    if method not in ("endpoint1", "renoise4", "solution4", "euler4", "euler32"):
        raise ValueError("Unsupported fixed sampling method")
    x = (torch.randn(shape, device=device, generator=generator) if noise is None
         else noise.to(device=device, dtype=torch.float32))
    if tuple(x.shape) != shape:
        raise ValueError("Noise shape differs")
    steps = 1 if method == "endpoint1" else 32 if method == "euler32" else 4
    with precision_context(device, precision):
        for i in range(steps):
            t = torch.full((shape[0],), 1 - i / steps, device=device, dtype=torch.float32)
            next_t = 1 - (i + 1) / steps
            if method.startswith("euler"):
                x = x + (next_t - t[:, None, None, None]) * model.predict_velocity(x, t, category)
            else:
                destination = torch.full_like(t, next_t) if method == "solution4" else torch.zeros_like(t)
                x = model.solution(x, t, destination, category)
                if method == "renoise4" and i + 1 < steps:
                    eps = torch.randn(shape, device=device, generator=extra_generator)
                    x = (1 - next_t) * x + next_t * eps
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("Sampler produced nonfinite images")
    return x.float().clamp(-1, 1)
