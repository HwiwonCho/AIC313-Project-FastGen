"""Four endpoint calls, with the original independent re-noising stream."""
import contextlib
import torch

EXTRA_SEED = 20261008
GRID = (1., .75, .5, .25, 0.)

@contextlib.contextmanager
def sampling_precision(device):
    device = torch.device(device)
    if device.type!='cuda':
        yield
        return
    previous = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        with torch.autocast('cuda', dtype=torch.bfloat16): yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.benchmark) = previous

@torch.inference_mode()
def linear_renoise4(backbone, noise, categories, extra_generator):
    images = noise
    with sampling_precision(noise.device):
        for index in range(4):
            start = torch.full((len(noise),), GRID[index], device=noise.device, dtype=torch.float32)
            destination = torch.zeros_like(start)
            images = backbone.solution(images, start, destination, categories)
            if index < 3:
                epsilon = torch.randn(tuple(noise.shape), device=noise.device, dtype=torch.float32,
                                      generator=extra_generator)
                next_time = GRID[index+1]
                images = (1-next_time)*images + next_time*epsilon
    return images.float().clamp(-1, 1)
