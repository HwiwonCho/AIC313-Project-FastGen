import torch
from torchvision.transforms.functional import affine
from torchvision.transforms import InterpolationMode


def augmentation(images, mode="weak"):
    if mode == "none":
        return images
    brightness = torch.rand((len(images), 1, 1, 1), device=images.device) * 0.2 - 0.1
    dx, dy = torch.randint(-4, 5, (2,), device="cpu").tolist()
    padded = torch.nn.functional.pad(images + brightness, (4, 4, 4, 4), value=1)
    output = padded[:, :, 4 + dy:68 + dy, 4 + dx:68 + dx]
    if mode == "cutout":
        y, x = torch.randint(0, 48, (2,), device="cpu").tolist()
        mask = torch.ones((1, 1, 64, 64), device=images.device)
        mask[:, :, y:y + 16, x:x + 16] = 0
        output = output * mask + (1 - mask)
    return output


def image_augmentation(images, mode, rng):
    if mode == "none":
        return images
    images = torch.stack([image.flip(-1) if rng.random() < 0.5 else image for image in images])
    if mode == "spatial":
        images = torch.stack([affine(image, angle=rng.uniform(-5, 5), translate=[rng.randint(-3, 3), rng.randint(-3, 3)], scale=rng.uniform(0.9, 1.05), shear=[0.0, 0.0], interpolation=InterpolationMode.BILINEAR, fill=[1.0, 1.0, 1.0]) for image in images])
    return images
