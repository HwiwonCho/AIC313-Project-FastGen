"""FastGen 64x64/151-class channel adaptation of pinned StyleGAN2-ADA.

Source: https://github.com/NVlabs/stylegan2-ada-pytorch (upstream.py).
Uses original blocks/forward methods; only channel tables and label API differ.
"""
import torch
from torch import nn
from contextlib import nullcontext
from . import portable_ops
portable_ops.install()
from training import networks as sg

G_CHANNELS = {4: 896, 8: 896, 16: 896, 32: 896, 64: 512}
D_CHANNELS = {4: 592, 8: 592, 16: 592, 32: 592, 64: 512}
EXPECTED_G = 64_005_784
EXPECTED_D = 35_048_352


def labels_to_condition(labels, batch=None):
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError('category must be a one-dimensional int64 tensor')
    if batch is not None and labels.shape[0] != batch:
        raise ValueError('category batch mismatch')
    if labels.device.type != 'meta' and ((labels < 0).any() or (labels >= 151).any()):
        raise ValueError('category outside [0,150]')
    return torch.nn.functional.one_hot(labels, 151).float()


class Synthesis(sg.SynthesisNetwork):
    def __init__(self, channels, mixed=True):
        nn.Module.__init__(self)
        self.w_dim, self.img_resolution, self.img_resolution_log2 = 512, 64, 6
        self.img_channels = 3
        self.block_resolutions = [4, 8, 16, 32, 64]
        self.num_ws = 0
        for r in self.block_resolutions:
            block = sg.SynthesisBlock(channels.get(r//2, 0), channels[r],
                w_dim=512, resolution=r, img_channels=3, is_last=r == 64,
                use_fp16=mixed and r >= 8, conv_clamp=256, architecture='skip')
            setattr(self, f'b{r}', block)
            self.num_ws += block.num_conv + (block.num_torgb if r == 64 else 0)


class Generator(sg.Generator):
    def __init__(self, channels=None, mixed=True):
        nn.Module.__init__(self)
        self.z_dim, self.c_dim, self.w_dim = 512, 151, 512
        self.img_resolution, self.img_channels = 64, 3
        self.synthesis = Synthesis(channels or G_CHANNELS, mixed)
        self.num_ws = self.synthesis.num_ws
        self.mapping = sg.MappingNetwork(512, 151, 512, self.num_ws, num_layers=2)

    @torch.no_grad()
    def sample(self, shape, *, device=None, category=None, seed=None):
        if len(shape) != 4 or tuple(shape[1:]) != (3, 64, 64):
            raise ValueError('shape must be (N,3,64,64)')
        actual_device = next(self.parameters()).device
        requested = torch.device(device) if device is not None else actual_device
        if requested.type != actual_device.type or (requested.index is not None and requested.index != actual_device.index):
            raise ValueError('move the generator to the requested device first')
        if category is None:
            raise ValueError('category is required')
        c = labels_to_condition(category.to(actual_device), shape[0])
        # Preserve caller RNG and training mode; one complete G call per batch.
        devices = [actual_device.index or 0] if actual_device.type == 'cuda' else []
        with (torch.random.fork_rng(devices=devices) if seed is not None else nullcontext()):
            if seed is not None:
                torch.manual_seed(seed)
            z = torch.randn(shape[0], 512, device=actual_device)
            was_training = self.training
            try:
                self.eval()
                return self(z, c, truncation_psi=1, noise_mode='random').clamp(-1, 1)
            finally:
                self.train(was_training)


class Discriminator(sg.Discriminator):
    def __init__(self, channels=None, mixed=True):
        nn.Module.__init__(self)
        channels = channels or D_CHANNELS
        self.c_dim, self.img_resolution, self.img_resolution_log2 = 151, 64, 6
        self.img_channels, self.block_resolutions = 3, [64, 32, 16, 8]
        layer = 0
        for r in self.block_resolutions:
            block = sg.DiscriminatorBlock(0 if r == 64 else channels[r], channels[r],
                channels[r//2], resolution=r, img_channels=3, first_layer_idx=layer,
                architecture='resnet', use_fp16=mixed and r >= 8, conv_clamp=256)
            setattr(self, f'b{r}', block)
            layer += block.num_layers
        self.mapping = sg.MappingNetwork(0, 151, 512, None, num_layers=2, w_avg_beta=None)
        self.b4 = sg.DiscriminatorEpilogue(channels[4], 512, 4, 3,
            architecture='resnet', mbstd_group_size=4, conv_clamp=256)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
