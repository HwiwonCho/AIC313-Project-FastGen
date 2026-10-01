import math

import torch
from torch import nn
from torch.nn import functional as F


class ConditionalGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, category_dim, latent_dim):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.condition_projection = nn.Linear(
            category_dim + latent_dim,
            out_channels * 2,
        )

    def forward(self, x, category_features, z):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv_in(x)

        condition = torch.cat([category_features, z], dim=1)
        scale, shift = self.condition_projection(condition).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]

        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return (x + h) / math.sqrt(2)


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim=128, num_categories=151, category_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        self.input = nn.Linear(latent_dim + category_dim, 1024 * 4 * 4)
        self.block1 = ConditionalGeneratorBlock(1024, 512, category_dim, latent_dim)
        self.block2 = ConditionalGeneratorBlock(512, 256, category_dim, latent_dim)
        self.block3 = ConditionalGeneratorBlock(256, 128, category_dim, latent_dim)
        self.block4 = ConditionalGeneratorBlock(128, 64, category_dim, latent_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, category):
        category_features = self.category_embedding(category)
        h = self.input(torch.cat([z, category_features], dim=1))
        h = h.view(z.shape[0], 1024, 4, 4)
        h = self.block1(h, category_features, z)
        h = self.block2(h, category_features, z)
        h = self.block3(h, category_features, z)
        h = self.block4(h, category_features, z)
        return self.output(h)


def spectral_conv(in_channels, out_channels, kernel_size=4, stride=2, padding=1):
    return nn.utils.spectral_norm(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    )


class ProjectionDiscriminator(nn.Module):
    def __init__(self, num_categories=151):
        super().__init__()
        self.features = nn.Sequential(
            spectral_conv(3, 96),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(96, 192),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(192, 384),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_conv(384, 768),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.unconditional = nn.utils.spectral_norm(nn.Linear(768, 1))
        self.category_embedding = nn.Embedding(num_categories, 768)

    def forward(self, images, category):
        features = self.features(images)
        features = features.sum(dim=(2, 3))
        unconditional_score = self.unconditional(features).squeeze(1)
        category_score = (
            features * self.category_embedding(category)
        ).sum(dim=1) / math.sqrt(features.shape[1])
        return unconditional_score + category_score
