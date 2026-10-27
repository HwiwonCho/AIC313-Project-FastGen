"""Student-facing NFE-specific model definitions and checkpoint utilities."""


import torch
from torch import nn
from torch.nn import functional as F

from src.utils import count_parameters, parameter_summary


class Model(nn.Module):
    """DO NOT MODIFY: Common parent class for both NFE-specific models."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def count_parameters(self) -> int:
        """Return the total number of parameters, including frozen ones."""
        return count_parameters(self, trainable_only=False)

    @property
    def num_parameters(self) -> int:
        return self.count_parameters()

    def parameter_summary(self) -> dict[str, int]:
        return parameter_summary(self)

    def forward(self, x: torch.Tensor, timestep: torch.Tensor, **kwargs):
        return self.backbone(x, timestep, **kwargs)

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in the selected model class.")

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str,
        evaluate_mode: str,
        device="cpu",
        **kwargs,
    ) -> "Model":
        """Instantiate the selected model class and load its checkpoint."""
        model_classes = {
            "one_nfe": ModelOneNFE,
            "few_nfe": ModelFewNFE,
        }
        try:
            model_class = model_classes[evaluate_mode]
        except KeyError as error:
            raise ValueError(
                f"Unsupported evaluate_mode={evaluate_mode!r}; "
                "choose 'one_nfe' or 'few_nfe'."
            ) from error

        model = model_class(device=device, **kwargs)
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        return model.to(device).eval()


class ConditionalResidualBlock(nn.Module):
    """Residual block modulated by a category embedding."""

    def __init__(self, channels: int, category_dim: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.category_projection = nn.Linear(category_dim, channels * 2)
        nn.init.normal_(self.category_projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.category_projection.bias)

    def forward(self, x, category_features):
        scale, shift = self.category_projection(category_features).chunk(2, dim=1)
        scale = scale[:, :, None, None]
        shift = shift[:, :, None, None]
        h = self.norm1(x)
        h = h * (1 + scale) + shift
        h = self.conv1(F.silu(h))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + 0.1 * h


class VAE(nn.Module):
    def __init__(self, latent_dim=128, num_categories=151, category_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_categories = num_categories
        self.category_dim = category_dim
        self.category_embedding = nn.Embedding(num_categories, category_dim)
        self.prior_mean = nn.Embedding(num_categories, latent_dim)
        self.prior_logvar = nn.Embedding(num_categories, latent_dim)
        nn.init.normal_(self.prior_mean.weight, mean=0.0, std=0.5)
        nn.init.zeros_(self.prior_logvar.weight)

        self.encoder_down1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.encoder_block1 = ConditionalResidualBlock(64, category_dim)
        self.encoder_down2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.encoder_block2 = ConditionalResidualBlock(128, category_dim)
        self.encoder_down3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.encoder_block3 = ConditionalResidualBlock(256, category_dim)
        self.encoder_down4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.encoder_block4 = ConditionalResidualBlock(512, category_dim)
        self.encoder_statistics = nn.Linear(
            512 * 4 * 4 + category_dim,
            latent_dim * 2,
        )

        self.decoder_input = nn.Linear(latent_dim + category_dim, 512 * 4 * 4)
        self.decoder_block4 = ConditionalResidualBlock(512, category_dim)
        self.decoder_up3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.decoder_block3 = ConditionalResidualBlock(256, category_dim)
        self.decoder_up2 = nn.Conv2d(256, 128, 3, 1, 1)
        self.decoder_block2 = ConditionalResidualBlock(128, category_dim)
        self.decoder_up1 = nn.Conv2d(128, 64, 3, 1, 1)
        self.decoder_block1 = ConditionalResidualBlock(64, category_dim)
        self.decoder_up0 = nn.Conv2d(64, 32, 3, 1, 1)
        self.decoder_block0 = ConditionalResidualBlock(32, category_dim)
        self.decoder_output = nn.Sequential(
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def encode(self, x, category):
        category_features = self.category_embedding(category)
        h = self.encoder_block1(self.encoder_down1(x), category_features)
        h = self.encoder_block2(self.encoder_down2(h), category_features)
        h = self.encoder_block3(self.encoder_down3(h), category_features)
        h = self.encoder_block4(self.encoder_down4(h), category_features)
        h = h.flatten(start_dim=1)
        statistics = self.encoder_statistics(
            torch.cat([h, category_features], dim=1)
        )
        mean, raw_logvar = statistics.chunk(2, dim=1)
        logvar = raw_logvar.clamp(min=-8.0, max=4.0)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def prior(self, category):
        mean = self.prior_mean(category)
        logvar = self.prior_logvar(category).clamp(min=-4.0, max=4.0)
        return mean, logvar

    def decode(self, z, category):
        category_features = self.category_embedding(category)
        h = self.decoder_input(torch.cat([z, category_features], dim=1))
        h = h.view(z.shape[0], 512, 4, 4)
        h = self.decoder_block4(h, category_features)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder_block3(self.decoder_up3(h), category_features)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder_block2(self.decoder_up2(h), category_features)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder_block1(self.decoder_up1(h), category_features)
        h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.decoder_block0(self.decoder_up0(h), category_features)
        return self.decoder_output(h)

    def forward(self, x, category):
        mean, logvar = self.encode(x, category)
        z = self.reparameterize(mean, logvar)
        return self.decode(z, category), mean, logvar



class ModelOneNFE(Model):
    """Student model evaluated with exactly one model function evaluation."""

    def __init__(self, device="cpu", **kwargs):
        from stylegan2.models import Generator
        if kwargs:
            raise TypeError(f"Unexpected model arguments: {sorted(kwargs)}")
        super().__init__(Generator().to(device))
        if self.num_parameters > 100_000_000:
            raise ValueError("One-NFE framework exceeds 100M parameters")

    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        nfe = kwargs.pop("nfe", 1)
        if nfe != 1:
            raise ValueError("ModelOneNFE supports exactly one NFE")
        seed = kwargs.pop("seed", None)
        if kwargs:
            raise TypeError(f"Unexpected sampling arguments: {sorted(kwargs)}")
        return self.backbone.sample(shape, device=device, category=category, seed=seed)


class ModelFewNFE(Model):
    """Student model evaluated with no more than four model evaluations."""

    def __init__(self, device="cpu", **kwargs):
        raise NotImplementedError("Implement the few-NFE model constructor.")
    def sample(
        self,
        shape: tuple[int, ...],
        *,
        device: torch.device | str = "cuda",
        category: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("Implement sample() in ModelFewNFE.")
