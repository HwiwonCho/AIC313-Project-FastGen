"""Class-conditional 64x64 adaptation of SoFlow.

The loss and Euler solution parameterization follow the official SoFlow
implementation: https://github.com/zlab-princeton/SoFlow (MIT license).
"""

import math

import torch
from torch import nn

from .soflow_unet import SongUNet


class SoFlow64(nn.Module):
    def __init__(
        self,
        num_categories=151,
        base_channels=96,
        channel_mult=(1, 2, 3, 4),
        num_blocks=2,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.network = SongUNet(
            img_resolution=64,
            in_channels=3,
            out_channels=3,
            label_dim=num_categories,
            augment_dim=0,
            model_channels=base_channels,
            channel_mult=list(channel_mult),
            channel_mult_emb=4,
            num_blocks=num_blocks,
            attn_resolutions=[16],
            dropout=0.10,
            label_dropout=0,
            embedding_type="positional",
            channel_mult_noise=2,
            encoder_type="standard",
            decoder_type="standard",
            resample_filter=[1, 3, 3, 1],
        )

    @staticmethod
    def _image_time(time):
        return time.reshape(time.shape[0], 1, 1, 1)

    def forward(self, images, timestep, *, category=None, destination_time=None):
        """Support the fixed base Model.forward API; diagonal means velocity."""
        destination = timestep if destination_time is None else destination_time
        return self.network(images, timestep, destination, category, None)

    def flow_pair(self, clean_images, time):
        noise = torch.randn_like(clean_images)
        time_image = self._image_time(time)
        noisy_images = (1 - time_image) * clean_images + time_image * noise
        velocity = noise - clean_images
        return noisy_images, velocity

    def predict_velocity(self, noisy_images, time, category=None):
        return self.network(noisy_images, time, time, category, None)

    def solution(self, noisy_images, time, destination_time, category=None):
        time_image = self._image_time(time)
        destination_image = self._image_time(destination_time)
        prediction = self.network(
            noisy_images,
            time,
            destination_time,
            category,
            None,
        )
        return noisy_images + (destination_image - time_image) * prediction

    @staticmethod
    def cfg_decay(time, decay_time=0.8, sharpness=0.025):
        if decay_time >= 1.0:
            return torch.ones_like(time)
        decay = ((1 - time) / (1 - decay_time)).clamp(0, 1 - 1e-6)
        smooth = 1 - torch.exp(-sharpness * decay / (1 - decay))
        return torch.where(time <= decay_time, torch.ones_like(time), smooth)

    def velocity_loss(
        self,
        clean_images,
        time,
        category,
        cfg_mask,
        *,
        cfg_scale=3.0,
        cfg_decay_time=0.8,
        mix_ratio=0.25,
        p=1.0,
        eps=0.01,
    ):
        batch_size = clean_images.shape[0]
        noisy_images, target_velocity = self.flow_pair(clean_images, time)
        masked_category = torch.where(cfg_mask, self.num_categories, category)
        prediction = self.predict_velocity(noisy_images, time, masked_category)

        with torch.no_grad():
            unconditional = self.predict_velocity(noisy_images, time, None)
            decay = self.cfg_decay(time, cfg_decay_time)
            guided_target = target_velocity + self._image_time(decay) * (
                cfg_scale - 1
            ) * (target_velocity - unconditional)
            guided_target = (
                mix_ratio * guided_target
                + (1 - mix_ratio) * prediction.detach()
            )
            target = torch.where(
                cfg_mask.reshape(-1, 1, 1, 1),
                target_velocity,
                guided_target,
            )

        delta = (prediction - target).reshape(batch_size, -1)
        per_image_mse = delta.square().mean(dim=1)
        weight = 1 / (per_image_mse + eps).pow(p)
        return (weight.detach() * per_image_mse).mean(), per_image_mse.mean()

    def consistency_loss(
        self,
        clean_images,
        time,
        middle_time,
        destination_time,
        category,
        cfg_mask,
        *,
        cfg_scale=3.0,
        cfg_decay_time=0.8,
        mix_ratio=0.25,
        p=1.0,
        eps=0.01,
    ):
        batch_size = clean_images.shape[0]
        noisy_images, target_velocity = self.flow_pair(clean_images, time)
        masked_category = torch.where(cfg_mask, self.num_categories, category)

        with torch.no_grad():
            unconditional = self.predict_velocity(noisy_images, time, None)
            decay = self.cfg_decay(time, cfg_decay_time)
            guided_velocity = target_velocity + self._image_time(decay) * (
                cfg_scale - 1
            ) * (target_velocity - unconditional)
            conditional = self.predict_velocity(noisy_images, time, category)
            guided_velocity = (
                mix_ratio * guided_velocity
                + (1 - mix_ratio) * conditional
            )
            transport_velocity = torch.where(
                cfg_mask.reshape(-1, 1, 1, 1),
                target_velocity,
                guided_velocity,
            )

        if noisy_images.is_cuda:
            random_state = torch.cuda.get_rng_state(noisy_images.device)
        else:
            random_state = torch.random.get_rng_state()

        prediction_at_time = self.network(
            noisy_images,
            time,
            destination_time,
            masked_category,
            None,
        )
        solution_at_time = noisy_images + self._image_time(
            destination_time - time
        ) * prediction_at_time

        with torch.no_grad():
            middle_images = noisy_images + self._image_time(
                middle_time - time
            ) * transport_velocity
            if noisy_images.is_cuda:
                torch.cuda.set_rng_state(random_state, noisy_images.device)
            else:
                torch.random.set_rng_state(random_state)
            prediction_at_middle = self.network(
                middle_images,
                middle_time,
                destination_time,
                masked_category,
                None,
            )
            solution_at_middle = middle_images + self._image_time(
                destination_time - middle_time
            ) * prediction_at_middle
            quotient = (
                (solution_at_time - solution_at_middle)
                / self._image_time(time - middle_time)
            ).reshape(batch_size, -1)
            quotient_mse = quotient.square().mean(dim=1)
            weight = 1 / (quotient_mse + eps).pow(p)

        sign = torch.sign(destination_time - time)
        proxy_loss = (
            sign
            * (
                prediction_at_time.reshape(batch_size, -1)
                * quotient
                * weight[:, None]
            ).mean(dim=1)
        ).mean()
        return proxy_loss, quotient_mse.mean()

    @staticmethod
    def middle_time(
        time,
        destination_time,
        step,
        total_steps,
        init_ratio=0.1,
        end_ratio=0.002,
    ):
        progress = min(step / max(total_steps, 1), 1.0)
        ratio = init_ratio * ((end_ratio / init_ratio) ** progress)
        middle = time + (destination_time - time) * ratio
        return torch.minimum(middle, time - 1e-4)

    @torch.no_grad()
    def sample(self, noise, category, nfe=1):
        images = noise
        batch_size = noise.shape[0]
        for index in range(nfe):
            start = torch.full(
                (batch_size,), 1 - index / nfe,
                device=noise.device, dtype=noise.dtype,
            )
            end = torch.full(
                (batch_size,), 1 - (index + 1) / nfe,
                device=noise.device, dtype=noise.dtype,
            )
            images = self.solution(images, start, end, category)
        return images.clamp(-1, 1)


def update_ema(ema_model, model, decay=0.9999):
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            ema_model.parameters(), model.parameters(), strict=True
        ):
            ema_parameter.lerp_(parameter, 1 - decay)
        for ema_buffer, buffer in zip(
            ema_model.buffers(), model.buffers(), strict=True
        ):
            ema_buffer.copy_(buffer)
