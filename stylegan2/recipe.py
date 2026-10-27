"""Single-run recipe and bounded update API. Importing never starts training.

Uses NVIDIA's loss implementation, including style mixing, R1 and PL.
No dataset access, checkpoint loading/saving, or external evaluation here.
"""
from dataclasses import dataclass
import torch
from . import models
from training.augment import AugmentPipe
from training.loss import StyleGAN2Loss
from torch_utils import training_stats


@dataclass(frozen=True)
class Config:
    batch: int = 32
    micro: int = 8
    lr: float = .0025
    r1_gamma: float = .0256
    r1_interval: int = 16
    pl_interval: int = 4
    ema_kimg: float = 10
    ada_target: float = .6
    ada_kimg: float = 500
    ada_interval: int = 4


def make_augment(device='cpu'):
    return AugmentPipe(xflip=1, rotate90=1, xint=1, scale=1, rotate=1,
        aniso=1, xfrac=1, brightness=1, contrast=1, lumaflip=1,
        hue=1, saturation=1).to(device).train().requires_grad_(False)


class UpdateEngine:
    """Explicit caller-driven updates; no background loop or persistence."""
    def __init__(self, generator, discriminator, cfg=Config()):
        if cfg.micro < 2 or cfg.batch % cfg.micro or cfg.micro % 4:
            raise ValueError('micro must be a multiple of 4 dividing batch')
        self.cfg, self.G, self.D = cfg, generator, discriminator
        self.device = next(generator.parameters()).device
        # NVIDIA persistence pickling/deepcopy reconstructs the upstream base
        # class and loses our subclass sample API. Recreate explicitly instead.
        channels = {r:getattr(generator.synthesis,f'b{r}').conv1.weight.shape[0]
                    for r in generator.synthesis.block_resolutions}
        mixed = any(getattr(generator.synthesis,f'b{r}').use_fp16 for r in channels)
        self.ema = models.Generator(channels,mixed).to(self.device)
        self.ema.load_state_dict(generator.state_dict())
        self.ema.eval().requires_grad_(False)
        self.augment = make_augment(self.device)
        self.augment.p.zero_()
        self.loss = StyleGAN2Loss(self.device, generator.mapping,
            generator.synthesis, discriminator, self.augment,
            style_mixing_prob=.9, r1_gamma=cfg.r1_gamma,
            pl_batch_shrink=2, pl_decay=.01, pl_weight=2)
        def optimizer(module, interval):
            ratio = interval/(interval+1)
            return torch.optim.Adam(module.parameters(), lr=cfg.lr*ratio,
                betas=(0.0, .99**ratio), eps=1e-8, foreach=False)
        self.g_opt = optimizer(generator, cfg.pl_interval)
        self.d_opt = optimizer(discriminator, cfg.r1_interval)
        self.iteration = self.images_seen = 0
        self.stats = training_stats.Collector(regex='Loss/signs/real')
        self.G.requires_grad_(False)
        self.D.requires_grad_(False)

    def step(self, real, category):
        cfg = self.cfg
        if real.shape != (cfg.batch, 3, 64, 64) or real.device != self.device:
            raise ValueError('real batch shape/device mismatch')
        if not torch.isfinite(real).all() or real.min() < -1 or real.max() > 1:
            raise ValueError('real images must be finite and normalized to [-1,1]')
        real_c = models.labels_to_condition(category, cfg.batch)
        phases = [('Gmain', self.G, self.g_opt, 1),
                  ('Greg', self.G, self.g_opt, cfg.pl_interval),
                  ('Dmain', self.D, self.d_opt, 1),
                  ('Dreg', self.D, self.d_opt, cfg.r1_interval)]
        for phase, module, opt, interval in phases:
            if self.iteration % interval:
                continue
            module.requires_grad_(True)
            opt.zero_grad(set_to_none=True)
            try:
                for offset in range(0, cfg.batch, cfg.micro):
                    z = torch.randn(cfg.micro, 512, device=self.device)
                    fake_y = torch.randint(151, (cfg.micro,), device=self.device)
                    self.loss.accumulate_gradients(phase,
                        real[offset:offset+cfg.micro], real_c[offset:offset+cfg.micro],
                        z, models.labels_to_condition(fake_y), sync=False,
                        gain=interval*cfg.micro/cfg.batch)
                # Fail explicitly on numerical faults; never silently sanitize NaNs.
                for p in module.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        raise FloatingPointError(f'Nonfinite gradient in {phase}')
                opt.step()
            finally:
                opt.zero_grad(set_to_none=True)
                module.requires_grad_(False)
        with torch.no_grad():
            half_life = min(cfg.ema_kimg*1000, self.images_seen*.05)
            beta = .5**(cfg.batch/max(half_life, 1e-8))
            for target, source in zip(self.ema.parameters(), self.G.parameters()):
                target.lerp_(source, 1-beta)
            for target, source in zip(self.ema.buffers(), self.G.buffers()):
                target.copy_(source)
            self.iteration += 1
            self.images_seen += cfg.batch
            if self.iteration % cfg.ada_interval == 0:
                self.stats.update()
                value = self.stats['Loss/signs/real']
                delta = (1 if value > cfg.ada_target else -1 if value < cfg.ada_target else 0)
                delta *= cfg.batch*cfg.ada_interval/(cfg.ada_kimg*1000)
                # Match upstream: p is a probability multiplier, lower bounded only.
                self.augment.p.copy_((self.augment.p + delta).clamp(min=0))
        return {'images_seen': self.images_seen, 'ada_p': float(self.augment.p)}
