"""Native PyTorch compatibility for pinned NVIDIA StyleGAN2-ADA.

No extension compilation. Bilinear sampling has input double-backward for R1;
augmentation coordinates are fixed (coordinate derivatives are not supported).
These replacements are installed at runtime. See SOURCES.md for the separate
NumPy compatibility edit in vendor/training/augment.py and vendor/LICENSE.txt for licensing.
"""
import sys
from pathlib import Path
import torch
from torch.nn import functional as F

VENDOR = str(Path(__file__).resolve().parent / 'vendor')
if VENDOR not in sys.path:
    sys.path.insert(0, VENDOR)


def bilinear_sample(x, grid):
    """Equivalent to grid_sample(bilinear, zeros, align_corners=False)."""
    if grid.requires_grad:
        raise ValueError('ADA grids must not require gradients')
    n, c, h, w = x.shape
    px = ((grid[..., 0] + 1) * w - 1) / 2
    py = ((grid[..., 1] + 1) * h - 1) / 2
    ix, iy = px.floor().long(), py.floor().long()
    fx, fy = px - ix, py - iy
    flat = x.flatten(2)
    out = 0
    for dx, dy, weight in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)),
                           (0, 1, (1-fx)*fy), (1, 1, fx*fy)]:
        xx, yy = ix + dx, iy + dy
        valid = (xx >= 0) & (xx < w) & (yy >= 0) & (yy < h)
        index = (yy.clamp(0, h-1)*w + xx.clamp(0, w-1)).flatten(1)
        values = flat.gather(2, index[:, None].expand(-1, c, -1))
        out = out + values * (weight*valid).flatten(1)[:, None].to(x.dtype)
    return out.reshape(n, c, grid.shape[1], grid.shape[2])


def modulated_conv2d(x, weight, styles, noise=None, up=1, down=1,
                     padding=0, resample_filter=None, demodulate=True,
                     flip_weight=True, fused_modconv=True):
    """Unfused equivalent; avoid allocating N copies of a large kernel."""
    from torch_utils.ops import conv2d_resample
    if x.dtype == torch.float16 and demodulate:
        weight = weight * ((weight.shape[1]*weight.shape[2]*weight.shape[3])**-0.5
                           / weight.norm(float('inf'), dim=[1, 2, 3], keepdim=True))
        styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)
    if demodulate:
        # Sum over kernel dimensions first: [out,in] @ [in,batch].
        coeff = (styles.float().square() @ weight.float().square().sum((2, 3)).T + 1e-8).rsqrt()
    x = x * styles.to(x.dtype)[:, :, None, None]
    x = conv2d_resample.conv2d_resample(x, weight.to(x.dtype), f=resample_filter,
                                      up=up, down=down, padding=padding,
                                      flip_weight=flip_weight)
    if demodulate:
        x = x * coeff.to(x.dtype)[:, :, None, None]
    if noise is not None:
        x = x + noise.to(x.dtype)
    return x


def install():
    from torch_utils.ops import bias_act, upfirdn2d, conv2d_gradfix, grid_sample_gradfix
    from training import networks
    if getattr(install, 'done', False):
        return
    original_bias, original_filter = bias_act.bias_act, upfirdn2d.upfirdn2d
    def bias_ref(*args, **kwargs):
        kwargs['impl'] = 'ref'
        return original_bias(*args, **kwargs)
    def filter_ref(*args, **kwargs):
        kwargs['impl'] = 'ref'
        return original_filter(*args, **kwargs)
    bias_act.bias_act = bias_ref
    upfirdn2d.upfirdn2d = filter_ref
    conv2d_gradfix.enabled = False
    grid_sample_gradfix.grid_sample = bilinear_sample
    networks.modulated_conv2d = modulated_conv2d
    install.done = True
