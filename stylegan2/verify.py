"""Bounded synthetic verification. No dataset reads, checkpoints, logs, or FID.

Run python -B -m stylegan2.verify for CPU checks; --gpu replays two full-phase scratch updates.
GPU --micro/--batch control only this finite verification, not the recipe default.
"""
import argparse
import ast
import gc
from pathlib import Path
import time
import torch
from . import portable_ops
from training import networks as upstream
REFERENCE_MODCONV = upstream.modulated_conv2d
from . import models
from .recipe import Config, UpdateEngine, make_augment

ROOT = Path(__file__).resolve().parent


def assert_finite(x):
    assert torch.isfinite(x).all()


def skeleton():
    import hashlib, json
    reference = json.loads((ROOT/'skeleton.json').read_text())
    project = ROOT.parent
    for name, expected in reference['files'].items():
        actual = hashlib.sha256((project/name).read_text(encoding='utf-8-sig').encode()).hexdigest()
        assert actual == expected, name
    tree = ast.parse((project/'model.py').read_text(encoding='utf-8-sig'))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in reference['classes']:
            assert hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == reference['classes'][node.name]
    print('PASS skeleton: fixed content and Model/ModelFewNFE AST unchanged', flush=True)


def operators():
    torch.manual_seed(12)
    x = torch.randn(2, 3, 7, 9, dtype=torch.double, requires_grad=True)
    grid = torch.rand(2, 5, 8, 2, dtype=torch.double)*2.8-1.4
    expected = torch.nn.functional.grid_sample(x, grid, align_corners=False)
    actual = portable_ops.bilinear_sample(x, grid)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    v = torch.randn_like(actual)
    a = torch.autograd.grad((actual*v).sum(), x, retain_graph=True)[0]
    b = torch.autograd.grad((expected*v).sum(), x)[0]
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    assert torch.autograd.gradgradcheck(lambda t: portable_ops.bilinear_sample(t, grid),
                                      (x,), fast_mode=True)
    for up in [1, 2]:
        xx = torch.randn(2, 4, 8, 8, requires_grad=True)
        w = torch.randn(6, 4, 3, 3, requires_grad=True)
        s = torch.randn(2, 4, requires_grad=True)
        from torch_utils.ops import upfirdn2d
        f = upfirdn2d.setup_filter([1, 3, 3, 1])
        kw = dict(up=up, padding=1, resample_filter=f, fused_modconv=False)
        a = portable_ops.modulated_conv2d(xx, w, s, **kw)
        b = REFERENCE_MODCONV(xx, w, s, **kw)
        torch.testing.assert_close(a, b, rtol=3e-5, atol=3e-6)
        ga = torch.autograd.grad(a.square().mean(), (xx,w,s), create_graph=True)
        gb = torch.autograd.grad(b.square().mean(), (xx,w,s), create_graph=True)
        for aa, bb in zip(ga, gb):
            torch.testing.assert_close(aa, bb, rtol=2e-4, atol=2e-6)
        ha = torch.autograd.grad(sum(t.square().sum() for t in ga), (xx,w,s))
        hb = torch.autograd.grad(sum(t.square().sum() for t in gb), (xx,w,s))
        for aa, bb in zip(ha, hb):
            torch.testing.assert_close(aa, bb, rtol=5e-4, atol=3e-6)
    print('PASS operators: bilinear forward/gradient/double-backward; modconv forward/first/second derivatives', flush=True)


def augmentation():
    aug = make_augment()
    x = torch.randn(2, 3, 64, 64).clamp(-1, 1).requires_grad_()
    for p in [0., .5, 1.]:
        aug.p.fill_(p)
        state = torch.get_rng_state()
        y = aug(x)
        # Compare the complete unchanged ADA pipeline against native grid_sample.
        from torch_utils.ops import grid_sample_gradfix
        torch.set_rng_state(state)
        grid_sample_gradfix.grid_sample = lambda a,b: torch.nn.functional.grid_sample(a,b,align_corners=False)
        try:
            reference = aug(x)
        finally:
            grid_sample_gradfix.grid_sample = portable_ops.bilinear_sample
        torch.testing.assert_close(y, reference, rtol=2e-4, atol=2e-5)
        assert y.shape == x.shape
        assert_finite(y)
        g = torch.autograd.grad(y.square().mean(), x, create_graph=True)[0]
        assert_finite(torch.autograd.grad(g.square().sum(), x)[0])
    # Disable colors to isolate geometric constant-fill artifacts. A constant
    # gray source should stay gray under interior/reflect geometric sampling.
    for name in ['brightness','contrast','lumaflip','hue','saturation']:
        setattr(aug, name, 0)
    for seed in range(4):
        torch.manual_seed(seed)
        y = aug(torch.full((2,3,64,64), .2))
        assert (y-.2).abs().max() < 2e-4, float((y-.2).abs().max())
    print('PASS official ADA bgc finite/double-backward and geometry-only constant-border probes', flush=True)


def counts():
    # Construct real modules on CPU, then release; no learned weights saved.
    g = models.Generator(mixed=False)
    assert models.count_parameters(g) == models.EXPECTED_G
    print('G parameters:', models.count_parameters(g), flush=True)
    del g; gc.collect()
    d = models.Discriminator(mixed=False)
    assert models.count_parameters(d) == models.EXPECTED_D
    print('D parameters:', models.count_parameters(d), flush=True)
    del d; gc.collect()


def tiny_update():
    channels = {4:32,8:32,16:32,32:16,64:8}
    g = models.Generator(channels, mixed=False)
    d = models.Discriminator(channels, mixed=False)
    engine = UpdateEngine(g,d,Config(batch=4,micro=4))
    engine.augment.p.fill_(.5)
    before = g.mapping.fc0.weight.detach().clone()
    result = engine.step(torch.rand(4,3,64,64)*2-1, torch.tensor([0,1,2,150]))
    assert not torch.equal(before, g.mapping.fc0.weight)
    assert engine.g_opt.state and engine.d_opt.state
    assert engine.images_seen == 4
    c = torch.tensor([0,150])
    assert_finite(engine.ema.sample((2,3,64,64),category=c,device='cpu',seed=9))
    state = torch.get_rng_state().clone()
    a = g.sample((2,3,64,64),category=c,device='cpu',seed=9)
    b = g.sample((2,3,64,64),category=c,device='cpu',seed=9)
    assert torch.equal(state,torch.get_rng_state())
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert_finite(a)
    # Unseeded requests must advance RNG, not repeat a batch indefinitely.
    a = g.sample((2,3,64,64),category=c,device='cpu')
    b = g.sample((2,3,64,64),category=c,device='cpu')
    assert not torch.equal(a,b)
    import io
    from unittest.mock import patch
    from model import Model, ModelOneNFE
    # Exercise the preserved skeleton loader in memory using a small fixture.
    factory = models.Generator
    with patch('stylegan2.models.Generator', lambda: factory(channels,mixed=False)):
        wrapper = ModelOneNFE(device='cpu')
        buf = io.BytesIO()
        torch.save({'state_dict':wrapper.state_dict()},buf)
        buf.seek(0)
        restored = Model.load_checkpoint(buf,'one_nfe',device='cpu')
        assert wrapper.count_parameters() == restored.count_parameters()
        calls = []
        handle = restored.backbone.register_forward_hook(lambda *args: calls.append(1))
        output = restored.sample((2,3,64,64),device='cpu',category=c,seed=19)
        handle.remove()
        assert len(calls) == 1 and output.shape == (2,3,64,64)
        assert_finite(output)
        assert output.min() >= -1 and output.max() <= 1
    print('PASS reduced-width smoke: Gmain/Greg/Dmain/Dreg, Adam, EMA, seeded sampling', result, flush=True)


def gpu(batch, micro, ada_p, fast_regularization=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Use the existing CUDA-enabled FastGen environment')
    free,total = torch.cuda.mem_get_info()
    if free < 10_000_000_000:
        raise RuntimeError('Insufficient idle GPU memory; do not interfere with other work')
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(23)
    start = time.perf_counter()
    g = models.Generator().cuda()
    d = models.Discriminator().cuda()
    engine = UpdateEngine(g,d,Config(batch=batch,micro=micro,fast_regularization=fast_regularization))
    engine.augment.p.fill_(ada_p)
    torch.cuda.reset_peak_memory_stats()
    print('GPU full-size synthetic probe starting', batch, micro, flush=True)
    for repeat in range(2):
        # Replay a regularization boundary with Adam states resident on pass 2.
        # This is explicitly a synthetic stress probe, not a training schedule.
        engine.iteration = 0
        r = engine.step(torch.rand(batch,3,64,64,device='cuda')*2-1,
                        torch.arange(batch,device='cuda')%151)
        print('completed synthetic full-phase pass',repeat+1,flush=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    assert peak < 20_000_000_000, peak
    print({'result':r, 'peak_allocated_bytes':peak,
           'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
           'seconds_including_construction':time.perf_counter()-start,
           'device':torch.cuda.get_device_name(),
           'scope':'two synthetic full-phase passes; NOT a full-training peak certificate'}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--micro', type=int, default=4)
    p.add_argument('--ada-p', type=float, default=1.)
    p.add_argument('--fast-regularization',action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.gpu:
        gpu(args.batch,args.micro,args.ada_p,args.fast_regularization)
    else:
        skeleton(); operators(); augmentation(); counts(); tiny_update()
