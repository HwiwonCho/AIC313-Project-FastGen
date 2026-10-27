"""Bounded synthetic timing only: no TRAIN, checkpoint, FID, or saved weights.

python -B -m stylegan2.benchmark --steps 16
CUDA synchronization at measurement boundaries adds overhead. This is not a
quality run, a server speed prediction, or a full-training memory certificate.
"""
import argparse
from collections import defaultdict
import json
import time
import torch
from .models import Generator, Discriminator
from .recipe import Config, UpdateEngine


def main(args):
    if args.steps < 16 or args.steps % 16:
        raise ValueError('Use complete 16-update lazy-regularization cycles')
    if not torch.cuda.is_available():
        raise RuntimeError('Existing CUDA-enabled environment required')
    if torch.cuda.mem_get_info()[0] < 10_000_000_000:
        raise RuntimeError('GPU must be idle; benchmark never interrupts other work')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(23)
    engine = UpdateEngine(Generator().cuda(), Discriminator().cuda(),
                          Config(batch=32, micro=args.micro))
    engine.augment.p.fill_(args.ada_p)
    real = torch.rand(32, 3, 64, 64, device='cuda') * 2 - 1
    labels = torch.arange(32, device='cuda') % 151
    # Allocate Adam state and exercise every phase before timed cycles.
    engine.step(real, labels)
    engine.iteration = 0
    totals = defaultdict(float)
    calls = defaultdict(int)

    def timed(fn, name):
        def call(*a, **kw):
            key = name or a[0]
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = fn(*a, **kw)
            torch.cuda.synchronize()
            totals[key] += time.perf_counter() - start
            calls[key] += 1
            return result
        return call

    engine.loss.accumulate_gradients = timed(engine.loss.accumulate_gradients, None)
    engine.g_opt.step = timed(engine.g_opt.step, 'G_Adam')
    engine.d_opt.step = timed(engine.d_opt.step, 'D_Adam')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.steps):
        engine.step(real, labels)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    assert peak < 20_000_000_000, peak
    print(json.dumps(dict(device=torch.cuda.get_device_name(), torch=torch.__version__,
        steps=args.steps, batch=32, micro=args.micro, initial_ada_p=args.ada_p,
        seconds=elapsed, seconds_per_update=elapsed/args.steps,
        measured_phase_seconds=dict(totals), phase_calls=dict(calls),
        other_seconds=elapsed-sum(totals.values()), peak_allocated_bytes=peak,
        scope='random scratch models; synchronized timing; no dataset or checkpoints'),
        indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--micro', type=int, default=8)
    parser.add_argument('--ada-p', type=float, default=.10624)
    main(parser.parse_args())
