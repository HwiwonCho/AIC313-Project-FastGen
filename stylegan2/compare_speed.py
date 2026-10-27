"""Reproducible ABBA scratch comparison against the packaged baseline.

No dataset/checkpoint access or saved weights. Git is used only by this developer
diagnostic to read the known baseline recipe, never by training or sampling.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time
import types
import torch
from . import models, recipe
from torch_utils import training_stats

BASELINE = 'eb9c864ee219c7bcec07a6f0804cced43ad2993e'


def digest(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(str((v.dtype, tuple(v.shape))).encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for x in v: visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def run(module, steps):
    gc.collect(); torch.cuda.empty_cache()
    torch.manual_seed(23)
    training_stats._counters.clear()
    training_stats._cumulative.clear()
    cfg = module.Config(fast_regularization=True) if module is recipe else module.Config()
    e = module.UpdateEngine(models.Generator().cuda(), models.Discriminator().cuda(),cfg)
    e.augment.p.fill_(.10624)
    real = torch.rand(32, 3, 64, 64, device='cuda') * 2 - 1
    labels = torch.arange(32, device='cuda') % 151
    e.step(real, labels)  # identical full-phase warmup and Adam allocation
    e.iteration = 0
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(steps): e.step(real, labels)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated()
    assert peak < 20_000_000_000
    # Hash outside timing; compare every tensor, not just scalar losses/samples.
    hashes = {name: digest(obj.state_dict()) for name, obj in
              [('G', e.G), ('D', e.D), ('EMA', e.ema), ('G_Adam', e.g_opt),
               ('D_Adam', e.d_opt), ('augment', e.augment)]}
    hashes['PL_mean'] = digest(e.loss.pl_mean)
    hashes['RNG'] = digest([torch.get_rng_state(), torch.cuda.get_rng_state_all()])
    hashes['loss_counters'] = digest(training_stats._counters)
    del e, real, labels
    return dict(seconds=seconds, seconds_per_update=seconds/steps,
                peak_allocated_bytes=peak, hashes=hashes)


def main(steps):
    if steps < 16 or steps % 16: raise ValueError('Use complete 16-step cycles')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 10_000_000_000:
        raise RuntimeError('An idle CUDA GPU is required')
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    source = subprocess.check_output(['git', 'show', BASELINE+':stylegan2/recipe.py'],
                                    cwd=Path(__file__).resolve().parent.parent).decode()
    baseline = types.ModuleType('stylegan2._baseline')
    baseline.__package__ = 'stylegan2'
    # dataclasses resolves the defining module through sys.modules.
    import sys
    sys.modules[baseline.__name__] = baseline
    exec(compile(source, '<baseline recipe>', 'exec'), baseline.__dict__)
    rows = []
    for label, module in [('baseline', baseline), ('optimized', recipe),
                          ('optimized', recipe), ('baseline', baseline)]:
        print('START '+label, flush=True)
        result = run(module, steps)
        result['variant'] = label
        rows.append(result)
        print(json.dumps(result), flush=True)
    identical = all(r['hashes'] == rows[0]['hashes'] for r in rows)
    before = sum(r['seconds'] for r in rows if r['variant']=='baseline')/2
    after = sum(r['seconds'] for r in rows if r['variant']=='optimized')/2
    print(json.dumps(dict(bitwise_identical=identical, steps_per_run=steps,
        baseline_seconds_per_update=before/steps, optimized_seconds_per_update=after/steps,
        time_reduction_percent=(1-after/before)*100, throughput_gain_percent=(before/after-1)*100,
        baseline_repeat_bitwise=rows[0]['hashes']==rows[3]['hashes'],
        optimized_repeat_bitwise=rows[1]['hashes']==rows[2]['hashes'],
        scope='ABBA random models; no TRAIN or saved checkpoints')), flush=True)
    assert all(r['hashes']['RNG']==rows[0]['hashes']['RNG'] for r in rows), 'Random stream changed'
    if not identical:
        print('NOT BITWISE IDENTICAL: run verify_conv and verify_update for numerical validation; no FID-equivalence claim.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=16)
    main(p.parse_args().steps)
