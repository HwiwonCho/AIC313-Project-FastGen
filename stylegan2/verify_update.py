"""Compare every phase gradient and final parameter of one scratch update.

Default: small CPU model. --gpu: full-size mixed-precision G/D, batch/micro 4.
--frozen-phases checks gradients at equal weights without optimizer steps.
Ordinary mode checks final parameter error and reports propagated gradient error.
Deterministic CUDA verification is enabled by default, only in this process.
No data or checkpoints. Tolerances bound numerical error, not future FID.
"""
import argparse
import contextlib
import gc
import json
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from . import models, recipe, conv_gradfix
from torch_utils import training_stats


def main(gpu, native_repeat=False, deterministic=True, frozen_phases=False):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
    if gpu and (not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 10_000_000_000):
        raise RuntimeError('Idle GPU required')
    device = 'cuda' if gpu else 'cpu'
    channels = None if gpu else {4:32, 8:32, 16:32, 32:16, 64:16}
    reference = {}
    errors = {}
    def compare(key, tensors, baseline):
        values = {n:t.detach().cpu().clone() for n,t in tensors}
        if baseline:
            reference[key] = values
            return
        old = reference.pop(key)
        assert values.keys() == old.keys(), key
        delta = denom = 0.
        maximum = 0.
        for name,x in values.items():
            d = x.double()-old[name].double()
            assert torch.isfinite(x).all(), (key,name)
            delta += d.square().sum().item()
            denom += old[name].double().square().sum().item()
            maximum = max(maximum,d.abs().max().item())
        relative = (delta/max(denom,1e-30))**.5
        errors[key] = dict(relative_l2=relative, max_abs=maximum)
        print(key,errors[key],flush=True)
        limit = .005 if 'gradient' in key else .0001
        if not gpu: limit = 1e-5
        # Derivative tolerances apply to equal-weight phase comparisons.
        # After Adam updates the operands differ: report those gradients, but
        # check the final parameter difference separately instead.
        if frozen_phases or not gpu or 'gradient' not in key:
            assert relative <= limit, (key,relative,limit)

    original_context = conv_gradfix.regularization
    rng = None
    try:
        for baseline in (True,False):
            conv_gradfix.regularization = (lambda *a,**kw: contextlib.nullcontext()) if baseline or native_repeat else original_context
            torch.manual_seed(42)
            training_stats._counters.clear(); training_stats._cumulative.clear()
            g = models.Generator(channels,mixed=gpu).to(device)
            d = models.Discriminator(channels,mixed=gpu).to(device)
            e = recipe.UpdateEngine(g,d,recipe.Config(batch=4,micro=4,fast_regularization=True))
            e.augment.p.fill_(.5)
            for label,module,opt in [('G',g,e.g_opt),('D',d,e.d_opt)]:
                original_step = opt.step
                def wrap(module=module,label=label,original_step=original_step):
                    counter = [0]
                    def step(*a,**kw):
                        key=f'{label}_gradient_{counter[0]}'
                        compare(key,[(n,p.grad) for n,p in module.named_parameters() if p.grad is not None],baseline)
                        counter[0]+=1
                        if not frozen_phases:
                            return original_step(*a,**kw)
                    return step
                opt.step = wrap()
            real = torch.rand(4,3,64,64,device=device)*2-1
            labels = torch.arange(4,device=device)
            e.step(real,labels)
            for label,module in [('G',g),('D',d),('EMA',e.ema)]:
                compare(label+'_parameters',list(module.named_parameters()),baseline)
            state=[torch.get_rng_state()]+(torch.cuda.get_rng_state_all() if gpu else [])
            if baseline: rng=state
            else: assert all(torch.equal(a,b) for a,b in zip(rng,state))
            print('PASS update '+('native' if baseline else 'adapted'),flush=True)
            del e,g,d,real,labels,opt,module,original_step
            gc.collect()
            if gpu: torch.cuda.empty_cache()
    finally:
        conv_gradfix.regularization=original_context
    print(json.dumps(errors,indent=2),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',action='store_true')
    p.add_argument('--native-repeat',action='store_true')
    p.add_argument('--deterministic',action='store_true',default=True)
    p.add_argument('--nondeterministic',dest='deterministic',action='store_false')
    p.add_argument('--frozen-phases',action='store_true',help='Compare derivatives at identical weights; no optimizer steps')
    args=p.parse_args()
    main(args.gpu,args.native_repeat,args.deterministic,args.frozen_phases)
