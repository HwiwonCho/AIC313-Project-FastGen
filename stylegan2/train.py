"""Explicit local TRAIN-only training, internal FID, atomic resume checkpoints.

No server access. Validation is never opened by this training program.
STOP file requests a checkpoint and clean exit after the current update.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torchvision.transforms.functional import to_pil_image
from .models import Generator, Discriminator, count_parameters
from .recipe import Config, UpdateEngine
from .data import TrainData, DEFAULT_ROOT, normalized
from torch_utils import training_stats

ROOT = Path(__file__).resolve().parent


def cpu(value):
    if isinstance(value,torch.Tensor): return value.detach().cpu().clone()
    if isinstance(value,dict): return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list): return [cpu(v) for v in value]
    if isinstance(value,tuple): return tuple(cpu(v) for v in value)
    return value


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2),encoding='utf-8')
    os.replace(temp,path)


def atomic_save(path, state):
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('wb') as f:
        torch.save(state,f); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def source_identity():
    h=hashlib.sha256((ROOT.parent/'model.py').read_text(encoding='utf-8-sig').encode())
    for p in sorted(ROOT.rglob('*.py')):
        if '__pycache__' in p.relative_to(ROOT).parts: continue
        h.update(p.relative_to(ROOT).as_posix().encode()); h.update(p.read_text(encoding='utf-8-sig').encode())
    return h.hexdigest()


def rng_state():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),
            'numpy':np.random.get_state(),'python':random.getstate()}


def restore_rng(r):
    torch.set_rng_state(r['torch']); torch.cuda.set_rng_state_all(r['cuda'])
    np.random.set_state(r['numpy']); random.setstate(r['python'])


def save_current(path,engine,sampler,data,source,best):
    state = {'G':cpu(engine.G.state_dict()),'D':cpu(engine.D.state_dict()),
        'ema':cpu(engine.ema.state_dict()),'G_opt':cpu(engine.g_opt.state_dict()),
        'D_opt':cpu(engine.d_opt.state_dict()),'augment':cpu(engine.augment.state_dict()),
        'pl_mean':cpu(engine.loss.pl_mean),'iteration':engine.iteration,
        'images_seen':engine.images_seen,'config':vars(engine.cfg),
        'rng':rng_state(),'sampler_rng':sampler.get_state(),'data_identity':data.identity,
        'source_identity':source,'best_internal_fid':best,
        'stats_counters':cpu(training_stats._counters),
        'stats_cumulative':cpu(training_stats._cumulative),
        'collector_cumulative':cpu(engine.stats._cumulative),
        'collector_moments':cpu(engine.stats._moments)}
    atomic_save(path,state)


def load_current(path,engine,sampler,data,source):
    s=torch.load(path,map_location='cpu',weights_only=False)
    if s['data_identity']!=data.identity or s['source_identity']!=source:
        raise RuntimeError('Resume refused: source or TRAIN split identity changed')
    if s['config']!=vars(engine.cfg): raise RuntimeError('Resume recipe mismatch')
    for name,module in [('G',engine.G),('D',engine.D),('ema',engine.ema),('augment',engine.augment)]:
        module.load_state_dict(s[name])
    engine.g_opt.load_state_dict(s['G_opt']); engine.d_opt.load_state_dict(s['D_opt'])
    engine.loss.pl_mean.copy_(s['pl_mean'])
    engine.iteration=s['iteration'];engine.images_seen=s['images_seen']
    training_stats._counters={name:{device:t.to(device) for device,t in counters.items()}
                             for name,counters in s['stats_counters'].items()}
    training_stats._cumulative=s['stats_cumulative']
    engine.stats._cumulative=s['collector_cumulative'];engine.stats._moments=s['collector_moments']
    sampler.set_state(s['sampler_rng']);restore_rng(s['rng'])
    return s['best_internal_fid']


@torch.no_grad()
def evaluate_internal(engine,data,run,limit=None):
    from cleanfid import fid
    ids=data.holdout
    if limit is not None:
        # Spread technical-smoke reference across labels; not a quality metric.
        ids=sorted(ids,key=lambda i:(int(data.labels[i]),i))
        ids=[ids[int(k)] for k in np.linspace(0,len(ids)-1,limit)]
    ref=run/('reference_smoke' if limit else 'reference_internal')
    generated=run/('generated_smoke' if limit else 'generated_internal')
    ref.mkdir(exist_ok=True);generated.mkdir(exist_ok=True)
    expected={f'{i:05d}.png' for i in range(len(ids))}
    if {p.name for p in generated.glob('*.png')} - expected:
        raise RuntimeError('Unexpected evaluation files; refusing mixed sample counts')
    if not any(ref.iterdir()):
        for j,i in enumerate(ids):
            to_pil_image(((normalized(data.pixels[i])+1)/2).clamp(0,1)).convert('RGB').save(ref/f'{j:05d}.png')
    if {p.name for p in ref.glob('*.png')} != expected:
        raise RuntimeError('Incomplete reference set')
    original=rng_state()
    try:
        torch.manual_seed(20261027)
        for offset in range(0,len(ids),8):
            labels=data.labels[ids[offset:offset+8]].to(engine.device)
            out=engine.ema.sample((len(labels),3,64,64),device=engine.device,category=labels)
            for j,x in enumerate(out):
                to_pil_image(((x.float().cpu()+1)/2).clamp(0,1)).convert('RGB').save(generated/f'{offset+j:05d}.png')
        # Same clean-fid mode/Inception as protected evaluate.py; num_workers=0
        # is only a Windows worker compatibility change. No custom FID network.
        # Windows glob treats *.png and *.PNG identically. clean-fid's default
        # extension list otherwise counts every exported PNG twice.
        extensions=fid.EXTENSIONS
        try:
            fid.EXTENSIONS=['png']
            score=float(fid.compute_fid(str(generated),str(ref),device=engine.device,
                                       batch_size=32,num_workers=0,mode='clean'))
        finally:
            fid.EXTENSIONS=extensions
    finally:
        restore_rng(original)
    return {'fid':score,'reference_count':len(ids),'generated_count':len(ids),
            'protocol':'TRAIN-holdout clean-fid, matched label counts, seed20261027',
            'technical_smoke_only':limit is not None,'images_seen':engine.images_seen}


@torch.no_grad()
def preview(engine,run):
    from torchvision.utils import save_image
    labels=torch.arange(8,device=engine.device).repeat_interleave(4)
    parts=[]
    for start in range(0,32,8):
        parts.append(engine.ema.sample((8,3,64,64),device=engine.device,
                    category=labels[start:start+8],seed=800+start).cpu())
    save_image(torch.cat(parts),run/'preview.png',nrow=4,normalize=True,value_range=(-1,1))


def main(args):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    lock=run/'worker.lock'
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        if (run/'current.pt').exists() and not args.resume:
            raise RuntimeError('Existing checkpoint: explicitly use --resume')
        if torch.cuda.mem_get_info()[0]<10_000_000_000:
            raise RuntimeError('GPU is not sufficiently idle')
        torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
        source=source_identity();data=TrainData(args.data_root)
        atomic_json(run/'split.json',{'summary':data.summary,'fit':data.fit,'holdout':data.holdout})
        cfg=Config(batch=32,micro=8)
        engine=UpdateEngine(Generator().cuda(),Discriminator().cuda(),cfg)
        assert count_parameters(engine.G)<=100_000_000 and count_parameters(engine.D)<=100_000_000
        sampler=torch.Generator().manual_seed(args.seed+1)
        best=float('inf')
        if args.resume:best=load_current(run/'current.pt',engine,sampler,data,source)
        atomic_json(run/'config.json',dict(recipe=vars(cfg),data=data.summary,seed=args.seed,
                      source_identity=source,target_steps=args.steps,eval_every=args.eval_every))
        print('START',{'pid':os.getpid(),'iteration':engine.iteration,'target':args.steps},flush=True)
        torch.cuda.reset_peak_memory_stats();start=time.perf_counter();start_step=engine.iteration
        next_eval=((engine.images_seen//args.eval_every)+1)*args.eval_every
        while engine.iteration<args.steps:
            if (run/'STOP').exists():break
            real,labels=data.batch(cfg.batch,sampler,engine.device)
            metrics=engine.step(real,labels)
            peak=torch.cuda.max_memory_allocated()
            if peak>=20_000_000_000:raise RuntimeError('20GB training limit exceeded')
            step=engine.iteration
            if step%10==0 or step==start_step+1:
                metrics.update(iteration=step,peak_allocated_bytes=peak,pid=os.getpid(),
                    seconds_per_update=(time.perf_counter()-start)/(step-start_step),state='training')
                atomic_json(run/'status.json',metrics);print(metrics,flush=True)
            if step%args.save_every==0:
                save_current(run/'current.pt',engine,sampler,data,source,best);preview(engine,run)
            if engine.images_seen>=next_eval:
                result=evaluate_internal(engine,data,run)
                atomic_json(run/'internal_fid.json',result);print('INTERNAL_FID',result,flush=True)
                if result['fid']<best:
                    best=result['fid']
                    atomic_save(run/'best_nfe1.pt',{'state_dict':{'backbone.'+k:cpu(v) for k,v in engine.ema.state_dict().items()},
                                'internal_evaluation':result,'source_identity':source})
                save_current(run/'current.pt',engine,sampler,data,source,best)
                next_eval+=args.eval_every
        save_current(run/'current.pt',engine,sampler,data,source,best);preview(engine,run)
        if args.smoke_eval:
            result=evaluate_internal(engine,data,run,limit=64)
            atomic_json(run/'smoke_fid.json',result);print('SMOKE_FID_NOT_QUALITY',result,flush=True)
        atomic_json(run/'status.json',dict(state='stopped' if (run/'STOP').exists() else 'completed',
                    iteration=engine.iteration,images_seen=engine.images_seen,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),pid=os.getpid()))
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    p.add_argument('--data-root',default=str(DEFAULT_ROOT))
    p.add_argument('--steps',type=int,default=125000)
    p.add_argument('--save-every',type=int,default=500)
    p.add_argument('--eval-every',type=int,default=250000,help='real-image exposures')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--smoke-eval',action='store_true')
    a=p.parse_args()
    if min(a.steps,a.save_every,a.eval_every)<=0:p.error('Intervals must be positive')
    main(a)
