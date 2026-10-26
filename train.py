"""Train the historical SoFlow recipe; select using the train-only holdout."""
import argparse
import copy
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time
import torch
from PIL import Image, ImageDraw
from soflow_model import SoFlow64, update_ema
from best4_data import TrainData
from best4_sampling import linear_renoise4, EXTRA_SEED
from best4_utils import ROOT, check_fixed, digest, MemoryGuard, rng_state, restore_rng, preserved_global_rng, save, json_write

def lr_at(config,step):
    if step<config['warmup']: multiplier=(step+1)/max(1,config['warmup'])
    else:
        p=min(1.,max(0.,(step-config['warmup'])/max(1,config['horizon']-config['warmup'])))
        multiplier=.1+.9*.5*(1+math.cos(math.pi*p))
    return config['lr']*multiplier

def objective(model,images,labels,step,horizon=170800):
    size=len(images);fm=round(size*.75);tc=size-fm;device=images.device
    cfg=torch.rand(size,device=device)<.1
    fm_times=torch.sigmoid(torch.randn(fm,device=device)-.2)
    tc_times=torch.sigmoid(torch.randn(tc,device=device)*.8+.2).clamp(1e-3,1-1e-3)
    proposal=torch.sigmoid(torch.randn(tc,device=device)*.8-1)
    destination=torch.minimum(proposal,tc_times-1e-4).clamp_min(0)
    loss_fm,mse_fm=model.velocity_loss(images[:fm],fm_times,labels[:fm],cfg[:fm],p=1.)
    middle=model.middle_time(tc_times,destination,step,horizon,end_ratio=.002)
    middle=torch.maximum(middle,destination)
    loss_tc,mse_tc=model.consistency_loss(images[fm:],tc_times,middle,destination,labels[fm:],cfg[fm:],p=1.)
    return .75*loss_fm+.25*loss_tc,{'fm_mse':float(mse_fm.detach()),'tc_mse':float(mse_tc.detach())}

class Trainer:
    def __init__(self,model,data,config,device):
        self.config,self.device,self.data=config,torch.device(device),data
        self.model=model.to(device,memory_format=torch.channels_last)
        self.ema=copy.deepcopy(self.model).eval().requires_grad_(False)
        self.optimizer=torch.optim.AdamW(self.model.parameters(),lr=config['lr'],betas=(.9,.99),weight_decay=0)
        self.data_rng,self.aug_rng=random.Random(42),random.Random(43)
        self.step,self.best,self.active_seconds=0,None,0.
    def update(self):
        started=time.perf_counter();self.model.train();self.optimizer.zero_grad(set_to_none=True)
        lr=lr_at(self.config,self.step)
        for group in self.optimizer.param_groups:group['lr']=lr
        micro,accum=(4,4) if self.step<self.config['grouped16_after'] else (16,1)
        rows=[]
        for _ in range(accum):
            images,labels=self.data.batch(micro,self.data_rng,self.device)
            images=torch.stack([x.flip(-1) if self.aug_rng.random()<.5 else x for x in images])
            with torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=='cuda'):
                loss,row=objective(self.model,images,labels,self.step,self.config['horizon'])
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
            (loss/accum).backward();rows.append({'loss':float(loss.detach()),**row})
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),10,error_if_nonfinite=True)
        self.optimizer.step();update_ema(self.ema,self.model,min(.9999,(1+self.step)/(10+self.step)))
        self.optimizer.zero_grad(set_to_none=True);self.step+=1
        seconds=time.perf_counter()-started;self.active_seconds+=seconds
        return {'step':self.step,'lr':lr,'gradient_norm':float(norm),'update_seconds':seconds,
                **{k:sum(r[k] for r in rows)/len(rows) for k in rows[0]}}
    def state(self):
        return {'format':'best4_training_v1','config':self.config,'step':self.step,'best_fid':self.best,
                'model':self.model.state_dict(),'ema':self.ema.state_dict(),'optimizer':self.optimizer.state_dict(),
                'rng':rng_state(self.data_rng,self.aug_rng),'active_seconds':self.active_seconds}
    def load(self,path):
        raw=torch.load(path,map_location='cpu',weights_only=False)
        if raw.get('format')!='best4_training_v1' or raw['config']!=self.config:
            raise ValueError('Resume needs this exact training configuration')
        self.model.load_state_dict(raw['model']);self.ema.load_state_dict(raw['ema'])
        self.optimizer.load_state_dict(raw['optimizer'])
        self.step,self.best,self.active_seconds=raw['step'],raw['best_fid'],raw['active_seconds']
        restore_rng(raw['rng'],self.data_rng,self.aug_rng)

def holdout_fid(ema,data,directory,device,batch_size=4):
    from torchvision.transforms.functional import to_pil_image
    from cleanfid import fid
    directory.mkdir(parents=True,exist_ok=False)
    reference,generated=directory/'reference',directory/'generated'
    reference.mkdir();generated.mkdir()
    labels=[]
    for category in sorted(data.holdout):
        for name in data.holdout[category]:
            pixels=data.pixels(name)
            to_pil_image(pixels).save(reference/f'{len(labels):06d}.png')
            labels.append(category)
    initial=torch.Generator(device=device).manual_seed(20261002)
    extra=torch.Generator(device=device).manual_seed(EXTRA_SEED)
    for start in range(0,len(labels),batch_size):
        cats=torch.tensor(labels[start:start+batch_size],device=device)
        noise=torch.randn((len(cats),3,64,64),device=device,generator=initial)
        images=linear_renoise4(ema,noise,cats,extra)
        for i,image in enumerate(images):to_pil_image(((image+1)/2).clamp(0,1).cpu()).save(generated/f'{start+i:06d}.png')
    score=float(fid.compute_fid(str(generated),str(reference),device=device,batch_size=4,num_workers=4,mode='clean'))
    if not math.isfinite(score):raise FloatingPointError('Nonfinite holdout FID')
    result={'FID_train_holdout3412':score,'reference_images':len(labels),'generated_images':len(labels),
            'official_validation_accessed':False,'generation_dtype':'CUDA BF16 / CPU FP32',
            'initial_seed':20261002,'extra_seed':EXTRA_SEED}
    json_write(directory/'result.json',result);return score

def loss_curve(rows,path):
    image=Image.new('RGB',(900,360),'white');draw=ImageDraw.Draw(image)
    draw.text((12,8),'SoFlow training loss',fill='black')
    if len(rows)>1:
        lo,hi=min(r['loss'] for r in rows),max(r['loss'] for r in rows)
        end=max(1,rows[-1]['step']-rows[0]['step'])
        points=[(30+840*(r['step']-rows[0]['step'])/end,320-280*(r['loss']-lo)/max(hi-lo,1e-8)) for r in rows]
        draw.line(points,fill='navy',width=2)
    image.save(path)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--run-root',type=Path,default=Path('runs/best4'))
    p.add_argument('--config',type=Path,default=ROOT/'configs/train.json');a=p.parse_args()
    process_started=time.perf_counter()
    check_fixed();cfg=json.loads(a.config.read_text());run=a.run_root.resolve();run.mkdir(parents=True,exist_ok=True)
    if cfg['selection']!='train_holdout3412' or any(cfg[k]<=0 for k in ('steps','lr','horizon','eval_every','save_every')):
        raise ValueError('Use positive training counts and train-only holdout selection')
    import fcntl
    with (run/'.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Expose one CUDA GPU')
        active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        if any(line.strip() and line.strip()!=str(os.getpid()) for line in active.splitlines()):
            raise RuntimeError('Another CUDA process is active; do not start a second trainer')
        device=torch.device('cuda:0');guard=MemoryGuard(device)
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=False
        if shutil.disk_usage(run).free<21_474_836_480:raise RuntimeError('Keep at least20GiB free for checkpoints')
        data=TrainData(a.data_root)
        model=SoFlow64(num_categories=151,base_channels=128,channel_mult=[1,2,2,4],num_blocks=3)
        if sum(x.numel() for x in model.parameters())!=95_524_611:raise ValueError('Unexpected architecture')
        trainer=Trainer(model,data,cfg,device);del model
        current,best=run/'current.pt',run/'best.pt'
        rows=[]
        if current.exists():
            trainer.load(current)
            rows=[json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()] if (run/'metrics.jsonl').exists() else []
        identity={'config':cfg,'fit_files':13667,'holdout_files':3412,'validation_used_for_selection':False,
                  'source':{str(q.relative_to(ROOT)):__import__('best4_utils').sha(q) for q in ROOT.glob('*.py')},
                  'torch':torch.__version__,'python':__import__('platform').python_version()}
        if (run/'identity.json').exists() and json.loads((run/'identity.json').read_text())!=identity:
            raise ValueError('Training source/config changed; use a new output directory')
        json_write(run/'identity.json',identity)
        stopped=False
        def stop(*_):
            nonlocal stopped
            stopped=True
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        while trainer.step<cfg['steps'] and not stopped:
            row=trainer.update();memory=guard.check()
            if trainer.step==1 or trainer.step%100==0:
                rows.append(row)
                with (run/'metrics.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
                loss_curve(rows,run/'loss_curve.png')
                json_write(run/'status.json',{'phase':'training',**row,'target':cfg['steps'],'best_holdout_fid':trainer.best,**memory})
                print(json.dumps({**row,**memory}),flush=True)
            if trainer.step==1:
                from torchvision.transforms.functional import to_pil_image
                folder=run/'early_samples';folder.mkdir(exist_ok=True)
                initial=torch.Generator(device=device).manual_seed(12345);extra=torch.Generator(device=device).manual_seed(EXTRA_SEED)
                for group in range(2):
                    labels=torch.tensor([9,14,103,129],device=device)
                    noise=torch.randn((4,3,64,64),device=device,generator=initial)
                    images=linear_renoise4(trainer.ema,noise,labels,extra)
                    for i,x in enumerate(images):to_pil_image(((x+1)/2).clamp(0,1).cpu()).save(folder/f'{4*group+i:02d}.png')
            if trainer.step%cfg['eval_every']==0 or trainer.step==cfg['steps']:
                with preserved_global_rng():
                    score=holdout_fid(trainer.ema,data,run/'evaluations'/f'{trainer.step:07d}_{time.time_ns()}',device)
                improved=trainer.best is None or score<trainer.best
                if improved:trainer.best=score
                save(current,trainer.state())
                if improved:save(best,trainer.state())
            elif trainer.step%cfg['save_every']==0:save(current,trainer.state())
        save(current,trainer.state())
        json_write(run/'status.json',{'phase':'stopped' if stopped else 'complete','step':trainer.step,
                   'best_holdout_fid':trainer.best,'training_update_seconds':trainer.active_seconds,**guard.check()})
        with (run/'sessions.jsonl').open('a') as log:
            log.write(json.dumps({'session_wall_seconds':time.perf_counter()-process_started,
                                  'step':trainer.step,'phase':'stopped' if stopped else 'complete'})+'\n')
        if stopped:raise SystemExit(75)

if __name__=='__main__':main()
