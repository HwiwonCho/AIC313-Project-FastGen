"""Finite speed/resource comparison; never saves or adopts trained probe weights."""
import argparse
import copy
import hashlib
import ctypes
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

CASES = {'plain16':(16,1,False,False), 'prefetch16':(16,1,False,False)}


def run_case(args):
    sys.path.insert(0,args.project)
    import torch
    from fastgen.engine import build_model,soflow_objective,learning_rate
    from fastgen.data import TrainBatches,read_manifest
    from fastgen.runtime import MemoryGuard,restore_rng
    from fastgen.sampling import precision_context
    from networks.soflow_model import update_ema
    torch.set_num_threads(4)
    task=json.loads(Path(args.task).read_text())
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    micro,accum,channels_last,fast=CASES[args.case]
    settings={**task['settings'],'microbatch':micro,'accumulation':accum}
    guard=MemoryGuard('cuda:0',20_000_000_000)
    model=build_model(task['architecture']).cuda().train()
    model.load_state_dict(saved['model'])
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'])
    if channels_last:
        model.to(memory_format=torch.channels_last)
        ema.to(memory_format=torch.channels_last)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],betas=(.9,.99),weight_decay=0,fused=fast)
    optimizer.load_state_dict(saved['optimizer'])
    # load_state_dict restores the source optimizer's flags; choose the measured implementation explicitly.
    for group in optimizer.param_groups:
        group['fused']=fast
        if fast:group['foreach']=False
    if fast:
        for parameter,state in optimizer.state.items():
            if 'step' in state:state['step']=state['step'].cuda()
            if parameter.ndim==4:
                for key in ('exp_avg','exp_avg_sq'):
                    if key in state:state[key]=state[key].to(memory_format=torch.channels_last)
    manifest=read_manifest(task['manifest'],task['split_dir'])
    batches=TrainBatches(task['data_root'],manifest,pilot=False,augmentation=settings['augmentation'],seed=settings['seed'],unique=settings['unique_sampling'])
    batches.load_state_dict(saved['batch_state'])
    if args.case=="prefetch16":
        from fastgen.prefetch import PrefetchBatches
        batches=PrefetchBatches(batches,micro,capacity=2)
    restore_rng(saved['rng'])
    step=saved['step']
    del saved
    ema_parameters=list(ema.parameters());parameters=list(model.parameters())
    def fingerprint(value):
        h=hashlib.sha256()
        def visit(x):
            if isinstance(x,torch.Tensor):
                t=x.detach().cpu().contiguous()
                h.update(str((str(t.dtype),list(t.shape))).encode())
                h.update(ctypes.string_at(t.data_ptr(),t.numel()*t.element_size()))
            elif isinstance(x,dict):
                for key in sorted(x,key=str):h.update(str(key).encode());visit(x[key])
            elif isinstance(x,(list,tuple)):
                for v in x:visit(v)
            else:h.update(repr(x).encode())
        visit(value)
        return h.hexdigest()
    data_digest=hashlib.sha256()
    elapsed=[];data_times=[];losses=[]
    for i in range(32):
        torch.cuda.synchronize()
        began=time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:group['lr']=learning_rate(settings,step+i)
        data_time=0
        values=[]
        for _ in range(accum):
            data_began=time.perf_counter()
            images,labels=batches.batch(micro,'cuda:0')
            if channels_last:images=images.to(memory_format=torch.channels_last)
            data_digest.update(fingerprint((images,labels)).encode())
            data_time+=time.perf_counter()-data_began
            with precision_context('cuda:0','bf16'):
                loss,row=soflow_objective(model,images,labels,step+i,settings)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite probe loss')
            (loss/accum).backward()
            values.append(float(loss.detach()))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
        optimizer.step()
        decay=min(.9999,(1+step+i)/(10+step+i))
        if fast:
            with torch.no_grad():
                torch._foreach_lerp_(ema_parameters,parameters,1-decay)
                for a,b in zip(ema.buffers(),model.buffers()):a.copy_(b)
        else:update_ema(ema,model,decay)
        torch.cuda.synchronize()
        seconds=time.perf_counter()-began
        guard.check()
        if i>=2:
            elapsed.append(seconds);data_times.append(data_time);losses.append(sum(values)/len(values))
        print(json.dumps({'case':args.case,'probe_update':i+1,'seconds':seconds,'data_seconds':data_time}),flush=True)
    if hasattr(batches,"close"):batches.close()
    from fastgen.runtime import rng_state
    proof={"model_sha256":fingerprint(model.state_dict()),"ema_sha256":fingerprint(ema.state_dict()),"optimizer_sha256":fingerprint(optimizer.state_dict()),"rng_sha256":fingerprint(rng_state()),"batch_state_sha256":fingerprint(batches.state_dict()),"batch_images_labels_sha256":data_digest.hexdigest()}
    result={"proof":proof,'case':args.case,'microbatch':micro,'accumulation':accum,'effective_batch':micro*accum,
            'channels_last':channels_last,'fused_adamw_and_foreach_ema':fast,'timed_updates':len(elapsed),
            'median_seconds':statistics.median(elapsed),'mean_seconds':statistics.mean(elapsed),
            'mean_data_seconds':statistics.mean(data_times),'finite_losses':losses,
            'memory':guard.check(),'probe_weights_discarded':True,'quality_claim':False,
            'source_step':step,'estimated_200k_update_hours':statistics.median(elapsed)*200000/3600}
    target=Path(args.output)/(args.case+'.json')
    target.write_text(json.dumps(result,indent=2))
    print('RESULT',json.dumps(result),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['project','task','checkpoint','output']:parser.add_argument('--'+key,required=True)
    parser.add_argument('--case',choices=CASES)
    args=parser.parse_args()
    if args.case:return run_case(args)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    results=[]
    for case in CASES:
        command=[sys.executable,'-u','-B',str(Path(__file__).resolve()),'--project',args.project,'--task',args.task,'--checkpoint',args.checkpoint,'--output',args.output,'--case',case]
        with (output/(case+'.log')).open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            print('CASE_FAILED',case,result.returncode,flush=True)
        else:
            value=json.loads((output/(case+'.json')).read_text());results.append(value)
            print('CASE_DONE',json.dumps(value),flush=True)
    baseline=next((v for v in results if v['case']=='plain16'),None)
    for value in results:
        value['speedup_vs_baseline']=baseline['median_seconds']/value['median_seconds'] if baseline else None
    (output/'comparison.json').write_text(json.dumps(results,indent=2))
    print('COMPLETE',json.dumps(results),flush=True)


if __name__=='__main__':
    main()
