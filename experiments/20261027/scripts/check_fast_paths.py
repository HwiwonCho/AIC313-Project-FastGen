"""Two fixed TRAIN-only inference checks; do not change the running trainer."""
import argparse
from pathlib import Path
import gc
import json
import math
import os
import sys
import time

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'fast_paths_train755'


def main():
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('MKL_NUM_THREADS','2')
    sys.path.insert(0,str(PROJECT))
    import torch
    from cleanfid import fid
    from model import Model
    from fastgen.sampling import sample_images
    from fastgen.metrics import peer_evaluation_precision
    from fastgen.runtime import sha256
    from evaluate import generate_samples

    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    torch.cuda.set_per_process_memory_fraction(.70,device)
    checkpoint=ROOT/'current_few_nfe.ckpt'
    assert sha256(checkpoint)=='2469bf3458feb223d30de8e2d974c47a09f6153df9b582bffd1224de02c45db6'
    OUTPUT.mkdir(exist_ok=False)
    reference=ROOT/'evaluation_current/reference'
    assert len(list(reference.glob('*.png')))==755
    baseline=json.loads((ROOT/'evaluation_current/result.json').read_text())['FID_train_proxy']
    plan={'checkpoint_step':135000,'checkpoint_sha256':sha256(checkpoint),
          'reference':'fixed TRAIN755; same pixels as current training metric',
          'seed':27182,'extra_seed':20261008,'batch_size':8,'precision':'fp32',
          'candidates':[{'name':'quadratic_renoise4','grid':[1.,.5625,.25,.0625,0.],'nfe':4},
                        {'name':'euler32_teacher_only','nfe':32}],
          'baseline_train755_FID':baseline,'validation_accessed':False}
    (OUTPUT/'plan.json').write_text(json.dumps(plan,indent=2))
    backbone=Model.load_checkpoint(str(checkpoint),'few_nfe',device=device).backbone

    class View(torch.nn.Module):
        def __init__(self,kind):
            super().__init__();self.backbone=backbone;self.kind=kind
            self.initial=torch.Generator(device=device).manual_seed(27182)
            self.extra=torch.Generator(device=device).manual_seed(20261008)
        def sample(self,shape,*,device,category):
            if self.kind=='euler32_teacher_only':
                return sample_images(self.backbone,shape,category,device=device,method='euler32',
                                     generator=self.initial,precision='fp32')
            x=torch.randn(shape,device=device,generator=self.initial)
            grid=(1.,.5625,.25,.0625,0.)
            for index in range(4):
                t=torch.full((len(x),),grid[index],device=device,dtype=torch.float32)
                x=self.backbone.solution(x,t,torch.zeros_like(t),category)
                if index<3:
                    epsilon=torch.randn(shape,device=device,generator=self.extra)
                    x=(1-grid[index+1])*x+grid[index+1]*epsilon
            if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite sample')
            return x.float().clamp(-1,1)

    results={}
    labels=torch.arange(151,dtype=torch.long).repeat_interleave(5)
    os.chdir(CACHE)
    for spec in plan['candidates']:
        name=spec['name'];view=View(name).eval();calls={'batches':0,'image_forwards':0}
        def count(_module,inputs,_output):
            calls['batches']+=1;calls['image_forwards']+=len(inputs[0])
        hook=backbone.network.register_forward_hook(count)
        started=time.perf_counter()
        print('Evaluating fixed candidate',name,flush=True)
        try:
            with peer_evaluation_precision():
                generated=generate_samples(view,OUTPUT/name,labels,batch_size=8,device=device)
                assert calls=={'batches':math.ceil(755/8)*spec['nfe'],'image_forwards':755*spec['nfe']}
                score=float(fid.compute_fid(str(generated),str(reference),device=device,batch_size=8,num_workers=0))
        finally:hook.remove()
        if not math.isfinite(score):raise FloatingPointError('Nonfinite FID')
        result={'FID_train755':score,'difference_from_baseline':score-baseline,'nfe':spec['nfe'],
                'actual_calls':calls,'seconds':time.perf_counter()-started,
                'valid_four_nfe_submission':spec['nfe']<=4,'validation_accessed':False,
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(device)}
        results[name]=result
        (OUTPUT/'results.json').write_text(json.dumps(results,indent=2))
        print('RESULT',name,result,flush=True)
        del view;gc.collect();torch.cuda.empty_cache()
    print('COMPLETED',results,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--metric-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ROOT = args.work_root.resolve()
    PROJECT = args.project_root.resolve() if args.project_root else ROOT / 'project'
    CACHE = args.metric_cache.resolve()
    OUTPUT = args.output.resolve() if args.output else ROOT / 'fast_paths_train755'
    main()
