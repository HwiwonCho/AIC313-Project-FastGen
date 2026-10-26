"""Verify the real exported backbone and its four calls in a fresh process."""
import argparse
from pathlib import Path
import torch
from model import Model
from best4_utils import check_fixed, json_write

def check(checkpoint, device='cpu'):
    check_fixed();torch.set_num_threads(2)
    model=Model.load_checkpoint(str(checkpoint),'few_nfe',device=device)
    calls=[]
    hook=model.backbone.network.register_forward_hook(lambda *args:calls.append(1))
    try:
        output=model.sample((1,3,64,64),device=device,category=torch.tensor([103],device=device),
                            generator=torch.Generator(device=device).manual_seed(20261002))
    finally: hook.remove()
    if len(calls)!=4 or tuple(output.shape)!=(1,3,64,64) or not torch.isfinite(output).all() or output.abs().max()>1:
        raise ValueError('Four-NFE output verification failed')
    if model.count_parameters()!=95_524_611: raise ValueError('Parameter count differs')
    result={'official_base_loader':True,'backbone_calls_per_image':len(calls),
            'parameters':model.count_parameters(),'device':str(device),'finite_output':True,
            'complete_pair_submission':False,'one_nfe_in_this_branch':False}
    print(result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/few_nfe.ckpt'))
    p.add_argument('--device',default='cpu');p.add_argument('--record',type=Path)
    a=p.parse_args();r=check(a.checkpoint,a.device)
    if a.record:json_write(a.record,r)
