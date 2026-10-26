"""CPU-only checks for the extracted server training/sampling pipeline."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import copy
from pathlib import Path
import random
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from soflow_model import SoFlow64
from train import Trainer
from best4_utils import check_fixed,save
from best4_sampling import linear_renoise4
torch.set_num_threads(2)

class Data:
    def batch(self,size,rng,device):
        labels=torch.tensor([rng.randrange(2) for _ in range(size)],device=device)
        values=torch.tensor([rng.random() for _ in range(size)],device=device)
        return values[:,None,None,None].expand(-1,3,64,64).contiguous(),labels

class Checks(unittest.TestCase):
    def test_grouped16_keeps_original_four_image_draw_order(self):
        from best4_data import TrainData
        def data():
            item=TrainData.__new__(TrainData)
            item.fit={0:['a','b'],1:['c','d']}
            item.pixels=lambda name:torch.full((3,64,64),ord(name),dtype=torch.uint8)
            item.last_paths=[]
            return item
        small,large=data(),data();a=random.Random(42);b=random.Random(42)
        groups=[small.batch(4,a,'cpu') for _ in range(4)]
        x,y=large.batch(16,b,'cpu')
        self.assertTrue(torch.equal(x,torch.cat([v for v,_ in groups])))
        self.assertTrue(torch.equal(y,torch.cat([v for _,v in groups])))
        self.assertEqual(a.getstate(),b.getstate())

    def test_fixed_starter_and_training_resume(self):
        check_fixed()
        cfg={'lr':1e-4,'warmup':1,'horizon':4,'grouped16_after':1}
        def fresh():
            torch.manual_seed(42);random.seed(42)
            return Trainer(SoFlow64(num_categories=2,base_channels=8,channel_mult=[1,2],num_blocks=1),Data(),cfg,'cpu')
        original=fresh()
        for _ in range(3):original.update()
        split=fresh();split.update()
        with tempfile.TemporaryDirectory() as folder:
            cp=Path(folder)/'current.pt';save(cp,split.state())
            resumed=fresh();resumed.load(cp)
            for _ in range(2):resumed.update()
        for key in ('model','ema'):
            a=getattr(original,key).state_dict();b=getattr(resumed,key).state_dict()
            self.assertTrue(all(torch.equal(a[k],b[k]) for k in a))
        def equal(a,b):
            if isinstance(a,torch.Tensor):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
            return a==b
        self.assertTrue(equal(original.optimizer.state_dict(),resumed.optimizer.state_dict()))
        self.assertEqual(original.data_rng.getstate(),resumed.data_rng.getstate())
        self.assertEqual(original.aug_rng.getstate(),resumed.aug_rng.getstate())
        self.assertFalse(torch.cuda.is_initialized())

    def test_four_calls_times_and_private_noise(self):
        class Backbone:
            def __init__(self):self.calls=[]
            def solution(self,x,t,s,c):
                self.calls.append((float(t[0]),float(s[0])))
                return x*.1
        model=Backbone();noise=torch.ones(2,3,64,64);labels=torch.tensor([0,1])
        extra=torch.Generator().manual_seed(20261008)
        global_before=torch.get_rng_state().clone()
        output=linear_renoise4(model,noise,labels,extra)
        self.assertEqual(model.calls,[(1.,0.),(.75,0.),(.5,0.),(.25,0.)])
        self.assertTrue(torch.equal(global_before,torch.get_rng_state()))
        self.assertTrue(torch.equal(noise,torch.ones_like(noise)))
        reference=noise;reference_rng=torch.Generator().manual_seed(20261008)
        for i,t in enumerate((.75,.5,.25,0.)):
            reference=reference*.1
            if i<3:reference=(1-t)*reference+t*torch.randn(noise.shape,generator=reference_rng)
        self.assertTrue(torch.equal(output,reference.clamp(-1,1)))

if __name__=='__main__':unittest.main()
