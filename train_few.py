"""고정 GAN의 판별 신호로 작은 잠재 보정망만 학습하는 격리 파일럿."""
import csv,json,time,hashlib,traceback,subprocess,sys,argparse
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from model import ModelFewNFE,Model
from stylegan2.models import Discriminator,labels_to_condition,count_parameters
from stylegan2.train import atomic_save,atomic_json,cpu,evaluate_internal
from stylegan2.data import TrainData
P=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--generator',required=True);ap.add_argument('--discriminator-state',required=True);ap.add_argument('--data-root',required=True);ap.add_argument('--run',required=True)
a=ap.parse_args();R=Path(a.run).resolve();R.mkdir(parents=True,exist_ok=True)

def main():
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.manual_seed(42)
    m=ModelFewNFE(device='cuda').eval()
    cp=Path(a.generator)
    s=torch.load(cp,map_location='cpu',weights_only=False)['state_dict']
    m.backbone.load_state_dict({k.removeprefix('backbone.'):v for k,v in s.items()});del s
    d=Discriminator().cuda().eval();d.load_state_dict(torch.load(Path(a.discriminator_state),map_location='cpu',weights_only=False)['D'])
    m.backbone.requires_grad_(False);d.requires_grad_(False)
    opt=torch.optim.Adam(m.refiner.parameters(),lr=1e-4)
    counts=dict(G=count_parameters(m.backbone),D=count_parameters(d),R=count_parameters(m.refiner));counts['total']=sum(counts.values());assert counts['total']<100000000
    atomic_json(R/'config.json',dict(seed=42,updates=1000,batch=32,micro=8,lr=1e-4,displacement_penalty=10,per_call_bound=.02,refiner_calls=3,G_calls=1,counts=counts,G_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),frozen_D_metadata=dict(path=str(Path(a.discriminator_state).resolve()),sha256=hashlib.sha256(Path(a.discriminator_state).read_bytes()).hexdigest()),source='https://ojs.aaai.org/index.php/AAAI/article/view/5933',method='Independent amortized latent correction; not the paper implementation',training_data='Frozen G and D learned only on TRAIN; new R uses synthetic latent draws and frozen D signals; no external data',policy='Validation reporting only; fixed recipe chosen before evaluation'))
    def step():
        opt.zero_grad(set_to_none=True);av=pen=0.
        for _ in range(4):
            y=torch.randint(151,(8,),device='cuda');z0=torch.randn(8,512,device='cuda');z=z0
            for _ in range(3):z=m.refiner(z,y)
            im=m.backbone(z,labels_to_condition(y),noise_mode='random')
            adv=F.softplus(-d(im,labels_to_condition(y))).mean();reg=(z-z0).square().mean()
            ((adv+10*reg)/4).backward();av+=adv.item()/4;pen+=reg.item()/4
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.refiner.parameters())
        opt.step();assert torch.cuda.max_memory_allocated()<20000000000
        return av,pen
    # 사전검증 업데이트는 버리고 실제 초기 상태와 RNG를 복원한다.
    initial={k:v.detach().clone() for k,v in m.refiner.state_dict().items()};rng=torch.get_rng_state();crng=torch.cuda.get_rng_state_all()
    torch.cuda.reset_peak_memory_stats();step();step();calls={'G':0,'R':0}
    h=m.backbone.register_forward_hook(lambda *args:calls.__setitem__('G',calls['G']+1));j=m.refiner.register_forward_hook(lambda *args:calls.__setitem__('R',calls['R']+1))
    m.sample((4,3,64,64),device='cuda',category=torch.arange(4,device='cuda'));h.remove();j.remove();assert calls=={'G':1,'R':3}
    atomic_json(R/'preflight.json',dict(status='passed',counts=counts,full_updates=2,peak_allocated=torch.cuda.max_memory_allocated(),calls=calls))
    m.refiner.load_state_dict(initial);opt=torch.optim.Adam(m.refiner.parameters(),lr=1e-4);torch.set_rng_state(rng);torch.cuda.set_rng_state_all(crng)
    def sample(n):
        with torch.random.fork_rng(devices=[0]):
            torch.manual_seed(20261105);ys=torch.arange(8,device='cuda').repeat_interleave(8)
            im=torch.cat([m.sample((8,3,64,64),device='cuda',category=ys[k:k+8]) for k in range(0,64,8)])
            save_image((im+1)/2,R/f'samples_{n:04d}.png',nrow=8)
    sample(0);start=time.time()
    with (R/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['step','adversarial','displacement_mse','seconds','peak_allocated']);writer.writeheader()
        for n in range(1,1001):
            av,reg=step();row=dict(step=n,adversarial=av,displacement_mse=reg,seconds=time.time()-start,peak_allocated=torch.cuda.max_memory_allocated());writer.writerow(row);f.flush();atomic_json(R/'status.json',dict(status='training',**row))
            if n%250==0:
                sample(n);atomic_save(R/'refiner_state.pt',dict(step=n,R=cpu(m.refiner.state_dict()),optimizer=cpu(opt.state_dict()),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
    atomic_save(R/'few_nfe.pt',{'state_dict':cpu(m.state_dict())})
    del d,opt;torch.cuda.empty_cache()
    # 동일 TRAIN holdout 조건으로 무보정/보정을 비교하되 계수를 사후 탐색하지 않는다.
    data=TrainData(a.data_root);scores={}
    for control in [True,False]:
        m.control=control;dest=R/('train_control' if control else 'train_four');dest.mkdir(exist_ok=True)
        scores[str(control)]=evaluate_internal(SimpleNamespace(ema=m,device='cuda',images_seen=0),data,dest)
    atomic_json(R/'train_scores.json',scores);del m,data;torch.cuda.empty_cache()
    # 저장된 few_nfe.pt에 대해 공식 evaluate.py를 별도 실행한다.
    atomic_json(R/'result.json',dict(status='completed',updates=1000,seconds=time.time()-start,counts=counts,train=scores))

if __name__=='__main__':
    try:main()
    except BaseException:
        atomic_json(R/'failure.json',dict(traceback=traceback.format_exc(),unix=time.time()));raise
