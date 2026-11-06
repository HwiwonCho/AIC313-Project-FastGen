"""학습된 잠재 보정 3회 뒤 G 1회. 추론 중 D/역전파/표본 선택 없음.

연구 동기: https://ojs.aaai.org/index.php/AAAI/article/view/5933
논문의 추론 중 gradient refinement와 다르게 작은 보정망을 미리 학습한다.
이는 독립적인 실험 변형이며 논문의 공식 구현 또는 검증된 성능이 아니다.
"""
import torch
from torch import nn
from stylegan2.models import labels_to_condition

class LatentRefiner(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(151,16)
        self.net=nn.Sequential(nn.Linear(528,128),nn.SiLU(),nn.Linear(128,512))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,z,y):
        return z+0.02*torch.tanh(self.net(torch.cat([z,self.embed(y)],1)))

def few_init(self,device='cpu',**kwargs):
    from model import Model
    from stylegan2.models import Generator
    if kwargs:raise TypeError(kwargs)
    Model.__init__(self,Generator().to(device))
    self.refiner=LatentRefiner().to(device);self.control=False

@torch.no_grad()
def few_sample(self,shape,*,device='cuda',category=None,**kwargs):
    if kwargs:raise TypeError(kwargs)
    assert tuple(shape[1:])==(3,64,64) and len(category)==shape[0]
    z=torch.randn(shape[0],512,device=device)
    if not self.control:
        for _ in range(3):z=self.refiner(z,category)
    return self.backbone(z,labels_to_condition(category),noise_mode='random').clamp(-1,1)
