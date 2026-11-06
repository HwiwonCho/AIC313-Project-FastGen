"""NoisyTwins 기반 1NFE 파일럿, Rangwani et al. CVPR 2023.

수식/설계 출처: https://github.com/val-iisc/NoisyTwins
공식 코드 src/utils/barlow.py, src/models/stylegan2.py, src/worker.py 확인.
차이: 균등 클래스 샘플링에 맞춰 클래스별 effective-number 스케일 대신
고정 임베딩 잡음 표준편차 0.1을 사용한다. GAN 배치와 별개인 32개 mapping
쌍의 통계로 정규화한다. projector/새 학습 파라미터/추론 추가 호출은 없다.
"""
import torch
from stylegan2.models import labels_to_condition
from torch_utils import training_stats

def attach(engine):
    def noise(module,inputs,out):
        if engine.G.mapping.training:
            return out+0.1*torch.randn_like(out)
        return out
    engine.G.mapping.embed.register_forward_hook(noise)
    def regularize():
        z=torch.randn(32,512,device=engine.device)
        c=labels_to_condition(torch.randint(151,(32,),device=engine.device))
        a=engine.G.mapping(z,c,skip_w_avg_update=True)[:,0]
        b=engine.G.mapping(z,c,skip_w_avg_update=True)[:,0]
        # 학습 파라미터가 없는 배치 표준화.
        a=(a-a.mean(0))/torch.sqrt(a.var(0,unbiased=False)+1e-5)
        b=(b-b.mean(0))/torch.sqrt(b.var(0,unbiased=False)+1e-5)
        cross=a.T@b/len(a);diag=torch.diagonal(cross)
        on=(diag-1).square().sum()
        off=cross.square().sum()-diag.square().sum()
        loss=0.01*(on+0.005*off)
        training_stats.report('Loss/G/noisy_twins',loss)
        training_stats.report('Loss/G/twins_on_diag',on)
        training_stats.report('Loss/G/twins_off_diag',off)
        loss.backward()
    engine.twins_regularize=regularize
