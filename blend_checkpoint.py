"""TRAIN으로 정한 비율로 G parameter만 평균한다. 버퍼와 R은 기준값 유지."""
import argparse
from pathlib import Path
import torch
from model import ModelOneNFE, ModelFewNFE


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', required=True, help='기준 G 체크포인트')
    p.add_argument('--end', required=True, help='후기 G 체크포인트')
    p.add_argument('--middle', help='Optional third G: end and middle are averaged equally before interpolation')
    p.add_argument('--early', help='Optional fourth G; requires --middle; averages end + middle + early in that order')
    p.add_argument('--few', help='4NFE일 때 고정 R을 포함한 체크포인트')
    p.add_argument('--alpha', required=True, type=float)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    if not 0 <= a.alpha <= 1:
        p.error('alpha must be between 0 and 1')
    if a.early and not a.middle:
        p.error('--early requires --middle')
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    def state(path):
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in payload:
            return payload['state_dict']
        if 'ema' in payload:
            return {'backbone.' + k: v for k, v in payload['ema'].items()}
        raise ValueError('Expected state_dict or full training checkpoint with ema')
    start, end = state(a.start), state(a.end)
    if a.middle:
        middle = state(a.middle)
        if start.keys() != end.keys() or end.keys() != middle.keys():
            raise ValueError('G checkpoint keys differ')
        if a.early:
            early = state(a.early)
            if early.keys() != end.keys():
                raise ValueError('G checkpoint keys differ')
            end = {k: (end[k] + middle[k] + early[k]) / 3 if end[k].is_floating_point() else end[k] for k in end}
        else:
            end = {k: (end[k] + middle[k]) * 0.5 if end[k].is_floating_point() else end[k] for k in end}
    base = state(a.few) if a.few else start
    model = (ModelFewNFE if a.few else ModelOneNFE)(device='cpu')
    names = {n for n, _ in model.named_parameters() if n.startswith('backbone.')}
    result = dict(base)
    for k in names:
        if start[k].shape != end[k].shape:
            raise ValueError(f'Shape mismatch: {k}')
        result[k] = (1-a.alpha)*start[k] + a.alpha*end[k]
    # G 버퍼는 언제나 start에서 가져온다. R은 few의 원본 그대로 둔다.
    for k in result:
        if k.startswith('backbone.') and k not in names:
            result[k] = start[k]
    model.load_state_dict(result, strict=True)
    if not all(torch.isfinite(v).all() for v in result.values() if v.is_floating_point()):
        raise ValueError('Nonfinite checkpoint')
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': result}, out)


if __name__ == '__main__':
    main()
