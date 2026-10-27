"""TRAIN-only image loading and deterministic duplicate-group holdout.

Protected dataset.py is unchanged. Decode/resize/normalization match its default.
No downloader, external data, or validation image access.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

SPLITS = Path(__file__).resolve().parent.parent/'data/pokemon-generation-one-22k'
DEFAULT_ROOT = SPLITS


def normalized(x):
    return x.float().div(255).sub(.5).div(.5)


class TrainData:
    def __init__(self, root=DEFAULT_ROOT, seed=20261027):
        self.root = Path(root).resolve()
        self.paths = (SPLITS/'train_split.txt').read_text().splitlines()
        mapping = json.loads((SPLITS/'category_to_id.json').read_text())
        self.labels = torch.tensor([mapping[Path(p).parent.name] for p in self.paths])
        assert len(self.paths) == 17079 and len(mapping) == 151
        resize = transforms.Resize((64,64))
        pixels, groups = [], {}
        digest = hashlib.sha256((SPLITS/'train_split.txt').read_bytes())
        for i, relative in enumerate(self.paths):
            path = (self.root/relative).resolve()
            if self.root not in path.parents:
                raise ValueError('Image path escapes dataset root')
            with Image.open(path) as im:
                a = np.array(resize(im.convert('RGB')),dtype=np.uint8,copy=True)
            raw = a.tobytes()
            key = hashlib.sha256(raw).hexdigest()
            groups.setdefault(key,[]).append(i)
            digest.update(bytes.fromhex(key))
            pixels.append(torch.from_numpy(a).permute(2,0,1))
            if (i+1)%4000 == 0:
                print('TRAIN decoded',i+1,flush=True)
        self.pixels = torch.stack(pixels)
        # Group-level stratification, including cross-label duplicates.
        target = torch.bincount(self.labels,minlength=151).double()*.2
        held = torch.zeros(151,dtype=torch.long)
        rng = np.random.RandomState(seed)
        keys = sorted(groups)
        order = rng.permutation(len(keys))
        hold, fit = [], []
        for k in order:
            indices = groups[keys[k]]
            labels = self.labels[indices]
            counts = torch.bincount(labels,minlength=151)
            involved = counts > 0
            before = ((held[involved]-target[involved])**2).sum()
            after = ((held[involved]+counts[involved]-target[involved])**2).sum()
            if after < before:
                hold.extend(indices); held += counts
            else:
                fit.extend(indices)
        self.fit = sorted(fit)
        self.holdout = sorted(hold)
        assert set(self.fit).isdisjoint(self.holdout)
        assert len(self.fit)+len(self.holdout) == len(self.paths)
        membership = set(self.holdout)
        assert all(all(i in membership for i in v) or all(i not in membership for i in v)
                   for v in groups.values())
        self.by_class = [torch.tensor([i for i in self.fit if self.labels[i] == c]) for c in range(151)]
        assert all(len(v)>0 for v in self.by_class) and held.min()>0
        digest.update(json.dumps(self.holdout).encode())
        self.identity = digest.hexdigest()
        self.summary = dict(total=len(self.paths),fit=len(fit),holdout=len(hold),
                            unique_rgb64=len(groups),holdout_min=int(held.min()),
                            holdout_max=int(held.max()),identity=self.identity)
        print('TRAIN split',self.summary,flush=True)

    def batch(self, size, rng, device):
        labels = torch.randint(151,(size,),generator=rng)
        indices = [int(self.by_class[c][torch.randint(len(self.by_class[c]),(1,),generator=rng)])
                   for c in labels.tolist()]
        return normalized(self.pixels[indices]).to(device), labels.to(device)
