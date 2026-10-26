"""Original train-only fit partition and grouped four-image RNG draws."""
from collections import OrderedDict
import json
from pathlib import Path, PurePosixPath
import torch
from PIL import Image
from best4_utils import ROOT, sha

SPLIT_SHA = '783d68b4a0e820c96c361435440223ce59f9eb702d712b139790ff337edbef32'
TRAIN_SHA = '65a82a52c7e88455d5b00b0fead4d6b3bf87205f31a4732eb96800e88e75d0f9'
MAPPING_SHA = 'a1e60fb8c2acacdc1f21445902ddd7b42b86ae5ed8f6d54393d1b3829ee7cc8c'

def image64(path):
    with Image.open(path) as image:
        rgba = image.convert('RGBA')
        white = Image.new('RGBA', rgba.size, (255,255,255,255))
        white.alpha_composite(rgba)
        rgb = white.convert('RGB').resize((64,64), Image.Resampling.BILINEAR)
        return torch.frombuffer(bytearray(rgb.tobytes()), dtype=torch.uint8).reshape(64,64,3).permute(2,0,1).clone()

class TrainData:
    def __init__(self, root, split_dir=None):
        self.root = Path(root).resolve(strict=True)
        split_dir = Path(split_dir or ROOT/'data/pokemon-generation-one-22k')
        if sha(split_dir/'train_split.txt')!=TRAIN_SHA or sha(split_dir/'category_to_id.json')!=MAPPING_SHA:
            raise ValueError('Official train/mapping differs')
        path = ROOT/'configs/internal_split.json'
        if sha(path)!=SPLIT_SHA: raise ValueError('Original fit/holdout split changed')
        split = json.loads(path.read_text())
        self.fit = {int(k):v for k,v in split['fit'].items()}
        self.holdout = {int(k):v for k,v in split['holdout'].items()}
        fit = [p for rows in self.fit.values() for p in rows]
        holdout = [p for rows in self.holdout.values() for p in rows]
        official = set((split_dir/'train_split.txt').read_text().splitlines())
        if len(fit)!=13667 or len(holdout)!=3412 or set(fit)&set(holdout) or set(fit)|set(holdout)!=official:
            raise ValueError('Train-only fit/holdout membership differs')
        self.cache = OrderedDict()
        self.last_paths = []
    def path(self, name):
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts: raise ValueError('Image path escapes root')
        path = self.root.joinpath(*relative.parts).resolve(strict=True)
        if self.root not in path.parents: raise ValueError('Image path escapes root')
        return path
    def pixels(self, name):
        if name not in self.cache:
            self.cache[name] = image64(self.path(name))
            if len(self.cache)>8192: self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        return self.cache[name]
    def batch(self, size, rng, device):
        if size==16:
            groups = [self.batch(4,rng,device) for _ in range(4)]
            return (torch.cat([x for x,_ in groups]).contiguous(memory_format=torch.channels_last),
                    torch.cat([y for _,y in groups]))
        if size!=4: raise ValueError('Historical batches are 4 or grouped16')
        labels = [rng.randrange(len(self.fit)) for _ in range(4)]
        paths = [rng.choice(self.fit[c]) for c in labels]
        self.last_paths = paths
        images = torch.stack([self.pixels(p) for p in paths]).to(device, memory_format=torch.channels_last)
        return images.float().div_(127.5).sub_(1), torch.tensor(labels, device=device)
