"""Run the unmodified official evaluate.py CLI with a fixed seed and NFE audit.

This is evaluation only. Supply an exported ModelOneNFE checkpoint, not current.pt.
Outputs must be new so reference/generated images cannot be mixed across runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import runpy
import sys
import time

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=20261029)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--project-root', default=str(Path(__file__).resolve().parents[1]))
    args = p.parse_args()
    root = Path(args.project_root).resolve()
    sys.path.insert(0, str(root))
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cp = Path(args.checkpoint).resolve()
    metadata = torch.load(cp, map_location='cpu', weights_only=False)
    step = metadata.get('iteration')
    del metadata
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    from model import ModelOneNFE
    from stylegan2.models import Generator
    counts = dict(sample_batches=0, generated_images=0, generator_calls=0)
    original_sample, original_forward = ModelOneNFE.sample, Generator.forward

    def forward(self, *a, **kw):
        counts['generator_calls'] += 1
        return original_forward(self, *a, **kw)

    def sample(self, shape, **kw):
        before = counts['generator_calls']
        result = original_sample(self, shape, **kw)
        assert counts['generator_calls'] - before == 1, 'Not exactly one G call'
        assert tuple(result.shape) == tuple(shape) and torch.isfinite(result).all()
        counts['sample_batches'] += 1
        counts['generated_images'] += shape[0]
        return result

    class Tee:
        def __init__(self, original, log):
            self.original, self.log, self.parts = original, log, []
        def write(self, text):
            self.parts.append(text)
            self.original.write(text)
            return self.log.write(text)
        def flush(self):
            self.original.flush()
            self.log.flush()

    Generator.forward, ModelOneNFE.sample = forward, sample
    old_argv, old_stdout = sys.argv, sys.stdout
    start = time.time()
    try:
        with (out/'official_stdout.log').open('w') as log:
            tee = Tee(old_stdout, log)
            sys.stdout = tee
            sys.argv = ['evaluate.py', '--model_checkpoint', str(cp),
                        '--evaluate_mode', 'one_nfe', '--data_root', args.data_root,
                        '--split_dir', args.data_root, '--reference_dir', str(out/'reference'),
                        '--output_dir', str(out/'generated'), '--batch_size', str(args.batch_size),
                        '--num_workers', '2', '--device', 'cuda']
            runpy.run_path(str(root/'evaluate.py'), run_name='__main__')
    finally:
        Generator.forward, ModelOneNFE.sample = original_forward, original_sample
        sys.argv, sys.stdout = old_argv, old_stdout
    refs = list((out/'reference').glob('*.png'))
    gens = list((out/'generated/one_nfe').glob('*.png'))
    assert len(refs) == len(gens) == counts['generated_images'] == 3020
    assert {f.name for f in gens} == {f'{c:04d}_{i:02d}.png' for c in range(151) for i in range(20)}
    score = float(re.findall(r'one_nfe FID: ([0-9.]+)', ''.join(tee.parts))[-1])
    result = dict(iteration=step, fid=score, seed=args.seed, batch_size=args.batch_size,
                  reference_count=len(refs), generated_count=len(gens), nfe=1,
                  calls=counts, seconds=time.time()-start,
                  checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),
                  evaluate_sha256=hashlib.sha256((root/'evaluate.py').read_bytes()).hexdigest(),
                  protocol='Unmodified evaluate.py __main__, validation3020, no sample filtering')
    (out/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
