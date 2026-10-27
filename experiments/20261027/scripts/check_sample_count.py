"""One fixed balanced subsample of the completed validation evaluation."""
import argparse
from pathlib import Path
import json
import os
import random
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'validation3020_current135k'
OUTPUT = ROOT/'validation755_fixed_subset'


def main():
    os.environ.setdefault('OMP_NUM_THREADS','2')
    os.environ.setdefault('MKL_NUM_THREADS','2')
    sys.path.insert(0,str(PROJECT))
    import torch
    from cleanfid import fid
    from dataset import category_from_filename
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.70,0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=True
    torch.backends.cudnn.benchmark=False
    split=PROJECT/'data/pokemon-generation-one-22k'
    mapping=json.loads((split/'category_to_id.json').read_text())
    paths=[p.strip() for p in (split/'val_split.txt').read_text().splitlines() if p.strip()]
    groups={category:[] for category in mapping.values()}
    for index,path in enumerate(paths):groups[mapping[category_from_filename(path)]].append(index)
    assert len(paths)==3020 and all(len(rows)==20 for rows in groups.values())
    OUTPUT.mkdir(exist_ok=False)
    for name in ('reference','generated'):(OUTPUT/name).mkdir()
    references,generated=[],[]
    reference_rng,generation_rng=random.Random(42),random.Random(43)
    for category in sorted(groups):
        for index in sorted(reference_rng.sample(groups[category],5)):
            source=SOURCE/'reference'/f'{index:04d}.png'
            os.link(source,OUTPUT/'reference'/source.name);references.append(source.name)
        for index in sorted(generation_rng.sample(range(20),5)):
            source=SOURCE/'generated'/f'{category:04d}_{index:02d}.png'
            os.link(source,OUTPUT/'generated'/source.name);generated.append(source.name)
    assert len(references)==len(generated)==755
    os.chdir(CACHE)
    started=time.perf_counter()
    score=float(fid.compute_fid(str(OUTPUT/'generated'),str(OUTPUT/'reference'),
                               device=torch.device('cuda:0'),batch_size=8,num_workers=0))
    full=json.loads((SOURCE/'result.json').read_text())
    result=dict(FID_validation755=score,FID_same_validation3020=full['FID_validation3020'],
        difference_from_full=score-full['FID_validation3020'],reference_images=755,generated_images=755,
        reference_subset_seed=42,generated_subset_seed=43,subset_rule='one uniform random 5-of-20 subset per class',
        newly_generated_images=0,new_model_forwards=0,reference_png_names=references,
        generated_png_names=generated,seconds=time.perf_counter()-started)
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('COMPLETED',{k:v for k,v in result.items() if not k.endswith('_names')},flush=True)


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
    SOURCE = ROOT / 'validation3020_current135k'
    OUTPUT = args.output.resolve() if args.output else ROOT / 'validation755_fixed_subset'
    main()
