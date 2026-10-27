"""Evaluate the frozen current EMA using the submitted FewNFE interface."""
import argparse
from pathlib import Path, PurePosixPath
import hashlib
import json
import math
import os
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'project'
OUTPUT = ROOT / 'validation3020_current135k'
CACHE = ROOT.parent / 'FastGen-local-lab/evaluation_cache'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    with tarfile.open(ROOT/'validation3020_data.tar.gz') as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                raise ValueError('Unsafe archive entry: '+member.name)
        archive.extractall(ROOT)
    cfg = json.loads((ROOT/'validation3020_config.json').read_text())
    checkpoint = ROOT/'current_few_nfe.ckpt'
    assert sha(checkpoint) == cfg['export_sha256']
    assert sha(PROJECT/'evaluate.py') == cfg['evaluate_py_sha256']
    split = PROJECT/'data/pokemon-generation-one-22k'
    for name, expected in cfg['split_sha256'].items():
        assert sha(split/name) == expected, name
    assert sha(CACHE/'inception-2015-12-05.pt') == 'f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
    OUTPUT.mkdir(exist_ok=False)
    sys.path.insert(0, str(PROJECT))
    import torch
    from PIL import Image
    from cleanfid import fid
    from importlib.metadata import version
    from fastgen.project import check_official
    from dataset import PokemonDataModule
    from model import Model
    import evaluate as official
    check_official()
    torch.set_num_threads(2)
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed_all(cfg['seed'])
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(.70, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    started = time.perf_counter()

    # Match the pipeline's private white-composited validation view. The
    # original server dataset and the downloaded raw copies remain unchanged.
    view = OUTPUT/'data_view'
    paths = [p.strip() for p in (split/'val_split.txt').read_text().splitlines() if p.strip()]
    assert len(paths) == 3020
    for relative in paths:
        source = ROOT/'validation_data'/relative
        target = view/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            rgba = image.convert('RGBA')
            rgb = Image.alpha_composite(Image.new('RGBA', rgba.size, 'white'), rgba).convert('RGB')
            rgb.save(target, format='PNG')
    (view/'.transparency_composited_on_white').touch()
    model = Model.load_checkpoint(str(checkpoint), 'few_nfe', device=device)
    assert model.count_parameters() == 95_524_611
    data = PokemonDataModule(data_root=view, split_dir=split,
                             batch_size=cfg['generation_batch_size'], num_workers=0)
    print('Preparing official validation reference: 3020 images', flush=True)
    reference = official.prepare_reference_set(data, OUTPUT/'reference')
    assert len(list(reference.glob('*.png'))) == 3020
    categories = torch.arange(len(data.category_to_id), dtype=torch.long).repeat_interleave(official.SAMPLES_PER_CATEGORY)
    assert len(categories) == 3020 and official.SAMPLES_PER_CATEGORY == 20
    calls = {'batches': 0, 'image_forwards': 0}
    def count(_module, inputs, _output):
        calls['batches'] += 1
        calls['image_forwards'] += len(inputs[0])
    hook = model.backbone.network.register_forward_hook(count)
    print('Generating via unchanged submitted ModelFewNFE.sample: 3020 images', flush=True)
    try:
        generated = official.generate_samples(model, OUTPUT/'generated', categories,
            batch_size=cfg['generation_batch_size'], device=device)
    finally:
        hook.remove()
    assert len(list(generated.glob('*.png'))) == 3020
    assert calls == {'batches': math.ceil(3020/cfg['generation_batch_size'])*4, 'image_forwards':3020*4}
    print('Computing validation3020 FID', flush=True)
    os.chdir(CACHE)
    # Same clean-fid call and defaults as the provided compute_fid, with a
    # sequential image loader because its resize closure cannot be pickled
    # by Windows spawn. No library or provided source file is modified.
    score = float(fid.compute_fid(str(generated), str(reference), device=device,
                                 batch_size=cfg['generation_batch_size'], num_workers=0))
    assert math.isfinite(score)
    assert sha(checkpoint) == cfg['export_sha256']
    record = dict(FID_validation3020=score, checkpoint_step=cfg['checkpoint_step'],
        checkpoint_sha256=cfg['export_sha256'], parameters=model.count_parameters(),
        reference_images=3020, generated_images=3020, samples_per_category=20,
        actual_nfe=4, actual_calls=calls, seed=cfg['seed'], batch_size=cfg['generation_batch_size'],
        generation_precision='fp32', sampling='unchanged submitted ModelFewNFE.sample; default global noise streams',
        official_reference_and_generation_functions=True, evaluate_py_sha256=cfg['evaluate_py_sha256'],
        clean_fid_mode='clean (library default)', image_loader_workers=0,
        server_training_interrupted=False, TA_measured=False,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device), seconds=time.perf_counter()-started,
        gpu=torch.cuda.get_device_name(device),
        packages={name:version(name) for name in ('torch','torchvision','Pillow','clean-fid','numpy','scipy')})
    (OUTPUT/'result.json').write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8')
    print('COMPLETED', json.dumps(record), flush=True)


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
    OUTPUT = args.output.resolve() if args.output else ROOT / 'validation3020_current135k'
    main()
