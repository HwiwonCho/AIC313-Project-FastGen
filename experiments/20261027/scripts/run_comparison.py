"""Compare frozen server EMAs on one local GPU with the same TRAIN755 protocol."""
from pathlib import Path, PurePosixPath
import argparse
import gc
import hashlib
import json
import os
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT / 'project'
CACHE = ROOT.parent / 'FastGen-local-lab/evaluation_cache'


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def verify_reference_pixels(directory):
    from PIL import Image
    files = sorted((Path(directory) / 'reference').glob('*.png'))
    assert len(files) == 755
    for path in files:
        with Image.open(path) as local, Image.open(ROOT / 'server_current/reference' / path.name) as server:
            assert local.convert('RGB').tobytes() == server.convert('RGB').tobytes(), path.name


def prepare():
    for archive in (ROOT / 'evaluation_source.tar.gz',
                    ROOT / 'reference_data.tar.gz'):
        with tarfile.open(archive) as stream:
            for member in stream.getmembers():
                path = PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe archive entry: ' + member.name)
            stream.extractall(ROOT)
        print('Extracted', archive.name, flush=True)
    assert sha(ROOT / 'few_nfe.ckpt') == '81794ed4ec770a2d4fea1086cf886d228475fb6d844cf3967b865809a4b937ed'
    expected = json.loads((ROOT / 'current_export.json').read_text())['export_sha256']
    assert sha(ROOT / 'current_few_nfe.ckpt') == expected
    assert sha(CACHE / 'inception-2015-12-05.pt') == 'f58cb9b6ec323ed63459aa4fb441fe750cfe39fafad6da5cb504a16f19e958f4'
    print('Both checkpoint hashes and Inception hash match server', flush=True)


def compare():
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    sys.path.insert(0, str(PROJECT))
    sys.path.insert(0, str(ROOT / 'legacy'))
    import torch
    from importlib.metadata import version
    from fastgen.engine import build_model
    from fastgen.metrics import train_proxy, peer_evaluation_precision
    from fastgen.sampling import sample_images
    from fastgen.runtime import digest
    from soflow_model import SoFlow64 as LegacyModel
    from cleanfid import fid

    torch.set_num_threads(2)
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    expected_server = json.loads((ROOT / 'server_current_result.json').read_text())
    assert len(manifest['references']) == 755
    assert digest(manifest['references']) == expected_server['reference_groups_sha256']
    current_record = json.loads((ROOT / 'current_export.json').read_text())
    architecture = dict(num_categories=151, base_channels=128,
                        channel_mult=[1, 2, 2, 4], num_blocks=3)
    if (ROOT / 'backbone_parity.json').exists():
        parity = json.loads((ROOT / 'backbone_parity.json').read_text())
    else:
        old_state = torch.load(ROOT / 'few_nfe.ckpt', map_location='cpu', weights_only=True, mmap=True)['state_dict']
        old_state = {key.removeprefix('backbone.'): value for key, value in old_state.items()}
        old = LegacyModel(**architecture).eval().requires_grad_(False)
        new = build_model(architecture).eval().requires_grad_(False)
        old.load_state_dict(old_state, strict=True)
        new.load_state_dict(old_state, strict=True)
        generator = torch.Generator().manual_seed(27182)
        noise = torch.randn((1, 3, 64, 64), generator=generator)
        labels = torch.tensor([103])
        outputs = []
        for model in (old, new):
            outputs.append(sample_images(model, noise.shape, labels, device='cpu', method='renoise4',
                noise=noise, extra_generator=torch.Generator().manual_seed(20261008), precision='fp32'))
        parity = dict(bitwise_equal=torch.equal(*outputs),
                      max_absolute_difference=float((outputs[0] - outputs[1]).abs().max()))
        write('backbone_parity.json', parity)
        del old, new, outputs, old_state
    print('Backbone parity', parity, flush=True)
    if parity['max_absolute_difference'] > 1e-5:
        raise ValueError('Backbone implementations differ unexpectedly')
    gc.collect()

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    total = torch.cuda.get_device_properties(device).total_memory
    torch.cuda.set_per_process_memory_fraction(.70, device)
    os.chdir(CACHE)
    environment = dict(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
        packages={name: version(name) for name in ('torch', 'torchvision', 'Pillow', 'clean-fid', 'numpy', 'scipy')},
        image_loader_workers=0, server_image_loader_workers=4,
        inception_sha256=sha(CACHE / 'inception-2015-12-05.pt'), backbone_parity=parity)
    write('environment.json', environment)

    # Score already-generated server PNGs to measure feature-extraction drift
    # separately from local model generation.
    if (ROOT / 'metric_calibration.json').exists():
        calibration = json.loads((ROOT / 'metric_calibration.json').read_text())
    else:
        started = time.perf_counter()
        with peer_evaluation_precision():
            cached_score = float(fid.compute_fid(str(ROOT / 'server_current/generated'),
                str(ROOT / 'server_current/reference'), device=device, batch_size=8, num_workers=0))
        calibration = dict(server_FID=expected_server['FID_train_proxy'],
            same_server_PNGs_local_FID=cached_score,
            difference=cached_score - expected_server['FID_train_proxy'], seconds=time.perf_counter()-started)
        write('metric_calibration.json', calibration)
    print('Metric calibration', calibration, flush=True)

    results = {}
    for name, checkpoint, constructor, step in (
        ('current', 'current_few_nfe.ckpt', lambda: build_model(architecture), current_record['step']),
        ('previous_best', 'few_nfe.ckpt', lambda: LegacyModel(**architecture), 188000)):
        directory = ROOT / ('evaluation_' + name)
        if (directory / 'result.json').exists():
            result = json.loads((directory / 'result.json').read_text())
            verify_reference_pixels(directory)
            result.update(step=step, checkpoint_sha256=sha(ROOT / checkpoint),
                          reference_decoded_pixels_match_server=True, reused_completed_evaluation=True)
            results[name] = result
            print('Reused completed result', name, result['FID_train_proxy'], flush=True)
            continue
        model = constructor().eval().requires_grad_(False)
        state = torch.load(ROOT / checkpoint, map_location='cpu', weights_only=True, mmap=True)['state_dict']
        model.load_state_dict({key.removeprefix('backbone.'): value for key, value in state.items()}, strict=True)
        del state
        model.to(device=device, memory_format=torch.contiguous_format)
        assert sum(p.numel() for p in model.parameters()) == 95_524_611
        started = time.perf_counter()
        print('Evaluating', name, 'step', step, flush=True)
        result = train_proxy(model, manifest, ROOT / 'data', directory,
            device=device, method='renoise4', batch_size=8, precision='fp32', seed=27182,
            workers=0, reference_overlaps_training=True)
        verify_reference_pixels(directory)
        result.update(step=step, checkpoint_sha256=sha(ROOT / checkpoint),
                      reference_decoded_pixels_match_server=True,
                      seconds=time.perf_counter()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        if result['peak_allocated_bytes'] >= 20_000_000_000:
            raise RuntimeError('Memory limit exceeded')
        results[name] = result
        write('comparison_partial.json', results)
        print('Result', name, result['FID_train_proxy'], flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    record = dict(results=results, metric_calibration=calibration, environment=environment,
        current_minus_previous=results['current']['FID_train_proxy']-results['previous_best']['FID_train_proxy'],
        server_training_interrupted=False, training_updates=0)
    write('comparison.json', record)
    print('COMPLETED', json.dumps({name: value['FID_train_proxy'] for name, value in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'compare'])
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--metric-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    ROOT = args.work_root.resolve()
    PROJECT = args.project_root.resolve() if args.project_root else ROOT / 'project'
    CACHE = args.metric_cache.resolve()
    prepare() if args.action == 'prepare' else compare()
