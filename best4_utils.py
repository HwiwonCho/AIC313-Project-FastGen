"""Fixed-file checks, atomic output and complete training state."""
import ast
import contextlib
import hashlib
import json
import os
import random
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def json_write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(value, temp); os.replace(temp, path)

def check_fixed():
    record = json.loads((ROOT/'official_files.json').read_text())
    for name in record['initial_starter_paths']:
        if not (ROOT/name).is_file(): raise ValueError('Missing starter file: '+name)
    for name, expected in record['sha256'].items():
        if sha(ROOT/name) != expected: raise ValueError('Fixed starter changed: '+name)
    text = (ROOT/'model.py').read_text()
    base = next(x for x in ast.parse(text).body if isinstance(x, ast.ClassDef) and x.name=='Model')
    if hashlib.sha256(ast.get_source_segment(text, base).encode()).hexdigest() != record['base_Model_source_sha256']:
        raise ValueError('Official Model base changed')

def rng_state(data_rng, aug_rng):
    return {'python': random.getstate(), 'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'data': data_rng.getstate(), 'augmentation': aug_rng.getstate()}

def restore_rng(state, data_rng, aug_rng):
    random.setstate(state['python']); torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']: torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    data_rng.setstate(state['data']); aug_rng.setstate(state['augmentation'])

@contextlib.contextmanager
def preserved_global_rng():
    import numpy as np
    python, cpu, numpy = random.getstate(), torch.get_rng_state(), np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    try: yield
    finally:
        random.setstate(python); torch.set_rng_state(cpu); np.random.set_state(numpy)
        if cuda: torch.cuda.set_rng_state_all(cuda)

class MemoryGuard:
    def __init__(self, device):
        self.device = torch.device(device)
        if self.device.type=='cuda':
            total = torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min(total,18_000_000_000)/total, self.device)
    def check(self):
        if self.device.type!='cuda': return {'allocated_peak_bytes':0,'reserved_peak_bytes':0}
        a = torch.cuda.max_memory_allocated(self.device)
        r = torch.cuda.max_memory_reserved(self.device)
        if a >= 20_000_000_000: raise RuntimeError('Training peak reached 20GB')
        return {'allocated_peak_bytes':a,'reserved_peak_bytes':r}
