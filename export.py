"""Export a raw SoFlow EMA through the unchanged official checkpoint loader."""
import argparse
from pathlib import Path
import torch
from model import Model
from best4_utils import check_fixed, save, json_write, sha

def export(source, output):
    check_fixed()
    source, output = Path(source).resolve(strict=True), Path(output).resolve()
    source_sha = sha(source)
    raw = torch.load(source, map_location='cpu', weights_only=False, mmap=True)
    ema = raw.get('ema', raw.get('ema_model'))
    if not isinstance(ema, dict): raise ValueError('Source has no SoFlow EMA')
    state = {'backbone.'+key:value for key,value in ema.items()}
    if output.exists():
        previous = torch.load(output, map_location='cpu', weights_only=True)
        previous = previous.get('state_dict',previous)
        if set(previous)!=set(state) or not all(torch.equal(previous[k],state[k]) for k in state):
            raise ValueError('Refusing to replace different exported weights')
        del previous
    else: save(output, {'state_dict':state})
    loaded = Model.load_checkpoint(str(output), 'few_nfe', device='cpu')
    if loaded.count_parameters()!=95_524_611: raise ValueError('Parameter count differs')
    if not all(torch.equal(loaded.state_dict()[k],value) for k,value in state.items()):
        raise ValueError('Export is not the exact source EMA')
    receipt = {'source_sha256':source_sha,'source_step':raw.get('step'),'output_sha256':sha(output),
               'parameters':loaded.count_parameters(),'strict_official_loader_passed':True,
               'all_EMA_tensors_exactly_equal':True,'training_updates':0}
    json_write(output.with_suffix('.provenance.json'),receipt)
    print(receipt)
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--output',type=Path,default=Path('checkpoints/few_nfe.ckpt'))
    a=p.parse_args();export(a.source,a.output)
