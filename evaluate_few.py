"""Run the unmodified official evaluator using a private validation image view."""
import argparse
from pathlib import Path
import subprocess
import sys
import time
from PIL import Image
from best4_utils import ROOT, check_fixed, json_write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,default=Path('results/few_nfe'))
    p.add_argument('--checkpoint',type=Path,default=Path('checkpoints/few_nfe.ckpt'))
    p.add_argument('--device',default='cuda');a=p.parse_args();check_fixed()
    output=a.output.resolve();output.mkdir(parents=True,exist_ok=True)
    view=output/'data_view';view.mkdir(exist_ok=True)
    split=ROOT/'data/pokemon-generation-one-22k'
    for name in (split/'val_split.txt').read_text().splitlines():
        if not name:continue
        source=(a.data_root/name).resolve(strict=True)
        if a.data_root.resolve() not in source.parents:raise ValueError('Validation path escapes data root')
        target=view/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            with Image.open(source) as image:
                rgba=image.convert('RGBA');white=Image.new('RGBA',rgba.size,'white');white.alpha_composite(rgba)
                white.convert('RGB').save(target,format='PNG')
    (view/'.transparency_composited_on_white').touch()
    code=("import torch,runpy;torch.manual_seed(20261002);torch.cuda.manual_seed_all(20261002);"
          "torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;"
          "torch.backends.cudnn.benchmark=False;runpy.run_path('evaluate.py',run_name='__main__')")
    command=[sys.executable,'-B','-c',code,'--model_checkpoint',str(a.checkpoint.resolve()),'--evaluate_mode','few_nfe',
             '--data_root',str(view),'--split_dir',str(split),'--reference_dir',str(output/'reference'),
             '--output_dir',str(output/'generated'),'--batch_size','4','--device',a.device]
    started=time.perf_counter()
    with (output/'console.log').open('w') as log:
        result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Provided evaluator failed: '+str(output/'console.log'))
    json_write(output/'completed.json',{'provided_evaluator_returncode':0,'samples':3020,'seed':20261002,
               'batch_size':4,'seconds':time.perf_counter()-started,'TA_measured':False,'used_for_selection':False})
    print((output/'console.log').read_text()[-1400:])

if __name__=='__main__':main()
