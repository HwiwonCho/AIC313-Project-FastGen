"""Create a self-contained four-NFE package and verify it in a fresh CPU process."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from export import export
from best4_utils import ROOT, check_fixed, json_write, sha

def package(source,output):
    check_fixed();output=Path(output).resolve()
    if output==ROOT or output in ROOT.parents:raise ValueError('Use a new output folder')
    for folder in ('configs','assets','data','src','tests','checkpoints'):
        if ROOT/folder in output.parents or output==ROOT/folder:raise ValueError('Output overlaps source files')
    output.mkdir(parents=True,exist_ok=False)
    paths=set()
    for pattern in ('*.py','*.sh','*.md','requirements.txt','.gitattributes','.gitignore',
                    'configs/*.json','src/*.py','tests/*.py','assets/*','.orca/.gitignore',
                    'data/pokemon-generation-one-22k/train_split.txt',
                    'data/pokemon-generation-one-22k/val_split.txt',
                    'data/pokemon-generation-one-22k/category_to_id.json','official_files.json'):
        paths.update(p for p in ROOT.glob(pattern) if p.is_file())
    for source_file in sorted(paths):
        target=output/source_file.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source_file,target)
    receipt=export(source,output/'checkpoints/few_nfe.ckpt')
    env={k:v for k,v in os.environ.items() if k!='PYTHONPATH'}
    command=[sys.executable,'-B','check_few.py','--device','cpu','--record','checkpoints/cpu_check.json']
    result=subprocess.run(command,cwd=output,env=env,capture_output=True,text=True)
    (output/'checkpoints/cpu_check.log').write_text(result.stdout+result.stderr)
    if result.returncode:raise RuntimeError('Packaged four-NFE check failed; see checkpoints/cpu_check.log')
    historical=json.loads((ROOT/'configs/historical_best.json').read_text())
    record={'mode':'few_nfe','parameters':receipt['parameters'],'source_sha256':receipt['source_sha256'],
            'checkpoint_sha256':receipt['output_sha256'],'fresh_process_official_loader_and_four_calls':True,
            'matches_historical_best':receipt['source_sha256']==historical['source_sha256'],
            'one_nfe_checkpoint_included':False,'GPU_official_evaluation_repeated':False,
            'source_files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}}
    json_write(output/'package_manifest.json',record)
    print(json.dumps({'output':str(output),**{k:v for k,v in record.items() if k!='source_files'}},indent=2))
    return record

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('submission'))
    a=p.parse_args();package(a.source,a.output)
