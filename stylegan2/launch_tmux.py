"""Linux server launcher: real training in tmux, separate live log window."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session',default='fastgen-gan')
    p.add_argument('--run',required=True)
    args,extra=p.parse_known_args()
    if os.name!='posix' or not shutil.which('tmux'):raise RuntimeError('Requires existing Linux tmux; installs nothing')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.session):raise ValueError('Invalid tmux session name')
    if subprocess.run(['tmux','has-session','-t',args.session],capture_output=True).returncode==0:
        raise RuntimeError('Session already exists; refusing duplicate training')
    root=Path(__file__).resolve().parent.parent
    run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-u','-B','-m','stylegan2.train','--run',str(run),*extra]
    revision=(root/'REVISION').read_text().strip() if (root/'REVISION').exists() else 'unknown'
    environment=['env','CUDA_VISIBLE_DEVICES=0','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4',f'FASTGEN_REVISION={revision}']
    shell='exec '+shlex.join(environment+command)+' >'+shlex.quote(str(run/'stdout.log'))+' 2>'+shlex.quote(str(run/'stderr.log'))
    (run/'launcher.json').write_text(json.dumps(dict(command=command,revision=revision,
        started_unix=time.time(),tmux_session=args.session,fresh_start=True),indent=2))
    subprocess.run(['tmux','new-session','-d','-s',args.session,'-n','train','-c',str(root),shell],check=True)
    subprocess.run(['tmux','set-option','-w','-t',args.session+':train','remain-on-exit','on'],check=True)
    logs='exec tail -F '+shlex.quote(str(run/'stdout.log'))+' '+shlex.quote(str(run/'stderr.log'))
    subprocess.run(['tmux','new-window','-t',args.session,'-n','logs',logs],check=True)
    print(json.dumps(dict(session=args.session,run=str(run),command=command)),flush=True)

if __name__=='__main__':main()
