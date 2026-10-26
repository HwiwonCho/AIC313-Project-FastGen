"""Start exactly one pipeline in tmux, with persistent logs and a read-only status window."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    parser.add_argument('--session',default='fastgen')
    parser.add_argument('--continue-from',type=Path)
    parser.add_argument('--reuse-session',action='store_true',help='Reuse an existing session only when its training pane has exited')
    args=parser.parse_args()
    if shutil.which('tmux') is None:
        parser.error('tmux is not installed')
    if not args.session.replace('-','').replace('_','').isalnum():
        parser.error('Use a simple tmux session name')
    existing = subprocess.run(['tmux','has-session','-t',args.session],capture_output=True).returncode==0
    if existing:
        if not args.reuse_session:
            parser.error('The tmux session already exists; use --reuse-session only after training has exited')
        dead = subprocess.check_output(['tmux','display-message','-p','-t',args.session+':train.0','#{pane_dead}'],text=True).strip()
        if dead != '1':
            parser.error('Existing training pane is alive; refusing to start a second pipeline')
        status_command = subprocess.check_output(['tmux','display-message','-p','-t',args.session+':status.0','#{pane_current_command}'],text=True).strip()
        if status_command not in ('python','python3'):
            parser.error('Status pane is not a known Python monitor')
    project=Path(__file__).resolve().parents[1]
    root=args.run_root.resolve()
    root.mkdir(parents=True,exist_ok=True)
    python=str(args.python.absolute())
    if not args.python.is_file() or not args.data_root.is_dir():
        parser.error('Python or data root is missing')
    command=[python,'-u','-B',str(project/'pipeline.py'),'run','--config',str(project/'configs/deadline.json'),
             '--data-root',str(args.data_root.resolve()),'--run-root',str(root)]
    if args.continue_from:
        command += ['--continue-from',str(args.continue_from.resolve(strict=True))]
    launcher=root/'launch_tmux.sh'
    if launcher.exists():
        parser.error('A launch script already exists; inspect the previous run before resuming')
    launcher.write_text('#!/usr/bin/env bash\nset -o pipefail\nexport CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4\n'
        +'cd '+shlex.quote(str(project))+'\n'+shlex.join(command)+' 2>&1 | tee -a '+shlex.quote(str(root/'console.log'))
        +'\ncode=${PIPESTATUS[0]}\nprintf "%s\\n" "$code" > '+shlex.quote(str(root/'pipeline_exit_code.txt'))+'\nexit "$code"\n')
    receipt={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'session':args.session,
             'command':command,'project':str(project),'run_root':str(root),
             'launcher_sha256':hashlib.sha256(launcher.read_bytes()).hexdigest(),
             'environment':{'CUDA_VISIBLE_DEVICES':'0','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTHONUNBUFFERED':'1'},
             'attach_command':'tmux attach -t '+args.session}
    (root/'launch.json').write_text(json.dumps(receipt,indent=2))
    if existing:
        subprocess.run(['tmux','respawn-pane','-t',args.session+':train.0','-c',str(project),'bash '+shlex.quote(str(launcher))],check=True)
    else:
        subprocess.run(['tmux','new-session','-d','-s',args.session,'-n','train','-c',str(project),'bash '+shlex.quote(str(launcher))],check=True)
    subprocess.run(['tmux','set-option','-t',args.session,'remain-on-exit','on'],check=True)
    monitor=shlex.join([python,'-u','-B',str(project/'tools/status.py'),str(root)])
    if existing:
        subprocess.run(['tmux','respawn-pane','-k','-t',args.session+':status.0','-c',str(project),monitor],check=True)
    else:
        subprocess.run(['tmux','new-window','-d','-t',args.session,'-n','status','-c',str(project),monitor],check=True)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
