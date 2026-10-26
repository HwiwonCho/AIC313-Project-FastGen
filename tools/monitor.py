"""Read-only terminal monitor; needs no Torch, TensorBoard or GPU allocation."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path, lines=12):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-24000))
            return stream.read().decode('utf-8', errors='replace').splitlines()[-lines:]
    except OSError:
        return []


def render(root):
    status = read_json(root/'status.json')
    output = [f"FastGen | {datetime.datetime.now().astimezone().isoformat(timespec='seconds')}",
              f"Run: {root}", f"State: {status.get('phase','preparing')} / {status.get('stage','data audit')}",
              "Step counts and loss refresh at the logged cadence; FID appears after evaluation.", '']
    for name in ('pilot_flip', 'pilot_spatial', 'teacher', 'distill'):
        value = read_json(root/name/'summary.json')
        if not value:
            output.append(f"{name:14} pending")
            continue
        last = tail(root/name/'metrics.jsonl', 1)
        try:
            row = json.loads(last[0]) if last else {}
        except ValueError:
            row = {}
        memory = value.get('memory',{}).get('peak_allocated_bytes',0)/1e9
        seconds = value.get('active_wall_seconds',0)
        step = value.get('step',0)
        target = value.get('target_steps',0)
        eta = ((target-step)*seconds/step/3600) if step else None
        eta_text = f"rough remaining {eta:.1f}h (includes past evaluation overhead)" if eta is not None else ''
        output.append(f"{name:14} {value.get('phase')} {step}/{target} | loss={row.get('loss','-')} | best TRAIN755 FID={value.get('best_train_proxy_fid')} | peak={memory:.2f}GB")
        if eta_text:
            output.append(' '*15+eta_text)
    output += ['', 'Latest console output:'] + tail(root/'console.log')
    try:
        result = subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
        output += ['', 'GPU utilization %, current used MiB, total MiB: '+result.stdout.strip()]
    except (OSError, subprocess.TimeoutExpired):
        pass
    output += ['', 'TRAIN755 selects checkpoints; official validation3020 is evaluated only at the end.',
               'tmux: Ctrl+B then D detaches; windows 0=train, 1=status. Ctrl+C in TRAIN stops training.']
    return '\n'.join(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--interval',type=float,default=5)
    args=parser.parse_args()
    if args.interval < 1:
        parser.error('interval must be at least one second')
    while True:
        if not args.once:
            print('\033[2J\033[H',end='')
        print(render(args.run_root.resolve()),flush=True)
        if args.once:
            break
        time.sleep(args.interval)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
