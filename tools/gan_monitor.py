"""Read-only tmux monitor of durable trainer logs. No Torch import or model load."""
import csv,io,json,os,subprocess,time
from pathlib import Path
import argparse
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run',required=True);p.add_argument('--output',required=True)
args=p.parse_args()
RUN=Path(args.run).resolve();OUT=Path(args.output).resolve();OUT.mkdir(parents=True,exist_ok=True)

def read(path,default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def atomic(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)

offset=0;header=None;history=[];bucket=[];latest=None
while True:
    try:
        path=RUN/'metrics.csv'
        if path.exists():
            with path.open() as f:
                f.seek(offset)
                while True:
                    position=f.tell();line=f.readline()
                    if not line or not line.endswith('\n'):break
                    offset=f.tell()
                    values=next(csv.reader([line]))
                    if header is None:header=values;continue
                    row=dict(zip(header,values));latest=row;bucket.append(row)
                    if len(bucket)==10:
                        means={k:sum(float(r[k]) for r in bucket if r.get(k))/sum(bool(r.get(k)) for r in bucket)
                               for k in ['Loss/G/loss','Loss/D/loss','Loss/pl_penalty','Loss/r1_penalty'] if any(r.get(k) for r in bucket)}
                        history.append(dict(start=int(bucket[0]['step']),end=int(bucket[-1]['step']),means=means))
                        history=history[-300:];bucket=[]
        status=read(RUN/'status.json',{})
        if not status:time.sleep(5);continue
        pid=status.get('pid',0)
        try:alive=b'stylegan2.train' in Path(f'/proc/{pid}/cmdline').read_bytes()
        except OSError:alive=False
        events=[]
        if (RUN/'events.jsonl').exists():
            for line in (RUN/'events.jsonl').read_text().splitlines():
                try:events.append(json.loads(line))
                except ValueError:pass
        checkpoint=max([e['step'] for e in events if e.get('event')=='checkpoint']+[0])
        errors=[e.get('error') for e in events if e.get('event')=='error'][-3:]
        loss=None
        if latest:
            loss=dict(iteration=checkpoint,metrics={k:dict(mean=float(v)) for k,v in latest.items() if k.startswith('Loss/') and v})
            if history:loss['interval']=history[-1]
        gpu=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=6).stdout.strip()
        image=RUN/'preview.png'
        data=dict(observed_unix=time.time(),status=status,status_age=time.time()-(RUN/'status.json').stat().st_mtime,
                  alive=alive,gpu=gpu,loss=loss,loss_history=history,latest_metrics=latest,
                  fid=read(RUN/'internal_fid.json'),preview_mtime=image.stat().st_mtime if image.exists() else None,errors=errors)
        atomic(OUT/'snapshot.json',data)
        print('\033[2J\033[HFastGen GAN — READ ONLY; training tmux: fastgen-gan',flush=True)
        print('Step',status.get('iteration'),'/ 125000; state',status.get('state'),'alive',alive)
        print('Seconds/update',status.get('seconds_per_update'),'peak GB',status.get('peak_allocated_bytes',0)/1e9,'GPU',gpu)
        print('Latest 10-step loss:',history[-1] if history else 'waiting for 10 updates')
        print('TRAIN holdout FID:',data['fid'],'errors:',errors or 'none',flush=True)
    except Exception as e:print('Monitor read retry:',type(e).__name__,str(e),flush=True)
    time.sleep(10)
