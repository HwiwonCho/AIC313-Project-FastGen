"""Durable experiment evidence, independent of model math and random streams."""
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from importlib.metadata import version
from PIL import Image, ImageDraw
import torch
from . import portable_ops  # Establish the pinned vendor import path for the CLI.
from torch_utils import training_stats

LOSS_KEYS = ['Loss/G/loss','Loss/D/loss','Loss/G/reg','Loss/D/reg',
             'Loss/pl_penalty','Loss/r1_penalty','Loss/scores/real',
             'Loss/scores/fake','Loss/signs/real','Loss/signs/fake']

def atomic_json(path, value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def append_json(path,value):
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

class Recorder:
    def __init__(self,run,engine,project,resume=False,session_start=None):
        self.run,self.engine=Path(run),engine
        self.start=time.perf_counter() if session_start is None else session_start
        self.previous=getattr(engine,'recording_state',{})
        self.update_seconds=self.previous.get('update_seconds',0.)
        self.base_wall=self.previous.get('active_wall_seconds',0.)
        if resume and (self.run/'timing.json').exists():
            timing=json.loads((self.run/'timing.json').read_text())
            self.update_seconds=max(self.update_seconds,timing.get('update_seconds',0.))
            self.base_wall=max(self.base_wall,timing.get('active_wall_seconds',0.))
        self.collector=training_stats.Collector(regex='Loss/.*',keep_previous=False)
        self.points=[]
        self.fields=['step','images_seen','unix','update_seconds','active_wall_seconds',
                     'total_update_seconds','peak_allocated_bytes','ada_p','g_lr','d_lr']+LOSS_KEYS
        path=self.run/'metrics.csv'
        if resume and path.exists():
            with path.open(newline='') as f: rows=list(csv.DictReader(f))
            if rows:
                self.update_seconds=max(self.update_seconds,float(rows[-1]['total_update_seconds']))
                self.base_wall=max(self.base_wall,float(rows[-1]['active_wall_seconds']))
            keep=[r for r in rows if int(r['step'])<=engine.iteration]
            orphan=[r for r in rows if int(r['step'])>engine.iteration]
            if orphan:
                append_json(self.run/'events.jsonl',dict(event='uncheckpointed_metrics',unix=time.time(),rows=orphan))
                temp=path.with_suffix('.tmp')
                with temp.open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=self.fields);writer.writeheader();writer.writerows(keep)
                os.replace(temp,path)
            self.points=[(int(r['step']),float(r['Loss/G/loss']),float(r['Loss/D/loss'])) for r in keep]
        elif path.exists():
            raise RuntimeError('Refusing to mix an existing metrics file into a new run')
        self.file=path.open('a',newline='',encoding='utf-8')
        self.writer=csv.DictWriter(self.file,fieldnames=self.fields)
        if path.stat().st_size==0:self.writer.writeheader();self.file.flush();os.fsync(self.file.fileno())
        if not resume:
            self.snapshot(project)
            atomic_json(self.run/'environment.json',dict(python=platform.python_version(),
                torch=torch.__version__,cuda=torch.version.cuda,device=str(engine.device),
                gpu=torch.cuda.get_device_name() if engine.device.type=='cuda' else None,
                packages={k:version(k) for k in ['torch','torchvision','Pillow','numpy','clean-fid','tqdm']},
                revision=os.environ.get('FASTGEN_REVISION','unknown'),
                parameters={k:sum(p.numel() for p in m.parameters()) for k,m in [('G',engine.G),('D',engine.D),('EMA',engine.ema)]}))
        self.event('resume' if resume else 'start',iteration=engine.iteration,argv=sys.argv)

    def snapshot(self,project):
        project=Path(project);dest=self.run/'source_snapshot';dest.mkdir(exist_ok=False)
        files=[project/n for n in ['model.py','dataset.py','download_dataset.py','evaluate.py','requirements.txt','README.md']]
        for folder in ['stylegan2','src','tools','data/pokemon-generation-one-22k']:
            files += [p for p in (project/folder).rglob('*') if p.is_file() and
                      '__pycache__' not in p.parts and p.suffix in ['.py','.json','.txt','.md']]
        manifest={}
        for p in sorted(set(files)):
            relative=p.relative_to(project);target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target);manifest[relative.as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
        atomic_json(self.run/'source_manifest.json',manifest)

    def timing(self):
        value=dict(active_wall_seconds=self.base_wall+time.perf_counter()-self.start,
                   update_seconds=self.update_seconds)
        self.engine.recording_state=value
        return value

    def event(self,name,**values):
        append_json(self.run/'events.jsonl',dict(event=name,unix=time.time(),step=self.engine.iteration,**values))

    def update(self,seconds,peak):
        self.update_seconds+=seconds;self.collector.update();e=self.engine
        row=dict(step=e.iteration,images_seen=e.images_seen,unix=time.time(),update_seconds=seconds,
                 active_wall_seconds=self.timing()['active_wall_seconds'],total_update_seconds=self.update_seconds,
                 peak_allocated_bytes=peak,ada_p=float(e.augment.p),g_lr=e.g_opt.param_groups[0]['lr'],d_lr=e.d_opt.param_groups[0]['lr'])
        for name in LOSS_KEYS:row[name]=self.collector.mean(name) if self.collector.num(name) else None
        self.writer.writerow(row);self.file.flush();os.fsync(self.file.fileno())
        self.points.append((e.iteration,row['Loss/G/loss'],row['Loss/D/loss']))
        atomic_json(self.run/'latest_metrics.json',row)
        return row

    def curves(self):
        # Raw data stay intact; draw 100-update bin means with explicit axes.
        image=Image.new('RGB',(1000,620),'white');draw=ImageDraw.Draw(image)
        bins=[]
        for offset in range(0,len(self.points),100):
            part=self.points[offset:offset+100]
            bins.append((part[-1][0],sum(x[1] for x in part)/len(part),sum(x[2] for x in part)/len(part)))
        for index,name,color,top in [(1,'G logistic loss','green',40),(2,'D logistic loss','blue',340)]:
            values=[p[index] for p in bins]
            if not values:continue
            lo,hi=min(values),max(values);pad=max((hi-lo)*.1,.01);lo-=pad;hi+=pad
            draw.text((55,top-25),f'{name} | 100-update means; raw: metrics.csv',fill='black')
            draw.rectangle((55,top,960,top+220),outline='gray')
            points=[(55+(p[0]-bins[0][0])/max(bins[-1][0]-bins[0][0],1)*905,top+220-(p[index]-lo)/(hi-lo)*220) for p in bins]
            if len(points)>1:draw.line(points,fill=color,width=2)
            for x,y in points:draw.ellipse((x-2,y-2,x+2,y+2),fill=color)
            draw.text((5,top),f'{hi:.3f}',fill='black');draw.text((5,top+205),f'{lo:.3f}',fill='black')
            draw.text((55,top+230),str(bins[0][0]),fill='black');draw.text((900,top+230),str(bins[-1][0])+' step',fill='black')
        temp=self.run/'loss_curve.tmp';image.save(temp,format='PNG');os.replace(temp,self.run/'loss_curve.png')

    def close(self):
        self.curves();atomic_json(self.run/'timing.json',self.timing());self.file.close()
