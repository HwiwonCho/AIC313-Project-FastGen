"""Project-owned, compact read-only tmux dashboard; no GPU allocation."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import unicodedata

KST = timezone(timedelta(hours=9))
STAGES = [('pilot_flip','증강 1 · 좌우반전'), ('pilot_spatial','증강 2 · 기하변환'),
          ('teacher','4NFE 새 학습'), ('pairs_teacher','증류 데이터 생성'),
          ('distill','1NFE 증류'), ('submission','제출 파일 / 평가')]
PHASES = {'training':'학습 중','complete':'완료','running':'실행 중','failed':'실패',
          'stopped':'중단','interrupted':'중단','stopped_or_failed':'중단/실패'}


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,ValueError):
        return {}


def tail(path, limit=16):
    try:
        with path.open('rb') as f:
            f.seek(0,2)
            f.seek(max(0,f.tell()-24000))
            lines=f.read().decode('utf-8',errors='replace').splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.startswith('{')]
    except (OSError,ValueError):
        return []


def width(text):
    return sum(2 if unicodedata.east_asian_width(c) in ('W','F') else 1 for c in text)


def cut(text, length):
    result=''
    for c in text:
        if width(result+c)>length:
            break
        result+=c
    return result


def pad(text,length):
    text=cut(text,length)
    return text+' '*max(0,length-width(text))


def number(value, decimals=4):
    return f'{value:.{decimals}f}' if isinstance(value,(float,int)) and math.isfinite(value) else '--'


def duration(seconds):
    if seconds is None or not math.isfinite(seconds):
        return '--'
    minutes=max(0,int(seconds)//60)
    return f'{minutes//60}시간 {minutes%60:02d}분' if minutes>=60 else f'{minutes}분'


def gpu():
    try:
        result=subprocess.run(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3)
        util, used=result.stdout.strip().splitlines()[0].split(',')
        return float(util),float(used)*1024**2/1e9
    except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):
        return None,None


def render(root,columns=100,lines=30):
    status=read(root/'status.json')
    current=status.get('stage','')
    summaries={key:read(root/key/'summary.json') for key,_ in STAGES}
    label=dict(STAGES).get(current,current or '준비')
    summary=summaries.get(current,{})
    records=tail(root/current/'metrics.jsonl') if current else []
    latest=records[-1] if records else {}
    step=summary.get('step',0)
    target=summary.get('target_steps',0)
    phase=status.get('phase','preparing')
    state=PHASES.get(summary.get('phase',phase),phase)
    if phase in ('failed','interrupted','stopped_or_failed'):
        state=PHASES.get(phase,phase)
    config=read(root/'pipeline_identity.json').get('config',{})
    phase_settings=config.get('teacher' if current=='teacher' else 'distill' if current=='distill' else 'pilot',{})
    eval_every=phase_settings.get('eval_every',phase_settings.get('steps_each',3000))
    evaluating=bool(step and eval_every and step%eval_every==0 and summary.get('phase')=='training')
    if evaluating:
        state='FID 평가 / 저장 처리'
    now=time.time()
    speed=None
    if len(records)>1:
        recent=records[-5:]
        elapsed=recent[-1]['unix']-recent[0]['unix']
        speed=(recent[-1]['step']-recent[0]['step'])/elapsed if elapsed>0 else None
    eta=(target-step)/speed if speed and target else None
    peak=max((v.get('memory',{}).get('peak_allocated_bytes',0) for v in summaries.values()),default=0)/1e9
    util,used=gpu()
    rows=[('36;1',f'FastGen  |  {datetime.now(KST):%m/%d %H:%M:%S} KST  |  5초 갱신'),
          ('1',f'현재: {label}  ·  {state}')]
    if target:
        count=max(10,min(34,columns-40))
        filled=int(count*min(1,step/target))
        rows.append(('36','['+'='*filled+'-'*(count-filled)+f'] {step:,} / {target:,}  ({100*step/target:.1f}%)'))
    rows.append(('',f'손실 {number(latest.get("loss"))}   |   최근 속도 {number(speed,2)} 업데이트/초'))
    if latest:
        age=max(0,int(now-latest.get('unix',now)))
        rows.append(('',f'이 단계 남은 시간 약 {duration(eta)}   |   마지막 학습 기록 {age}초 전'))
    if eta is not None and target>step and not evaluating:
        estimated=datetime.now(KST)+timedelta(seconds=eta)
        rows.append(('33;1',f'이 단계 예상 완료 {estimated:%m/%d %H:%M} KST · 후속 증류/최종평가는 별도'))
    events=tail(root/current/'events.jsonl',40) if current else []
    scores=[v for v in events if v.get('event')=='train_proxy']
    if scores:
        rows.append(('32;1',f'현재 모델 TRAIN755 FID: 최근 {number(scores[-1].get("score"),2)} / 최저 {number(min(v["score"] for v in scores),2)}'))
    elif current in ('teacher','distill','pilot_flip','pilot_spatial'):
        until=max(0,(min(target,(step//eval_every+1)*eval_every)-step)/speed) if speed and target else None
        rows.append(('33;1',f'현재 모델 FID: 아직 미측정 · 첫/다음 측정까지 약 {duration(until)}'))
    rows.append(('',f'학습 최고 메모리 {peak:.2f} / 20.00 GB   |   GPU 현재 {number(used,2)} GB'))
    rows.append(('',f'GPU 사용률 {number(util,0)}% (순간 측정)   |   모델 95.52M / 100M'))
    rows.append(('', ''))
    rows.append(('1',pad('단계',23)+pad('진행',17)+pad('상태',11)+'최저 TRAIN755 FID'))
    rows.append(('2','-'*min(columns-1,78)))
    for key,name in STAGES:
        value=summaries[key]
        if key=='pairs_teacher':
            value=read(root/key/'summary.json')
            progress=f'{value.get("count",0):,}쌍' if value else '--'
        elif key=='submission':
            done=phase=='complete'
            value={'phase':'complete'} if done else {}
            progress='1NFE + 4NFE' if (root/'submission').exists() else '--'
        else:
            progress=f'{value.get("step",0):,}/{value.get("target_steps",0):,}' if value else '--'
        this_phase=value.get('phase')
        state_label=PHASES.get(this_phase,this_phase or '대기')
        color='32' if this_phase=='complete' else '33;1' if key==current else '2' if not value else ''
        rows.append((color,pad(name,23)+pad(progress,17)+pad(state_label,11)+number(value.get('best_train_proxy_fid'),2)))
    selection=read(root/'augmentation_selection.json')
    if selection:
        chosen='기하변환' if selection.get('augmentation')=='spatial' else '좌우반전'
        rows.append(('32',f'선택된 증강: {chosen}'))
    rows.append(('', ''))
    if evaluating:
        rows.append(('33','학습 기록이 잠시 멈춰 보여도 FID 평가·저장 중일 수 있습니다.'))
    elif target and step<target:
        next_eval=min(target,(step//eval_every+1)*eval_every)
        rows.append(('',f'다음 FID: {next_eval:,} 업데이트   |   남은 시간은 최근 속도 기준 추정'))
    rows.append(('2','파일럿 FID는 초반 학습 비교값입니다. 최종 성능은 아직 미측정.'))
    rows.append(('2','TRAIN755 = 학습 중 선택 지표 · 공식 validation3020 = 마지막 별도 평가'))
    if phase in ('failed','interrupted','stopped_or_failed'):
        rows.append(('31;1','오류: '+str(status.get('error','학습 로그 창을 확인하세요.'))))
    rows.append(('', ''))
    rows.append(('36','로그 Ctrl+B → 0   |   상태 Ctrl+B → 1   |   화면 나가기 Ctrl+B → D'))
    rows=rows[:max(1,lines-1)]
    return [(color,cut(text,max(10,columns-1))) for color,text in rows]


def main():
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8',errors='replace')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root',type=Path)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    interactive=sys.stdout.isatty() and not args.once
    if interactive:
        print('\033[?1049h\033[?25l',end='',flush=True)
    try:
        while True:
            size=shutil.get_terminal_size((110,30))
            rows=render(args.run_root,size.columns,size.lines)
            if interactive:
                print('\033[H',end='')
            for color,text in rows:
                if interactive:
                    print('\033[2K'+(f'\033[{color}m' if color else '')+text+'\033[0m',end='\r\n')
                else:
                    print(text)
            if interactive:
                print('\033[J',end='',flush=True)
            if args.once:
                break
            time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        if interactive:
            print('\033[?25h\033[?1049l',end='',flush=True)


if __name__=='__main__':
    main()
