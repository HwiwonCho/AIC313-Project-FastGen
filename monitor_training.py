"""Read-only FastGen training dashboard, served through an SSH tunnel.

Run on the KCloud host: python3 monitor_training.py
Open locally after port forwarding: http://127.0.0.1:8765/
No training process, checkpoint, or experiment file is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parent
RUNS = {
    "vae": ROOT / "runs/vae/20260929_server_fresh",
    "gan": ROOT / "runs/gan/20260929_server_fresh",
    "soflow_v2": ROOT / "runs/soflow_v2/20260929_server_fresh",
}
FID_RUNS = {
    ("vae", 1): ROOT / "results/self_fid/vae_20260929_server_fresh_nfe1",
    ("gan", 1): ROOT / "results/self_fid/gan_20260929_server_fresh_nfe1",
    ("soflow_v2", 1): ROOT / "results/self_fid/soflow_v2_20260929_server_fresh_nfe1",
    ("soflow_v2", 4): ROOT / "results/self_fid/soflow_v2_20260929_server_fresh_nfe4",
}
METRICS = {
    "vae": ("total_loss", "reconstruction_loss", "kl_loss"),
    "gan": ("generator_loss", "discriminator_loss"),
    "soflow_v2": ("loss", "velocity_mse", "consistency_mse"),
}
SAMPLE_NAME = re.compile(r"^epoch_(\d{3})(?:_(ema|raw))?\.png$")
COMPARE_SAMPLE_NAME = re.compile(r"^(?:0009|0014|0103|0129)_0[0-3]\.png$")
BATCH = re.compile(r"SoFlow v2:[^\r\n]*\|\s*(\d+)/(\d+)\s*\[([^\]]*)\]")


def read_json(path: Path) -> dict:
    try:
        if path.stat().st_size > 2_000_000:
            return {}
        with path.open(encoding="utf-8") as stream:
            result = json.load(stream)
        return result if isinstance(result, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def finite_float(value: str | None) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def read_losses(path: Path, model: str) -> list[dict]:
    rows = []
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            for row in reader:
                if len(rows) >= 5000:
                    break
                try:
                    epoch = int(row.get("epoch", ""))
                except (TypeError, ValueError):
                    continue
                if epoch < 1:
                    continue
                values = {key: finite_float(row.get(key)) for key in METRICS[model]}
                if values[METRICS[model][0]] is None:
                    continue  # An actively appended, partial CSV row.
                values["epoch"] = epoch
                values["elapsed_seconds"] = finite_float(row.get("elapsed_seconds"))
                values["peak_vram_gb"] = finite_float(row.get("peak_vram_gb"))
                rows.append(values)
    except (OSError, UnicodeError, csv.Error):
        return rows
    return rows


def fid_results() -> list[dict]:
    results = []
    for (model, nfe), folder in FID_RUNS.items():
        record = read_json(folder / "fid_result.json")
        value = finite_float(record.get("fid"))
        if record.get("status") != "self_computed_fid_not_ta_official" or value is None:
            continue
        results.append({
            "model": model,
            "nfe": nfe,
            "fid": value,
            "checkpoint_sha256": record.get("checkpoint_sha256"),
            "weight_key": record.get("weight_key"),
            "seed": record.get("seed"),
        })
    return results


def process_alive(pid: object) -> bool | None:
    if not isinstance(pid, int) or pid < 1:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def latest_sample(run_dir: Path) -> tuple[Path, dict] | None:
    sample_dir = run_dir / "samples"
    try:
        candidates = []
        for path in sample_dir.iterdir():
            match = SAMPLE_NAME.fullmatch(path.name)
            if not match or not path.is_file():
                continue
            stat = path.stat()
            if stat.st_size > 5_000_000:
                continue
            epoch = int(match.group(1))
            kind = match.group(2) or "sample"
            candidates.append(((epoch, kind == "ema", stat.st_mtime_ns), path, {
                "epoch": epoch, "kind": kind, "mtime_ns": stat.st_mtime_ns,
            }))
        if candidates:
            _, path, info = max(candidates, key=lambda item: item[0])
            return path, info
    except OSError:
        pass
    return None


def batch_progress(path: Path) -> dict | None:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 32_768))
            tail = stream.read().decode("utf-8", errors="replace")
        matches = list(BATCH.finditer(tail))
        if not matches:
            return None
        match = matches[-1]
        current, total = int(match.group(1)), int(match.group(2))
        if not 0 <= current <= total or total < 1:
            return None
        return {"current": current, "total": total, "detail": match.group(3)[-100:]}
    except OSError:
        return None


def gpu_state() -> dict | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2, check=False,
        )
        if result.returncode != 0:
            return None
        line = result.stdout.splitlines()[0]
        name, utilization, used, total, temp = [part.strip() for part in line.split(",", 4)]
        return {"name": name, "utilization_percent": int(utilization),
                "memory_used_mib": int(used), "memory_total_mib": int(total),
                "temperature_c": int(temp) if temp.isdecimal() else None}
    except (OSError, subprocess.TimeoutExpired, IndexError, ValueError):
        return None


def status_snapshot() -> dict:
    results = []
    for model, run_dir in RUNS.items():
        manifest = read_json(run_dir / "run_manifest.json")
        rows = read_losses(run_dir / "losses.csv", model)
        sample = latest_sample(run_dir)
        try:
            log_age = max(0, int(datetime.now(timezone.utc).timestamp() -
                                 (run_dir / "train.log").stat().st_mtime))
        except OSError:
            log_age = None
        config = manifest.get("config") or {}
        results.append({
            "model": model,
            "run_id": run_dir.name,
            "status": manifest.get("status", "not_started"),
            "epoch": rows[-1]["epoch"] if rows else 0,
            "target_epochs": config.get("epochs"),
            "started_utc": manifest.get("started_utc"),
            "ended_utc": manifest.get("ended_utc"),
            "wall_seconds": manifest.get("wall_seconds"),
            "exit_code": manifest.get("exit_code"),
            "process_alive": process_alive(manifest.get("pid")),
            "log_age_seconds": log_age,
            "metrics": list(METRICS[model]),
            "history": rows,
            "sample": sample[1] if sample else None,
            "batch": batch_progress(run_dir / "train.log") if model == "soflow_v2" and
                     manifest.get("status") == "running" else None,
        })
    return {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "runs": results, "fid_results": fid_results(), "gpu": gpu_state()}


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FastGen 학습 현황</title><style>
:root{color-scheme:dark;font-family:ui-sans-serif,system-ui,-apple-system,'Segoe UI',sans-serif;background:#0b1020;color:#e9eefb}
*{box-sizing:border-box}body{margin:0 auto;max-width:1320px;padding:24px 20px 48px}
header{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:26px}h1{font-size:clamp(24px,3vw,36px);margin:0 0 6px;letter-spacing:-.04em}p{margin:0;color:#9aa8c7}
.pill{padding:7px 12px;background:#202b46;border:1px solid #33425e;border-radius:999px;font-size:13px;white-space:nowrap}
.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px}.card{background:#151d31;border:1px solid #2c3853;border-radius:18px;padding:19px;min-width:0}
.top{display:flex;align-items:center;justify-content:space-between;gap:8px}.top h2{font-size:20px;margin:0}.state{border-radius:999px;padding:5px 10px;font-size:12px;background:#26344e;color:#baccf3}
.state.completed{color:#83e3bc;background:#163d36}.state.running{color:#90bbff;background:#1a335b}.state.failed,.state.stale{color:#ffb5a7;background:#512c32}
.progress-line{display:flex;justify-content:space-between;align-items:baseline;margin-top:20px}.progress-line strong{font-size:27px}.small{color:#9aa8c7;font-size:12px}
.track{height:8px;background:#263149;border-radius:999px;overflow:hidden;margin-top:9px}.fill{height:100%;background:linear-gradient(90deg,#55aaff,#8d7cfb);width:0%}
.batch{min-height:30px;margin-top:10px;color:#b9c9e7;font-size:13px}.detail{font-size:12px;color:#8fa0bf;white-space:nowrap;text-overflow:ellipsis;overflow:hidden}
.metric{display:flex;justify-content:space-between;align-items:center;margin:18px 0 6px;gap:8px}.metric select{background:#202b46;border:1px solid #394a6b;border-radius:7px;color:#dde8fc;padding:5px;max-width:72%}
canvas{display:block;width:100%;height:150px}.sample-title{font-size:12px;color:#9aa8c7;margin:15px 0 7px}.sample{display:block;width:100%;max-height:270px;object-fit:contain;background:#0e1526;border:1px solid #2c3853;border-radius:9px;image-rendering:auto}
.foot{margin-top:12px;border-top:1px solid #2c3853;padding-top:11px;font-size:12px;color:#9aa8c7;line-height:1.65}
.gpu{display:flex;flex-wrap:wrap;gap:10px;margin-bottom:20px}.gpu span{background:#151d31;border:1px solid #2c3853;border-radius:10px;padding:9px 13px;font-size:13px}
.section-head{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin:0 0 14px}.section-head h2{font-size:21px;margin:0}.section-head p{font-size:13px}
.fid-board{margin-bottom:20px}.fid-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:11px}.fid-item{background:#1d2940;border:1px solid #354563;border-radius:13px;padding:13px 15px}.fid-item span{font-size:13px;color:#b9c9e7}.fid-item strong{display:block;font-size:29px;margin:5px 0 2px;letter-spacing:-.04em}.fid-item small{color:#8fa0bf;font-size:12px}
.compare{margin-top:20px}.compare h2{font-size:23px;margin:0 0 6px}.compare-note{font-size:13px;line-height:1.6}.compare-panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:18px}.compare-panel{min-width:0}.compare-panel-head{display:flex;align-items:baseline;justify-content:space-between;gap:8px;margin-bottom:10px}.compare-panel h3{font-size:17px;margin:0}.compare-fid{font-size:13px;color:#b9c9e7}.compare-sheet{display:grid;gap:3px}.compare-row{display:grid;grid-template-columns:76px repeat(4,minmax(0,1fr));gap:3px;align-items:center}.compare-row span{font-size:12px;color:#b9c9e7;overflow:hidden;text-overflow:ellipsis}.compare-row img{display:block;width:100%;aspect-ratio:1/1;object-fit:cover;border:1px solid #3a4865;border-radius:4px;background:#0e1526}
@media(max-width:980px){.grid{grid-template-columns:1fr 1fr}.fid-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.compare-panels{grid-template-columns:1fr}}@media(max-width:660px){.grid{grid-template-columns:1fr}body{padding:18px 14px}header{display:block}.pill{display:inline-block;margin-top:10px}.section-head{display:block}.section-head p{margin-top:4px}.compare-row{grid-template-columns:65px repeat(4,minmax(0,1fr))}}
</style></head><body>
<header><div><h1>FastGen 학습 현황</h1><p>서버 실행 기록 · 5초마다 자동 갱신 · 읽기 전용</p></div><div id="updated" class="pill">연결 중…</div></header>
<div id="gpu" class="gpu"></div>
<section class="card fid-board" aria-label="모델별 FID"><div class="section-head"><h2>모델별 FID</h2><p>자체 평가 · 낮을수록 좋음 · 공식 제출 점수 아님</p></div><div class="fid-grid">
<div class="fid-item" data-fid="vae-1"><span>VAE · 1회 생성</span><strong>—</strong><small>평가 결과 읽는 중</small></div>
<div class="fid-item" data-fid="gan-1"><span>GAN · 1회 생성</span><strong>—</strong><small>평가 결과 읽는 중</small></div>
<div class="fid-item" data-fid="soflow_v2-1"><span>SoFlow v2 · 1회 생성</span><strong>—</strong><small>평가 결과 읽는 중</small></div>
<div class="fid-item" data-fid="soflow_v2-4"><span>SoFlow v2 · 4회 생성</span><strong>—</strong><small>평가 결과 읽는 중</small></div>
</div></section><main class="grid">
<section class="card" data-model="vae"><div class="top"><h2>VAE</h2><span class="state">확인 중</span></div><div class="progress-line"><strong class="epochs">—</strong><span class="small percent"></span></div><div class="track"><div class="fill"></div></div><div class="batch"></div><div class="metric"><select aria-label="VAE 손실 지표"></select><strong class="metric-value">—</strong></div><canvas aria-label="VAE 손실 곡선"></canvas><div class="sample-title">최근 샘플</div><img class="sample" alt="VAE 최근 생성 샘플"><div class="foot"></div></section>
<section class="card" data-model="gan"><div class="top"><h2>GAN</h2><span class="state">확인 중</span></div><div class="progress-line"><strong class="epochs">—</strong><span class="small percent"></span></div><div class="track"><div class="fill"></div></div><div class="batch"></div><div class="metric"><select aria-label="GAN 손실 지표"></select><strong class="metric-value">—</strong></div><canvas aria-label="GAN 손실 곡선"></canvas><div class="sample-title">최근 샘플</div><img class="sample" alt="GAN 최근 생성 샘플"><div class="foot"></div></section>
<section class="card" data-model="soflow_v2"><div class="top"><h2>SoFlow v2</h2><span class="state">확인 중</span></div><div class="progress-line"><strong class="epochs">—</strong><span class="small percent"></span></div><div class="track"><div class="fill"></div></div><div class="batch"></div><div class="metric"><select aria-label="SoFlow 손실 지표"></select><strong class="metric-value">—</strong></div><canvas aria-label="SoFlow 손실 곡선"></canvas><div class="sample-title">최근 EMA 샘플</div><img class="sample" alt="SoFlow 최근 EMA 생성 샘플"><div class="foot"></div></section>
</main>
<section class="card compare" aria-label="SoFlow 생성 횟수별 비교"><h2>SoFlow v2 · 1회와 4회 비교</h2><p id="compare-note" class="compare-note">평가 결과를 확인하는 중…</p>
<div class="compare-panels"><div class="compare-panel" data-nfe="1"><div class="compare-panel-head"><h3>1회 생성</h3><span class="compare-fid">FID —</span></div><div class="compare-sheet"></div></div>
<div class="compare-panel" data-nfe="4"><div class="compare-panel-head"><h3>4회 생성</h3><span class="compare-fid">FID —</span></div><div class="compare-sheet"></div></div></div></section><script>
const names={vae:'VAE',gan:'GAN',soflow_v2:'SoFlow v2'};
const fmt=(x,d=3)=>Number.isFinite(x)?Number(x).toFixed(d):'—';
const duration=s=>Number.isFinite(s)?`${Math.floor(s/3600)}시간 ${Math.floor(s%3600/60)}분`:'—';
function buildComparison(){
  const categories=[['피카츄','0103'],['파이리','0014'],['꼬부기','0129'],['이상해씨','0009']];
  for(const nfe of [1,4]){
    const sheet=document.querySelector(`[data-nfe="${nfe}"] .compare-sheet`);
    for(const [name,id] of categories){
      const row=document.createElement('div');row.className='compare-row';
      const label=document.createElement('span');label.textContent=name;row.append(label);
      for(let i=0;i<4;i++){
        const img=document.createElement('img');
        img.src=`/eval-sample/soflow_v2/${nfe}/${id}_${String(i).padStart(2,'0')}.png`;
        img.alt=`${name} ${i+1}번, ${nfe}회 생성`;img.loading='lazy';row.append(img);
      }
      sheet.append(row);
    }
  }
}
function renderFids(results){
  const byKey=new Map(results.map(r=>[`${r.model}-${r.nfe}`,r]));
  for(const item of document.querySelectorAll('.fid-item')){
    const record=byKey.get(item.dataset.fid);
    item.querySelector('strong').textContent=record?fmt(record.fid,2):'—';
    item.querySelector('small').textContent=record?'2026-09-29 서버 실행':'평가 결과 없음';
  }
  for(const nfe of [1,4]){
    const record=byKey.get(`soflow_v2-${nfe}`);
    document.querySelector(`[data-nfe="${nfe}"] .compare-fid`).textContent=`FID ${record?fmt(record.fid,2):'—'}`;
  }
  const one=byKey.get('soflow_v2-1'),four=byKey.get('soflow_v2-4');
  document.getElementById('compare-note').textContent=one&&four&&one.checkpoint_sha256===four.checkpoint_sha256&&one.seed===four.seed&&one.weight_key==='ema_model'&&four.weight_key==='ema_model'
    ?'같은 200에폭 EMA 체크포인트와 시드 사용 · 같은 행과 열은 동일한 카테고리·샘플 번호입니다.'
    :'두 평가의 모델과 시드 정보를 확인할 수 없습니다.';
}
function plot(canvas,rows,key){
  const width=canvas.clientWidth||300,height=150,dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(width*dpr);canvas.height=Math.round(height*dpr);
  const c=canvas.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,width,height);
  const values=rows.map(r=>r[key]).filter(Number.isFinite);if(!values.length)return;
  let lo=Math.min(...values),hi=Math.max(...values);if(hi===lo){hi+=1;lo-=1}
  const left=46,right=width-8,top=10,bottom=height-24;
  c.strokeStyle='#30405c';c.lineWidth=1;c.fillStyle='#8fa0bf';c.font='11px ui-sans-serif, sans-serif';
  for(let i=0;i<3;i++){const y=top+(bottom-top)*i/2;c.beginPath();c.moveTo(left,y);c.lineTo(right,y);c.stroke();c.fillText(fmt(hi-(hi-lo)*i/2,2),2,y+4)}
  c.fillText(`1`,left,bottom+17);c.fillText(`${rows[rows.length-1].epoch}`,Math.max(left,right-28),bottom+17);
  c.beginPath();values.forEach((v,i)=>{const x=left+(right-left)*i/Math.max(1,values.length-1),y=bottom-(v-lo)/(hi-lo)*(bottom-top);i?c.lineTo(x,y):c.moveTo(x,y)});
  c.strokeStyle='#72aaff';c.lineWidth=2;c.stroke();
}
function renderRun(r){
  const card=document.querySelector(`[data-model="${r.model}"]`),state=card.querySelector('.state');
  card._run=r;
  state.textContent=({completed:'완료',running:'학습 중',failed:'실패'})[r.status]||r.status;
  state.className='state '+r.status;
  if(r.status==='running'&&r.process_alive===false&&r.log_age_seconds>60){state.textContent='확인 필요';state.className='state stale'}
  const target=Number(r.target_epochs)||0,epoch=Number(r.epoch)||0,pct=target?Math.min(100,epoch/target*100):0;
  card.querySelector('.epochs').textContent=`${epoch} / ${target||'—'} 에폭`;
  card.querySelector('.percent').textContent=target?`${pct.toFixed(1)}%`:'';
  card.querySelector('.fill').style.width=`${pct}%`;
  card.querySelector('.batch').textContent=r.batch?`현재 배치 ${r.batch.current} / ${r.batch.total} · ${Math.round(r.batch.current/r.batch.total*100)}%`:'';
  const sel=card.querySelector('select'),selected=sel.value||r.metrics[0];
  if(sel.options.length===0){for(const metric of r.metrics){const opt=document.createElement('option');opt.value=metric;opt.textContent=metric.replaceAll('_',' ');sel.append(opt)}sel.onchange=()=>{const current=card._run;plot(card.querySelector('canvas'),current.history,sel.value);card.querySelector('.metric-value').textContent=fmt((current.history.at(-1)||{})[sel.value],4)}}
  sel.value=r.metrics.includes(selected)?selected:r.metrics[0];
  const last=r.history.at(-1)||{};card.querySelector('.metric-value').textContent=fmt(last[sel.value],4);
  plot(card.querySelector('canvas'),r.history,sel.value);
  const img=card.querySelector('img');
  if(r.sample){const src=`/sample/${r.model}?v=${r.sample.mtime_ns}`;if(img.getAttribute('src')!==src)img.src=src;
    card.querySelector('.sample-title').textContent=r.model==='soflow_v2'
      ?`학습 중 저장한 1회 생성 ${r.sample.kind==='ema'?'EMA ':''}샘플 · ${r.sample.epoch} 에폭`
      :`최근 ${r.sample.kind==='ema'?'EMA ':''}샘플 · ${r.sample.epoch} 에폭`;
  }else{img.removeAttribute('src');card.querySelector('.sample-title').textContent='샘플 없음'}
  const bits=[`실행: ${r.run_id}`];if(Number.isFinite(last.peak_vram_gb))bits.push(`최대 VRAM ${fmt(last.peak_vram_gb,2)} GiB`);
  if(r.status==='completed')bits.push(`총 ${duration(r.wall_seconds)}`);
  if(r.status==='running'&&r.log_age_seconds!=null)bits.push(`로그 갱신 ${r.log_age_seconds}초 전`);
  card.querySelector('.foot').textContent=bits.join(' · ');
}
async function refresh(){
  try{const response=await fetch('/api/status',{cache:'no-store'});if(!response.ok)throw Error(`HTTP ${response.status}`);
    const data=await response.json();for(const r of data.runs)renderRun(r);renderFids(data.fid_results||[]);
    document.getElementById('updated').textContent=`갱신 ${new Date(data.updated_utc).toLocaleTimeString('ko-KR')}`;
    const gpu=document.getElementById('gpu');gpu.replaceChildren();
    if(data.gpu){const info=[data.gpu.name,`GPU ${data.gpu.utilization_percent}%`,`VRAM ${data.gpu.memory_used_mib} / ${data.gpu.memory_total_mib} MiB`];if(data.gpu.temperature_c!==null)info.push(`온도 ${data.gpu.temperature_c}°C`);for(const text of info){const span=document.createElement('span');span.textContent=text;gpu.append(span)}}
  }catch(e){document.getElementById('updated').textContent=`연결 오류 · ${e.message}`}
}
buildComparison();refresh();setInterval(refresh,5000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self.send_bytes(PAGE.encode("utf-8"), "text/html; charset=utf-8")
        elif path == "/api/status":
            payload = json.dumps(status_snapshot(), ensure_ascii=False, allow_nan=False)
            self.send_bytes(payload.encode("utf-8"), "application/json; charset=utf-8")
        elif path.startswith("/sample/"):
            model = path.removeprefix("/sample/")
            selected = latest_sample(RUNS[model]) if model in RUNS else None
            if not selected:
                self.send_error(404, "Sample not found")
                return
            try:
                self.send_bytes(selected[0].read_bytes(), "image/png")
            except OSError:
                self.send_error(404, "Sample not found")
        elif path.startswith("/eval-sample/soflow_v2/"):
            parts = path.removeprefix("/eval-sample/soflow_v2/").split("/")
            if len(parts) != 2 or parts[0] not in ("1", "4") or not COMPARE_SAMPLE_NAME.fullmatch(parts[1]):
                self.send_error(404, "Evaluation sample not found")
                return
            sample_path = FID_RUNS[("soflow_v2", int(parts[0]))] / "generated" / parts[1]
            try:
                if sample_path.stat().st_size > 1_000_000:
                    raise OSError("Evaluation sample is too large")
                self.send_bytes(sample_path.read_bytes(), "image/png")
            except OSError:
                self.send_error(404, "Evaluation sample not found")
        else:
            self.send_error(404, "Not found")

    def log_message(self, format: str, *args: object) -> None:
        pass  # Five-second polling should not create an unbounded access log.


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.daemon_threads = True
    print(f"FastGen dashboard listening on http://127.0.0.1:{args.port}/", flush=True)
    server.serve_forever(poll_interval=1)


if __name__ == "__main__":
    main()
