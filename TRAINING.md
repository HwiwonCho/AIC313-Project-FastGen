# FastGen deadline pipeline

## Execution speed and full-state continuation (October 26)

Everything adopted is implemented here: bounded CPU batch preparation in
`fastgen/prefetch.py`, full-state continuation in `fastgen/continuation.py`,
the launcher in `tools/start_tmux.py`, and the dashboard in `tools/status.py`.
No installed compiler headers, runtime monkeypatches or extra packages are
required. `execution.soflow_prefetch_batches=2` starts one CPU process and queues
at most two batches. It preserves the consumed image/augmentation RNG position;
unconsumed lookahead is regenerated after a restart. GPU batch16, FP32
parameters/Adam, BF16 forwards, architecture, loss, learning rate, EMA, data,
augmentation and the peer FID protocol are unchanged.

The paired 32-update server check from the same step1729 checkpoint measured
median0.50825s without prefetch and0.35254s with it (1.44x). Both cases used
5,660,873,216 bytes peak allocated and5,909,774,336 reserved. Model, EMA, Adam,
all RNGs, consumed loader state and actual augmented batch fingerprints matched
exactly. This measures throughput, not FID. Probe weights were discarded.
cuDNN benchmarking was slower; `torch.compile` could not build without Python
headers. Neither is enabled and no packages were installed for them.

For an already stopped teacher, the following code-managed continuation keeps
all model/EMA/Adam/RNG/data state and the current global update. It adopts the
completed pilots and their original history without training them again:

```bash
python tools/start_tmux.py --python /root/AIC313-Project-FastGen/.venv/bin/python \
  --data-root /root/AIC313-Project-FastGen/data/pokemon-generation-one-22k \
  --run-root /root/FastGen/runs/e2e_20261026_prefetch \
  --continue-from /root/FastGen/runs/e2e_20261026_batch16 \
  --session fastgen --reuse-session
```

`--reuse-session` refuses to replace an active training pane. For a new session,
omit it. A fresh reproduction omits `--continue-from` and uses a fresh output
directory; it runs the complete augmentation -> teacher -> pairs -> student ->
export -> official evaluation sequence. To resume that same version after a
cooperative stop, repeat the recorded `pipeline.py run` command from launch.json
in the same output directory. Source/config hashes must still agree.

The continuation accepts only execution/source metadata changes. It rejects
changed learning/data/FID settings, preserves the source checkpoint, verifies
every non-identity checkpoint field after save/reload, and writes the proof to
`adopted_history/teacher_full_state_continuation.json`. The 1729 checkpoint is
continued at1730; there is no EMA-only initialization or rewind to best.
Prior executions, source snapshots and costs remain in submission records.
The 20GB lifetime guard and18GB allocator cap remain enabled. The actual
completed pipeline's peaks, timing and FID still depend on the later stages.

The original starter README, `.orca/`, `assets/`, fixed data split files and
source paths remain in this working tree. This separate guide documents our
added pipeline. `pipeline.py check` checks every initial starter path and the
unchanged files against their original bytes, as well as the base Model class.

For a persistent server run with a terminal monitor:

```bash
python tools/start_tmux.py --python /root/AIC313-Project-FastGen/.venv/bin/python \
  --data-root /root/AIC313-Project-FastGen/data/pokemon-generation-one-22k \
  --run-root /root/FastGen/runs/e2e_20261026 --session fastgen
tmux attach -t fastgen
```

Window 0 is training and window 1 is status. Ctrl+B then D detaches without
stopping training. `console.log`, stage `metrics.jsonl`/`summary.json` and loss
curves remain on disk. The monitor reads files and GPU status; it imports no
Torch and allocates no GPU memory. TensorBoard is not an added dependency.

Existing Documents checkout and legacy experiments are retained. This code
contains no SSH, upload, background scheduler or
automatic connection. Real training is started only by the `run` command.

## Fixed default run

1. Audit only the 17,079 official train files. Hash white-composited bilinear64
   RGB pixels and choose 5 unique single-label groups/class with seed42.
2. Run exactly two pilots, flip and weak spatial, 3,000 updates each. Identical
   initialization/image order; the 755 reference groups are excluded as whole
   groups from both pilots. Spatial must improve FID by at least 2 to be chosen;
   otherwise retain flip. This is a predeclared practical threshold, not a
   statistical significance claim or proof of long-run superiority.
3. Discard pilot weights. Train a fresh 95.52M SoFlow on ALL train files, using
   the selected augmentation: 200K updates, FM75/TC25, p1, LR1e-4, warmup4270,
   cosine horizon170800/floor.1, AdamW(.9,.99), clip10, EMA.9999.
4. Generate 512 balanced pairs/class using the frozen best teacher EMA and
   deterministic four-call solution transport (`solution4`, FP32). Quantize z to FP16 BEFORE teacher inference;
   store z/target FP16 shards128. No independent augmentation of paired targets.
   The submitted four-call sampler still uses linear re-noising. Distillation
   targets use deterministic transport so the one-call MSE does not average
   over independent extra teacher noise. This is a deliberate training mapping
   choice, not a claim that its FID is the historical re-noising score.
5. Warm-start a student from the exact teacher EMA. Train the one-call endpoint
   MSE for 25K updates, LR1e-5/warmup1000/cosine floor.1. The teacher is not kept
   on GPU during student training. Optional reflow is implemented but OFF.
6. Export EMA-only `checkpoints/one_nfe.ckpt` and `checkpoints/few_nfe.ckpt`.
   Both use the fixed base loader; the first uses exactly one network call,
   the second uses four. No runtime CFG or learned auxiliary component.
7. Freeze hashes, check actual NFE/parameter counts, then execute the unmodified
   official evaluator separately for each mode. Validation never selects a
   checkpoint, augmentation, sampler or training schedule.

## Run later on the server

Use the TA requirements in an existing environment. No extra dependency is added.

```bash
python pipeline.py check
python pipeline.py plan
python pipeline.py run --config configs/deadline.json \
  --data-root /root/AIC313-Project-FastGen/data/pokemon-generation-one-22k \
  --run-root /root/FastGen-runs/e2e_20261026
```

The shorter server entry is `bash run.sh DATA_ROOT RUN_ROOT`. It exposes one GPU
by default. Before the pilot, a separate preflight verifies the environment,
BF16 support, parameter count and provided FID Inception asset. `status.json`
shows the current stage; `console.log` can be supplied by the detached launcher.

The data root must already contain the original images; the training pipeline
does not download or rewrite them. Canonical fixed split files are included.
Use a fresh run root for changed settings. Repeating the SAME command resumes
the exact current checkpoint and skips completed stages. Ctrl+C/SIGTERM requests
a checkpoint at the next complete optimizer update and stops the chain. Partial
updates are never saved as completed recovery states.

The default effective batch is16, micro16 x accumulation1 for the teacher/student.
The two completed pilots use micro8 x accumulation2, explicitly fixed in the pilot config.
A finite server probe measured 0.957s/update for8x2 and0.458s/update for16x1
(8 timed updates each;20GB cap checked). No long-run throughput or FID guarantee
is inferred. Fused/channels-last measured0.431s but was not selected because its
small extra gain did not justify additional execution changes.
The old636-update teacher checkpoint is retained; the faster teacher starts
from scratch and does not adopt probe weights. Before the first run,
micro4 x accumulation4 can be configured if necessary; the two protocols are
not claimed bitwise equivalent. CUDA weights/Adam are FP32; training forward is
BF16. GPU allocation is capped with a 10% margin below min(device memory,20GB).
Each fresh stage checks lifetime `max_memory_allocated` against 20,000,000,000
bytes without resetting peaks after evaluation. Reserved memory is also logged.
OOM fails the stage and preserves the previous complete checkpoint; it never
silently changes batch size/precision. **GPU peak and throughput require a real
server run; CPU checks cannot certify them.** First-step and periodic durations
are logged so the training budget can be assessed without a parameter sweep.

## Same FID conditions as Yunseo

`fastgen/metrics.py` matches the inspected defaults of Yunseo's `scaled_train.py`
at `a80bbd0`:

| Item | Shared protocol |
|---|---|
| Reference | TRAIN, 5 unique single-label pixel groups/class, 755 images |
| Reference selection | seed42, sorted group hashes, same shuffle/order |
| Generation | classes0..150, 5 consecutive images/class, 755 images |
| Noise seed / batch | 27182 / 8 |
| Sampling precision | FP32; matmul TF32 off, cuDNN TF32 on, benchmark off |
| PNG | identical tensor round-trip for references; clamp and `to_pil_image` truncation for outputs |
| FID | pooled Clean-FID, default clean mode, batch8, workers4 |

The final teacher/student train on the reference images too, matching Yunseo's
overlapping TRAIN proxy. Pilot scores are explicitly labeled held-out TRAIN
scores, so they are not presented as the same in-sample training condition.
Our model/sampler stays our own: 1 or4 calls. Yunseo's guided32-step teacher is
64 NFE and must not be compared as if it were4 NFE.

Defaults in code do not prove the arguments Yunseo used on his server. To verify
the exact reference IDs/order/pixel hashes, exchange his data `manifest.json`:

```bash
python peer_fid.py --checkpoint runs/deadline/submission/checkpoints/few_nfe.ckpt \
  --mode few_nfe --data-root /path/to/data --peer-manifest /path/to/yunseo/manifest.json \
  --output runs/peer_comparison
```

Match software versions and the immutable data contents as well; environment
versions and reference PNG hashes are recorded. The official validation3020
result is a separate metric with a separate name. Historical 37.205/64.383 used
another seed/protocol and cannot be claimed as this TRAIN755 score.

## Source layout

`pipeline.py` is the short command entry point; `run.sh` is the server launcher.
Under `fastgen/`, `orchestration.py` controls the finite sequence, `engine.py`
trains, `data.py`/`pairs.py` prepare batches, `metrics.py` owns the peer FID
protocol, and `submission.py` exports/packages/runs the provided evaluator.
Architectures remain under `networks/`. All run outputs stay under `RUN_ROOT`.

## Submission and records

The completed `RUN_ROOT/submission/` is self-contained: both required checkpoints,
all architecture/training/data/pair/export/evaluation code, requirements, canonical
splits, citations and records. Fixed official files and base Model are checked
against their source hashes/AST. The evaluator receives a private lossless opaque
RGB validation view, so its downloader cannot rewrite the original dataset.

Recorded automatically:

- Code snapshot/hashes, complete config, source and dataset identities, environment.
- Optimizer/EMA/model and Python/Torch/data/augmentation RNG state in current/best.
- Loss and gradient/LR/timing logs, loss curves, train-proxy scores and references.
- Eight individual early images and their grid, retained from step1 of each stage.
- Lifetime allocated/reserved VRAM peaks, failures and resume sessions.
- Pair source/protocol/shard hashes; no stale teacher-pair mixing.
- Frozen submission checkpoints, actual NFE checks, original evaluator logs.
- `writeup_facts.json`: stage results and sequential child wall time, including
  attempts. Nested timing scopes are labeled to avoid double-counting.

Only current, best and one temporary atomic checkpoint are kept per stage.
Previous experiment folders/checkpoints are never deleted. An explicit
`--continue-from` adopts only the compatible full training state and history.
The generated records support the required two-page write-up; they are not a
claim that the final write-up has already been written.

## Local CPU verification

```bash
python -m unittest discover -s tests -v
```

CPU tests use tiny synthetic training/data and no Inception download, plus real
95.52M random weights for export and isolated packaged inference. They exercise
the full controller/packaging, training/resume, pair coupling, warm-started
distillation, PNG protocol, strict loader and actual 1/4 backbone calls.
Actual FID quality and20GB GPU compliance
remain unmeasured until the server run.
