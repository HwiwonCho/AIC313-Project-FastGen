# FastGen: conditional StyleGAN2-ADA, 1NFE



This branch contains the **actual 99,054,136-parameter G+D architecture**, freshly
trained from seed 42. G: 64,005,784; D: 35,048,352; frozen EMA: 64,005,784.
The previous 198.75M run is not a submission candidate and its weights are not used.
`ModelFewNFE` remains unimplemented here; merge the team's separate few-NFE work
before assembling a submission requiring both checkpoints.

### Reproduce the measured score

Install the official requirements and the course PyTorch 2.6 / torchvision 0.21
CUDA environment. No additional CUDA extension compilation is required. NumPy,
SciPy and requests are transitive dependencies of the provided packages.
Actual server: Python 3.12.3, torch 2.6.0+cu124, torchvision 0.21.0+cu124.
See verification/policy_audit_20261030.json for observed package versions.

Download the one-NFE checkpoint from the shared submission-materials folder:
https://drive.google.com/drive/folders/1h1NHtZsqkPKZHIiIXX3RkuzJ3HhqslMD
Place it at `checkpoints/one_nfe.ckpt`. It is NOT stored in Git.
SHA256: `4c29d11c50be9af2963181a7c58cdaec79d621dcd6b66b1f17f01e7dbac5eeed`.

```sh
python evaluate.py --model_checkpoint checkpoints/one_nfe.ckpt --evaluate_mode one_nfe --data_root data/pokemon-generation-one-22k --split_dir data/pokemon-generation-one-22k --reference_dir results/val-reference --output_dir results/val-generated --batch_size 8 --device cuda
```

For the fixed seed (20261029), fresh output directories, and a per-batch one-G-call
assertion, run the wrapper below. It executes the UNMODIFIED evaluate.py CLI via
runpy; it does not replace its FID implementation or filter generated samples.

```sh
python -m stylegan2.evaluate_official --checkpoint checkpoints/one_nfe.ckpt --data-root data/pokemon-generation-one-22k --output results/audit-108500
```

Step 108500: **31.9207** from the official CLI; earlier function-based evaluation:
31.9208749. Both use validation 3020 / generated 3020, 151 classes x 20, batch 8,
seed 20261029, 1NFE. Small GPU numerical differences are possible. This is our
self-evaluation, NOT a TA-measured leaderboard score. See
verification/official_cli_108500.json. Same-step validation for the prior
198.75M run is unavailable. Its 40.3475 at step 50000 is not a matched comparison.

### Training and organization

`stylegan2/models.py`: architecture; `recipe.py`: updates/EMA/ADA;
`data.py`: TRAIN-only loader; `train.py`: training/checkpoints;
`recording.py`: every-step loss/time/early images; `evaluate_official.py`: evaluation;
`vendor/` and `SOURCES.md`: pinned NVIDIA sources, licenses and adaptations.
Unused VAE/SoFlow programs and historical experiments are excluded.
Protected skeleton files and Model base class match official commit
3ccf3261495bd144bf844b7f298d10406b53117a (line-ending-normalized comparison).
requirements.txt is restored to that official package list.

```sh
# New training only; never use this to modify the currently running experiment.
python -m stylegan2.train --data-root data/pokemon-generation-one-22k --run runs/new-experiment --steps 125000 --save-every 500 --sample-every 500 --eval-every 250000 --seed 42 --fast-regularization --export-current
python -m stylegan2.verify
```

Training reads only provided TRAIN (17079), with duplicate-group fit/holdout split
13663/3416 and class-balanced replacement sampling. No pretrained generator,
teacher, external training images, or validation gradients. Inception is evaluation
only. Batch32/micro8, logistic loss, R1/16, PL/4, ADA bgc, EMA10k images.
Gradient accumulation is averaged by micro/batch (an explicit adaptation from
upstream summed gradients). No change was made to the running trainer for this
publication. Adding the evaluation wrapper changes source_identity: continue a
running checkpoint only with its original source snapshot, not this new branch.

Training selects its export using internal TRAIN FID. Validation-based human
checkpoint selection is allowed by the TA clarification supplied by the team;
validation is not used in training. Internal FID and validation FID are distinct.
G+D <=100M is enforced, following the supplied TA clarification on joint models.
Observed peak allocated VRAM and contiguous loss logging are in the policy audit.
The run is still ongoing; time/log evidence is a snapshot, not final total time.

### Mid-term submission materials

Drive folder 01 contains only this code and the verified one-NFE checkpoint
(parts plus a stdlib restore script). Folder 02 separately preserves loss CSV,
loss curve, early8 samples and time/config/evaluation evidence for the final submission.
No report draft is included in the current Drive package. The mid-term requires
source and checkpoint; the final submission additionally needs the team's write-up.
The real team ID and the separate few-NFE contribution remain the team's responsibility.
The run is ongoing; preserved time/log snapshots are not final training totals.

### Reused versus adapted code

Official evaluate.py, dataset.py, downloader, Model base and fixed manifests stay
unchanged. FID is computed by the provided clean-fid call, not a custom metric.
NVIDIA mapping/synthesis/discriminator blocks, loss and ADA are reused. Local
adaptations are the small-data channel configuration, TRAIN-only cached loader,
training/recording loop, and portable operators. Bilinear double-backward,
modulated convolution and modern ATen gradfix are compatibility/performance
adaptations, not claims of bitwise equivalence. Activation/filtering use NVIDIA's
existing reference implementations. Data RGB/resize is duplicated in the cached
TRAIN loader intentionally; it matches the official path but could be consolidated
in a future experiment. The NumPy filter-bank initialization replaces an existing
SciPy operation although SciPy is already a clean-fid dependency; it is not needed
for dependency reduction and the image-filter augmentation is disabled. We retain
this historical adaptation to avoid changing the trained implementation during
submission preparation. See SOURCES.md and numerical verification modules.
