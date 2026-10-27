# SoFlow experiments — 2026-10-27

## Frozen model

SongUNet SoFlow, 95,524,611 parameters. Current comparison snapshot: EMA at
135,000 updates. A second validation pass freezes the server's TRAIN755-selected
best at 175,000 updates. Training starts from scratch on all 17,079 provided train files.
No separate teacher was used to train this four-NFE snapshot. The 32-NFE check
uses the same EMA backbone; no student has been trained in these experiments.

Training: FM75/consistency25, CFG3 training targets, null probability .1,
mix .25, AdamW1e-4, betas(.9,.99), weight decay0, clip10, EMA max .9999,
batch16, BF16 forward with FP32 weights/optimizer. Warmup4,270; cosine horizon
170,800; minimum LR1e-5; target200,000 updates.

Augmentation: flip .5, rotation +/-5 degrees, translation +/-3 pixels,
scale .9–1.05, bilinear interpolation and white fill.
Class-balanced replacement sampling; unique-group sampling is disabled.

## Measured results

| Experiment | NFE | Reference/generated images | FID |
|---|---:|---|---:|
| Pilot flip, 3k updates | 4 | TRAIN internal reference755 / generated755 | 225.287974 |
| Pilot spatial, 3k updates | 4 | TRAIN internal reference755 / generated755 | 200.103306 |
| Current EMA135k | 4 | TRAIN755 / generated755 | 67.732711 |
| Historical EMA188k, matched local protocol | 4 | TRAIN755 / generated755 | 63.173985 |
| Current EMA135k, submission interface | 4 | validation3020 / generated3020 | 38.613173 |
| TRAIN755-selected best EMA175k, submission interface | 4 | validation3020 / generated3020 | 39.337681 |
| One balanced subset of that validation pass | 4 | validation755 / generated755 | 67.008437 |
| Current EMA135k, quadratic re-noising grid | 4 | TRAIN755 / generated755 | 75.626440 |
| Current EMA135k, Euler32 diagnostic | 32 | TRAIN755 / generated755 | 62.814648 |

The 32-NFE row is a teacher diagnostic, not a four-NFE submission result.
The validation755 check reuses generated PNGs; it performs no new model forwards.
Full numeric records and image-call counts are in `records/`.
The 175k validation pass uses the same seed, batch size, sampler and local runtime
as the 135k pass. Its TRAIN755 selection score is 67.150635. The validation score
increased by 0.724508; the training selection rule was not changed.

TRAIN755: five unique reference groups/class, seed27182, FP32, batch8,
re-noising extra seed20261008. Validation3020: twenty generated images/class,
seed20261002, FP32, batch8, unchanged `ModelFewNFE.sample` with its default noise
streams. Both use clean-fid0.1.35 and the same Inception weights.
Windows image loaders use zero workers; the provided reference/generation
functions and source files are unchanged. These are local self-evaluation scores.
White compositing was also checked against direct RGB conversion for all 3,020
original validation images: the resized RGB pixels are identical for every image.

Local runtime: Torch2.9.0+cu128, torchvision0.24.0+cu128, Pillow12.3.0,
clean-fid0.1.35, numpy2.2.6, scipy1.15.3, RTX5070Ti.
Training runtime: Torch2.6.0+cu124 on the server. Re-scoring the identical
server PNGs locally changed FID67.749717 to67.710494.

## Replay

Training entry point: `bash run.sh` with `configs/deadline.json`.
The evaluation scripts below add path arguments to the scripts used for the
recorded experiments; the numerical methods are retained. Original executed
script hashes are in `records/executed_script_hashes.json`.

Keep checkpoints and image archives outside Git. The frozen server inputs are in
`/root/FastGen/comparisons/train755_20261027/`:

```text
current_few_nfe.ckpt       # EMA135k, SHA2469bf3458feb223d30de8e2d974c47a09f6153df9b582bffd1224de02c45db6
few_nfe.ckpt               # historical EMA188k, SHA81794ed4ec770a2d4fea1086cf886d228475fb6d844cf3967b865809a4b937ed
evaluation_source.tar.gz
reference_data.tar.gz
validation3020_data.tar.gz
```

Use an evaluation GPU separate from the running trainer. Set `WORK` to a writable
folder containing those inputs and `CACHE` to the folder containing
`inception-2015-12-05.pt`. Output folders must be fresh.

```bash
SCRIPTS=experiments/20261027/scripts
python "$SCRIPTS/run_comparison.py" prepare --work-root "$WORK" --metric-cache "$CACHE"
python "$SCRIPTS/run_comparison.py" compare --work-root "$WORK" --metric-cache "$CACHE"
python "$SCRIPTS/evaluate_validation3020.py" --work-root "$WORK" --metric-cache "$CACHE"
python "$SCRIPTS/check_sample_count.py" --work-root "$WORK" --metric-cache "$CACHE"
python "$SCRIPTS/check_fast_paths.py" --work-root "$WORK" --metric-cache "$CACHE"
```

The first command extracts the frozen source under `$WORK/project`. This preserves
the evaluated source version independently of the repository checkout.
For the 175k pass, use the checkpoint and config in
`/root/FastGen/comparisons/validation3020_20261027/latest_best/`, together with
the same data archive and frozen evaluation source. The evaluator extracts only
data/source entries from the archive, preserving the selected checkpoint config.
The architecture inspector accepts `--our-root`, `--peer-root`, and `--output`;
`--peer-root` should contain Yunseo's `scaled_network.py` from commita80bbd0.
It constructs meta-device models and performs no CUDA work.

## Sources

- [Training configuration](../../configs/deadline.json)
- [SoFlow adaptation](../../networks/soflow_model.py), [SongUNet](../../networks/soflow_unet.py)
- [Attribution](../../CITATIONS.md)
- [Yunseo scaled branch](https://github.com/yunseopark-kaist/FastGen/tree/a80bbd0adbaff5ac81f7d9330955ac5e0a1b8421)
