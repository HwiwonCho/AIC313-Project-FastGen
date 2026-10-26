# Code and method provenance

- **Official FastGen skeleton**: KAIST Visual AI Group,
  https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen .
  Fixed files were fetched on 2026-10-26; `official_files.json` records their hashes.
  The base `Model` class is unchanged. Both student subclasses are implemented here.
- **SoFlow**: https://github.com/zlab-princeton/SoFlow and
  https://arxiv.org/abs/2512.15657 . The existing Hwiwon project adaptation supplies
  solution-map parameterization, velocity/consistency objectives and time sampling.
  Four-call endpoint prediction plus linear re-noising follows SoFlow inference.
- **EDM / SongUNet architecture**: Karras et al., *Elucidating the Design Space of
  Diffusion-Based Generative Models*, https://github.com/NVlabs/edm .
  `networks/soflow_unet.py` preserves the NVIDIA copyright and CC BY-NC-SA 4.0
  notice. Scalar NumPy arithmetic is replaced with Python `math`; no new runtime
  dependency is required. See `network_provenance.json` for exact source hashes.
- **Existing Hwiwon training recipe**: local FastGen lab and preserved server
  runtime-v5 assets. This package retains the 95,524,611-parameter architecture,
  FM75/TC25, p=1, CFG3/mix.25/null.1 training, reference time distributions,
  EMA .9999 and the historical 170800-update LR horizon. It is a NEW scratch run:
  full train data, chosen augmentation, batch partition and FP32 metric sampling
  differ from the historical FID37.205 measurement. No historical score is claimed
  for these new untrained outputs.
  Distillation uses deterministic four-call solution transport from the same
  teacher, while the few-NFE submission keeps stochastic linear re-noising.
  These are logged as distinct samplers and their historical scores are not mixed.
- **Team pipeline ideas and peer FID protocol**: Yunseo Park's FastGen branch
  `exp/32stepODE-scaled`, commit `a80bbd0` (2026-10-24),
  https://github.com/yunseopark-kaist/FastGen/tree/a80bbd0 .
  We reference its white-composited resized-pixel duplicate definition,
  five unique single-label groups/class, seed42 reference selection, FP16 pair
  storage with pre-quantized teacher noise, EMA warm start, optional reflow and
  endpoint distillation. The portable training/orchestration code here is newly
  written; the teammate's 98.4M backbone is not imported.
- **Shared weak spatial augmentation**: Hwiwon's
  `codex/augmentation-utils` commit `5be41310cea89bc34ccd777096609b0d170d2ceb`,
  https://github.com/HwiwonCho/AIC313-Project-FastGen/tree/5be41310cea89bc34ccd777096609b0d170d2ceb .
  Ranges match the team policy: flip .5, angle +/-5 degrees, translation +/-3,
  scale .9--1.05, bilinear/white fill. This module uses PIL images before tensors;
  it does not claim bitwise identity to the earlier tensor wrapper.

The final write-up must cite the above sources and describe the actual selected
augmentation and distillation path. No external pretrained generative weights,
perceptual loss network or additional learned inference stage is used. The
provided Clean-FID Inception is used only for evaluation.
