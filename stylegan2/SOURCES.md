# 출처 및 변경 범위

## Skeleton

- 공식 과제: https://generative-models.kaist.ac.kr/fastgen/
- 공식 base repository: https://github.com/KAIST-Visual-AI-Group/AIC313-Project-FastGen
- 실제 복사 원본: 공식 skeleton을 유지한 기존 로컬 실험 사본
- `dataset.py`, `download_dataset.py`, `evaluate.py`, `requirements.txt`, `src/*.py`, 고정 split 및 category mapping을 원본 그대로 복사했다. 이미지와 체크포인트는 복사하지 않았다.
- `model.py`의 `Model` 기본 클래스 및 `ModelFewNFE`는 보존했다. `ModelOneNFE`만 새 생성기에 연결했다. 기존 VAE 정의는 원본 파일의 일부이며 제출 모델에 등록되지 않는다.

## 공식 StyleGAN2-ADA 소스

- https://github.com/NVlabs/stylegan2-ada-pytorch
- 고정 commit: `d72cc7d041b42ec8e806021a205ed9349f87c6a4`
- 해당 commit: https://github.com/NVlabs/stylegan2-ada-pytorch/tree/d72cc7d041b42ec8e806021a205ed9349f87c6a4
- NVIDIA 소스의 저작권 표시와 `vendor/LICENSE.txt`를 보존했다. 사전학습 가중치는 사용하지 않았다.
- `vendor/training/networks.py`: 원본 블록, mapping, forward.
- `vendor/training/augment.py`: 원본 ADA bgc와 경계 처리. 사용하지 않는 filter-bank 초기화의 SciPy convolution만 동등한 NumPy row convolution으로 교체했다.
- `vendor/training/loss.py`: 원본 logistic, R1, PL, style mixing.
- `vendor/torch_utils`, `vendor/dnnlib`: 원본 연산 및 유틸리티.

## 논문

- StyleGAN2: Karras et al., Analyzing and Improving the Image Quality of StyleGAN. https://arxiv.org/abs/1912.04958
- ADA: Karras et al., Training Generative Adversarial Networks with Limited Data. https://arxiv.org/abs/2006.06676
- 조건부 붕괴 검토: Shahbazi et al., Collapse by Conditioning. https://arxiv.org/abs/2201.06578
- 후속 후보이며 현재 미사용: Rangwani et al., NoisyTwins. https://arxiv.org/abs/2304.05866

## 이 프로젝트에서 작성한 부분

- `models.py`: 151클래스/64x64, G/D 채널 표, one-NFE sampling adapter. 각 기본 블록과 forward는 NVIDIA 구현을 사용한다.
- `portable_ops.py`: CUDA extension 없는 reference filtering/activation, 고차 입력 미분을 지원하는 고정 좌표 bilinear sampler, batch별 대형 weight 복제를 피하는 동등한 modulated convolution.
- `recipe.py`: caller-driven bounded update API, gradient accumulation 정규화, lazy Adam 보정, EMA 및 ADA 제어. 자동 본학습·저장 기능은 없다.
- `verify.py`: 합성 입력을 이용한 국소 수치검증 및 선택적 단기 GPU 검증. 성능/FID 검증이 아니다.
- `data.py`: 원본 dataset.py와 같은 RGB/resize/normalize, TRAIN 내부의 중복 그룹 분리 및 클래스 균등 샘플링. validation 이미지에는 접근하지 않는다.
- `train.py`: 실제 TRAIN 학습 루프, 상태 전체의 원자적 저장/재개, 내부 holdout의 clean-fid 평가. 평가 수식과 Inception은 제공 evaluate.py와 같고 Windows worker 수는 0으로 둔다. Windows에서 png/PNG가 중복 열거되지 않도록 평가 중에만 확장자를 png 하나로 제한하고 복원한다. 내부 평가 장수와 label 분포는 공식 validation3020과 다르므로 점수를 직접 비교할 수 없다.
- 평가 전용 Inception은 clean-fid의 표준 평가 모델이다. Git에는 가중치를 포함하지 않으며 첫 평가에서 clean-fid가 캐시/다운로드를 관리한다. 모델 학습에는 사용하지 않는다.

공식 공개 코드가 허용된다는 규정에 따라 출처를 명시했다. 연산 호환화는 공식 CUDA 구현과 bitwise 동일함을 주장하지 않으며, 수치 비교와 역전파 검증으로 확인한다. 제공 의존성 밖의 패키지를 새로 설치하지 않았다.

## 공유 브랜치 정리

- 패키지 분리, 상대 import, 데이터 루트 CLI로 개인 PC 절대 경로 의존성을 제거했다. 학습 레시피와 가중치 키는 유지한다.
- 구조 참고: https://github.com/yunseopark-kaist/FastGen/tree/a80bbd0adbaff5ac81f7d9330955ac5e0a1b8421 — 기본 skeleton 보존, 네트워크/데이터/학습/검증 분리. 학습 코드는 이 저장소에서 복사하지 않았다.
- `benchmark.py`: 데이터나 체크포인트 없이 합성 입력으로 단계별 비용을 측정하며 가중치는 저장하지 않는다.
