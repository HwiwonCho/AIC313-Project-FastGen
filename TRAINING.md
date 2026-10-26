# Server SoFlow 4NFE

| 설정 | 값 |
|---|---|
| 모델 | SongUNet SoFlow, 95,524,611 parameters |
| 데이터 | 제공 train의 fit13,667 / holdout3,412 (`configs/internal_split.json`) |
| 증강 | 좌우반전0.5 |
| 학습 | FM75% + TC25%, AdamW1e-4, betas(.9,.99), clip10, EMA.9999 |
| 스케줄 | warmup4,270 / cosine horizon170,800 / 최소 LR1e-5 |
| 배치 | 최초6,000회 micro4×누적4, 이후 grouped16×1 |
| 정밀도 | FP32 가중치·Adam, CUDA BF16 forward, channels-last |
| 4NFE | t=[1,.75,.5,.25,0], 각 t→0 예측 후 독립 noise로 재혼합, 마지막만 clamp |

새 학습·내보내기·공식 평가:
```bash
bash run.sh /path/to/data ./runs/best4
```

기존 서버 최고점 EMA를 그대로 내보내기:
```bash
python export.py --source /path/to/original/best_nfe4.pt
python check_few.py --device cpu
python evaluate_few.py --data-root /path/to/data
```

제출 폴더 생성:
```bash
python package.py --source /path/to/original/best_nfe4.pt --output ./submission
```

원본은188,000회 EMA + linear re-noising4, validation3020 자체 FID37.205018이다.
당시 validation으로 체크포인트·샘플러를 선택했다. 새 학습 코드는 train holdout만 선택에 쓴다.
새 학습이 원본 FID나 가중치를 동일하게 만든다는 보장은 없다.

출력은 `checkpoints/few_nfe.ckpt`다. 이 브랜치는4NFE 전용이며, 중간 제출 합본에는1NFE를 추가해야 한다.
학습 체크포인트·데이터 이미지는 Git에 포함하지 않는다. 출처는 `CITATIONS.md`에 있다.
