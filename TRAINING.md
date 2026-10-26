# SoFlow end-to-end

## 실행

TA 제공 환경을 활성화하고 데이터 이미지를 준비한 뒤 실행한다.

```bash
python pipeline.py check
python pipeline.py run --config configs/deadline.json \
  --data-root ./data/pokemon-generation-one-22k --run-root ./runs/soflow
```

증강 비교 → 4NFE teacher 학습 → pair 생성 → 1NFE 증류 → 체크포인트 출력 → 공식 평가 순서다.
중단 후에는 같은 코드·설정·명령으로 재개한다.

## 설정

전체 설정은 `configs/deadline.json`에 있다.

| 항목 | 값 |
|---|---|
| 모델 | SongUNet SoFlow, 95,524,611 parameters, 64×64, 151 classes |
| 학습 | FP32 가중치·Adam, BF16 forward, 배치16, seed42 |
| 증강 비교 | flip / flip+affine 각각3,000회, held-out TRAIN755 |
| affine | 회전±5°, 이동±3px, scale0.90–1.05, bilinear·흰 배경 |
| 4NFE teacher | 200,000회, AdamW lr1e-4, FM75%+TC25%, EMA |
| 추론 | EMA linear re-noising4, 추가 CFG forward 없음 |
| 1NFE 증류 | 자체 teacher EMA 초기화, endpoint MSE25,000회, lr1e-5 |
| pair | 클래스당512쌍, deterministic solution4, 생성FP32·저장FP16 |
| 학습 중 FID | TRAIN755, 생성 seed27182, batch8, FP32, Clean-FID clean |
| 최종 평가 | validation3,020장, 제공 evaluate.py, 두 모드 각각 평가 |

선택된 증강으로 train17,079장 전체를 학습한다. Validation은 학습·모델 선택에 쓰지 않는다.
CPU 배치 미리 준비는 `execution.soflow_prefetch_batches=2`로 설정한다.

## 출력

- `runs/soflow/teacher/`, `distill/`: current/best, loss, 샘플, FID, 메모리·환경 기록
- `runs/soflow/submission/`: 제출 소스와 `checkpoints/one_nfe.ckpt`, `few_nfe.ckpt`

학습된 체크포인트는 이 브랜치에 포함하지 않는다.

## 모니터·검사

```bash
python tools/status.py runs/soflow
python -m unittest discover -s tests -v
python submission_check.py runs/soflow/submission --device cuda
```

Tmux 실행 도구는 `tools/start_tmux.py`, 출처는 `CITATIONS.md`에 있다.
