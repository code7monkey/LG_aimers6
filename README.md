# Fold-Rank 프로젝트

이 저장소는 **IVF/DI 시술 데이터를 기반으로 임신 성공 여부를 예측**하기 위한 머신러닝 파이프라인을 제공합니다. 원본 노트북에서 구현한 데이터 전처리, 모델 학습, 교차 검증 및 가중 랭크 앙상블 로직을 깔끔한 Python 패키지 형태로 리팩터링했습니다.

## 개발 환경

- **Colab + GitHub**를 고려한 구조로, Colab에서 실험을 수행하고 모델을 학습한 후 GitHub 저장소에 코드를 버전 관리할 수 있습니다.
- 필요한 Python 라이브러리는 `requirements.txt`에 명시되어 있으며, `pip install -r requirements.txt` 명령으로 손쉽게 설치할 수 있습니다.

## 디렉터리 구조

```
├── src/                     # 실제 동작하는 핵심 코드 (패키지 형태)
│   ├── __init__.py          # src 패키지 모듈화
│   ├── dataset.py           # 데이터 로딩 및 전처리 모듈
│   ├── model.py             # 모델 생성 함수
│   ├── trainer.py           # 학습 루프 (K-Fold, 모델별 학습 등)
│   ├── losses.py            # 가중 랭크 앙상블 함수 등 공용 함수
│   └── utils.py             # 시드 고정 등 유틸리티 함수
├── train.py                 # 학습 실행 스크립트 (config 읽고 src 호출)
├── inference.py             # 추론 실행 스크립트 (모델 로드 후 submission 생성)
├── configs/                 # 코드 수정 없이 환경 변경 가능
│   ├── train.yaml           # 실험 및 학습 설정
│   └── submit.yaml          # 추론 및 제출 설정
├── data/                    # 입력 데이터 (예시 데이터 포함; 실제 대회 데이터로 교체 필요)
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── assets/                  # 모델, 토크나이저 파일 저장용 디렉터리 (Git LFS 사용 권장)
│   └── README.md            # 안내 문서
├── outputs/                 # submission 등 출력 파일 저장 경로
├── requirements.txt         # 실행 시 설치가 필요한 패키지 목록
├── .gitignore               # Git이 무시해야 할 파일/폴더 목록
└── .gitattributes           # Git 속성 및 LFS 설정
```

## 빠른 시작

1. **의존성 설치**

   ```bash
   pip install -r requirements.txt
   ```

2. **학습 데이터 준비**

   - `data/train.csv`, `data/test.csv`, `data/sample_submission.csv` 파일을 실제 데이터로 교체합니다. 예시 데이터는 구조를 이해하기 위한 용도로 제공됩니다.
   - 학습 데이터는 `임신 성공 여부` 컬럼을 포함해야 합니다.

3. **모델 학습**

   ```bash
   python train.py --config configs/train.yaml
   ```

   - `configs/train.yaml`에서 모델 하이퍼파라미터, K-Fold 설정, 앙상블 가중치 등을 변경할 수 있습니다.
   - 학습이 완료되면 AUC 점수가 출력되고, 학습된 모델은 `assets/` 디렉터리에 저장됩니다.
   - 최종 예측 결과는 `outputs/` 디렉터리의 `submission.csv`에 저장됩니다.

4. **추론 및 제출 파일 생성**

   ```bash
   python inference.py --config configs/submit.yaml
   ```

   - `configs/submit.yaml`에서 사용할 모델 목록, 가중치 등을 지정할 수 있습니다.
   - 저장된 모델들을 로드하여 테스트 데이터에 대한 예측을 수행하고, 앙상블 후 submission 파일을 만듭니다.

## 커스텀 설정

- **하이퍼파라미터 튜닝**: `configs/train.yaml`의 `models` 항목에서 각 모델별 하이퍼파라미터를 수정하여 성능을 개선할 수 있습니다.
- **앙상블 전략 변경**: `src/losses.py`의 `weighted_rank_ensemble` 함수를 수정하거나 다른 앙상블 방식을 구현할 수 있습니다.
- **데이터 전처리 수정**: `src/dataset.py`는 노트북에서 수행한 전처리 로직을 함수화한 것입니다. 데이터의 특성에 맞게 컬럼 처리 방식이나 파생변수 생성을 변경할 수 있습니다.

## 주의 사항

- `assets/` 디렉터리에는 학습된 모델 파일과 같은 대용량 바이너리가 저장됩니다. GitHub에 업로드할 때는 Git LFS를 사용하거나 `.gitignore`에 추가하여 커밋되지 않도록 주의하세요.
- 예시 데이터는 실제 문제 풀이에 적합하지 않습니다. 반드시 실제 대회 데이터로 교체한 뒤 학습/추론을 진행하세요.

## 라이선스

본 프로젝트는 교육 및 연구 목적으로만 사용 가능하며, 다른 용도로 사용하고자 할 경우 원저자의 허가를 받아야 합니다.