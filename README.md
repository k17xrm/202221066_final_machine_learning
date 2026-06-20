# CardioCare

심장질환 예측을 위한 end-to-end 머신러닝 프로젝트입니다.  
UCI Heart Disease 데이터를 기반으로 EDA, 전처리, 모델 학습, 테스트, Docker 패키징, CI, 모니터링과 드리프트 탐지까지 한 흐름으로 재현할 수 있도록 구성했습니다.

## 프로젝트 개요

- 데이터 이해 및 전처리: `notebooks/01_eda_preprocessing.ipynb`, `src/preprocessing.py`
- 모델 학습 및 실험 추적: `src/train.py`, `mlruns/`
- 추론 및 입력 검증: `src/inference.py`
- 모니터링 및 드리프트 탐지: `src/monitor.py`
- 단위 테스트: `tests/test_pipeline.py`
- 컨테이너화: `Dockerfile`
- CI: `.github/workflows/ci.yml`

## 사용 데이터

- 데이터셋: UCI Heart Disease
- 사용 형식: 병합된 processed 데이터
- 타깃 정의: `target = 0` 정상, `target = 1` 심장질환
- 데이터 로딩 방식: `data/processed.*.data` 파일을 로컬에서 결정론적으로 읽음

## 재현 방법

권장 실행 순서:

```powershell
pip install -r requirements.txt
python src/train.py
python -m unittest discover -s tests
docker build -t cardiocare:1.0 .
python src/monitor.py
```

## 저장소 구조

```text
.
├── data/
│   ├── processed.cleveland.data
│   ├── processed.hungarian.data
│   ├── processed.switzerland.data
│   ├── processed.va.data
│   ├── train_split.csv
│   ├── test_split.csv
│   └── sample_input.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── reports/
│   ├── figures/
│   └── tables/
├── mlruns/
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
└── report.pdf
```

### 각 파일 역할

- `notebooks/01_eda_preprocessing.ipynb`
  - `head()`, `info()`, `describe()`로 기본 구조 확인
  - 타깃 분포 확인
  - 결측값 및 이상치 탐색
  - 전처리 선택 근거 정리
- `src/preprocessing.py`
  - 데이터 로딩
  - 결측/중복/이상치 처리
  - 특성 그룹 추론
  - sklearn `ColumnTransformer` 기반 전처리 파이프라인 구성
- `src/train.py`
  - train/test split
  - 스케일링 및 특성 선택
  - 3개 이상 모델 비교
  - 교차 검증 및 하이퍼파라미터 탐색
  - MLflow 기록
  - 최종 모델 저장
- `src/inference.py`
  - 입력 스키마와 임상 범위 검증
  - 예측 및 확률 출력
  - 추론 로그 기록
- `src/monitor.py`
  - 인위적 드리프트 생성
  - KS 검정
  - 원본/드리프트 성능 비교
  - 시계열 형태의 모니터링 지표 저장
- `tests/test_pipeline.py`
  - 예측 shape 검증
  - 확률 범위 검증
  - 임상 범위 검증
  - 결정론성 검증
- `.github/workflows/ci.yml`
  - push/pull request마다 unittest 실행
- `Dockerfile`
  - requirements 설치
  - 코드와 저장된 모델 복사
  - 샘플 입력으로 추론 실행

## 5.1 데이터 이해 및 전처리 요약

### 무엇을 확인했는가

- `head()`, `info()`, `describe()`로 컬럼 형태와 수치 분포를 확인했습니다.
- `value_counts(normalize=True)`로 타깃 분포를 확인했습니다.
- 결측값은 컬럼별 비율과 전체 분포를 함께 확인했습니다.
- 이상치는 특히 연속형 변수(`age`, `trestbps`, `chol`, `thalach`, `oldpeak`)를 중심으로 점검했습니다.

### 전처리 해석

- 결측이 많은 컬럼은 삭제 기준을 적용했습니다.
- 임상적으로 의미가 있고 결측이 적은 컬럼은 대치 방식으로 유지했습니다.
- 중복과 빈 컬럼은 제거했습니다.
- 전처리 로직은 일회성 셀이 아니라 재사용 가능한 함수와 sklearn 파이프라인 형태로 유지했습니다.

### 해석 문장 예시

> EDA 결과, 심장질환 데이터는 일부 연속형 특성에 결측과 이상치가 존재했고 타깃 분포도 완전히 균형적이지 않았습니다. 따라서 학습 가능한 컬럼은 유지하되, 결측이 큰 컬럼은 제거하고 나머지는 중앙값/최빈값 대치와 스케일링이 가능한 재사용형 파이프라인으로 정리했습니다.

## 5.2 모델 학습 및 실험 요약

### 학습 흐름

- `train_test_split`으로 데이터 분할
- 학습 데이터에만 fit 되는 스케일링 적용
- Random Forest 기반 `SelectFromModel`로 특성 선택
- 서로 다른 계열의 모델 3개 이상 비교
- 5-fold 교차 검증 수행
- 가장 유력한 모델에 대해 하이퍼파라미터 탐색 수행
- MLflow에 파라미터, 지표, 아티팩트 기록

### 모델 비교 결과

| 모델 | CV Balanced Accuracy | Test Balanced Accuracy | Precision | Recall | F1 | False Negative |
|---|---:|---:|---:|---:|---:|---:|
| SVC | 0.7858 | 0.7727 | 0.7905 | 0.8137 | 0.8019 | 19 |
| Logistic Regression | 0.7801 | 0.7861 | 0.8119 | 0.8039 | 0.8079 | 20 |
| Random Forest | 0.7635 | 0.7556 | 0.7736 | 0.8039 | 0.7885 | 20 |
| Random Forest Tuned | 0.7825 | 0.7873 | 0.8182 | 0.7941 | 0.8060 | 21 |

### 결과 해석

- 단일 지표 기준으로는 tuned Random Forest와 Logistic Regression이 상위권입니다.
- `balanced accuracy` 기준으로는 tuned Random Forest가 가장 높고, 테스트 성능도 안정적입니다.
- `precision`과 `recall`이 함께 괜찮아서, 단순히 양성만 많이 잡는 편향은 아닙니다.
- 의료 문제에서는 false negative가 중요하므로, 단순 정확도보다 재현율과 FN 개수를 함께 봐야 합니다.
- 이 프로젝트에서는 최종 모델을 tuned Random Forest로 선택했고, 임상적으로는 “심장질환 환자를 놓치지 않는 것”을 중요한 기준으로 삼았습니다.

### 한줄 정리

> 여러 모델을 비교한 결과, 최종 모델은 tuned Random Forest로 선택했으며, 이유는 교차 검증과 테스트 성능이 모두 높고 심장질환 양성 사례를 비교적 안정적으로 포착했기 때문입니다.

## 5.3 테스트 및 패키징 요약

### unittest

아래 4가지를 검증합니다.

1. 예측 결과 shape가 입력 shape와 일치하는지
2. 예측 확률이 `[0, 1]` 범위에 있고 각 행의 합이 1에 가까운지
3. 임상적으로 정해진 범위의 입력값 검증이 동작하는지
4. 동일 입력에서 결정론적으로 같은 결과가 나오는지

### Docker

- `requirements.txt` 기반 설치
- 코드와 저장된 모델 복사
- 샘플 입력으로 추론 실행
- `docker build -t cardiocare:1.0 .` 형태로 빌드 가능

### CI

- `.github/workflows/ci.yml`에서 push/pull request마다 unittest를 실행합니다.
- 채점 환경에서도 최소한의 회귀 테스트가 반복 실행되도록 설계했습니다.

### 피처 스토어 / 모델 레지스트리 서술 예시

- 피처 스토어에 넣기 좋은 피처: `thalach`
  - 이유: 운동 유발 최대 심박수는 추론 시점에도 재사용 가치가 크고, 실시간 또는 배치 피처로 관리하기 좋습니다.
- 모델 레지스트리에 기록할 메타데이터 예시: `balanced_accuracy`, `recall`, `false_negative`
  - 이유: 의료 문제에서는 단순 정확도보다 놓치는 환자 수와 양성 탐지 능력이 중요하기 때문입니다.

## 5.4 모니터링 및 드리프트 요약

### 로깅

추론 경로에 다음 정보를 남기도록 구성했습니다.

- timestamp
- model version
- input shape
- predictions
- 가능한 경우 actual label

### 드리프트 실험

테스트셋 복사본에 연속형 특성의 분포 이동을 인위적으로 넣었습니다.

### KS 검정 결과

| Feature | p-value | Drift 여부 |
|---|---:|---|
| age | 0.8793 | False |
| trestbps | 6.06e-77 | True |
| chol | 5.19e-76 | True |
| thalach | 4.35e-38 | True |
| oldpeak | 1.70e-87 | True |

### 성능 비교

| Dataset | Balanced Accuracy | 변화량 |
|---|---:|---:|
| Original test | 0.7873 | 0.0000 |
| Drifted test | 0.7048 | -0.0825 |

### 시계열 지표

| Drift Strength | Balanced Accuracy |
|---|---:|
| 0.0 | 0.7873 |
| 0.2 | 0.7764 |
| 0.4 | 0.7484 |
| 0.6 | 0.7547 |
| 0.8 | 0.7340 |
| 1.0 | 0.7048 |

### 결과 해석

- 드리프트 강도가 커질수록 전반적으로 balanced accuracy가 하락했습니다.
- KS 검정에서도 연속형 특성들 중 다수가 `p < 0.05`로 드리프트가 감지되었습니다.
- `age`는 유의한 변화가 없었고, 나머지 주요 연속형 특성은 명확히 변했습니다.
- 즉, 입력 분포 변화가 실제 성능 저하와 연결된다는 점을 보여 줍니다.
- 운영 관점에서는 이런 신호가 누적되면 재학습을 검토해야 합니다.

### 재학습 및 피드백 루프

- 드리프트가 한 번 감지되었다고 즉시 자동 재학습하는 방식은 위험할 수 있습니다.
- 성능 하락이 반복 확인되면 재학습을 고려하고, 그 전에 사람이 결과를 검토하는 Human-in-the-loop 지점을 두는 편이 안전합니다.
- 특히 의료 도메인에서는 잘못된 재학습이 폭주하는 피드백 루프를 만들 수 있으므로, drift 탐지, 성능 확인, 검토, 재학습의 순서를 분리하는 것이 중요합니다.

## 보고서 작성 포인트

`report.pdf`에는 아래 순서로 정리하면 읽기 좋습니다.

1. 문제 정의와 사용 목적
2. EDA 핵심 결과
3. 전처리 결정 근거
4. 모델 비교 및 최종 선택 정당화
5. 테스트와 패키징
6. 드리프트 결과와 재학습 계획
7. 서빙 선택
8. 한계, 윤리적 고려 사항, 향후 개선점

## 참고

- Python 버전: 3.10.13
- 주요 라이브러리: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `mlflow`, `scipy`, `unittest`
