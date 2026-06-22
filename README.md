# CardioCare

CardioCare는 UCI Heart Disease 계열 데이터를 사용하여 심장질환 가능성을 예측하는 end-to-end 머신러닝 프로젝트입니다. 목표는 단순히 분류 모델을 학습하는 것이 아니라, EDA, 전처리, 모델 학습, MLflow 실험 관리, 단위 테스트, Docker 패키징, CI, 추론, 모니터링과 데이터 드리프트 탐지까지 하나의 재현 가능한 파이프라인으로 구성하는 것입니다.

이 시스템은 `inform, not decide` 원칙을 따릅니다. 모델은 환자의 진단을 단독으로 결정하지 않으며, 의료진이 추가 검토할 수 있도록 위험 가능성을 알려주는 보조 도구로 사용되어야 합니다.

## 빠른 재현 방법

아래 명령은 Windows PowerShell 기준입니다. 저장소 루트에서 실행합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

전체 학습 파이프라인을 다시 실행합니다.

```powershell
python src/train.py
```

학습 후 주요 산출물은 다음 위치에 생성되거나 갱신됩니다.

- `models/final_model.joblib`
- `data/train_split.csv`
- `data/test_split.csv`
- `data/sample_input.csv`
- `reports/tables/final_model_comparison.csv`
- `mlruns/`

테스트를 실행합니다.

```powershell
python -m unittest discover -s tests -v
```

저장된 모델로 샘플 추론을 실행합니다.

```powershell
python src/inference.py
```

추론 결과는 `reports/inference_predictions.csv`에 저장되고, 추론 로그는 `reports/inference.log`에 저장됩니다.

Docker 이미지 빌드와 컨테이너 추론을 확인합니다.

```powershell
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

데이터 드리프트 모니터링을 실행합니다.

```powershell
python src/monitor.py
```

모니터링 결과는 `reports/tables/`, `reports/figures/`, `reports/monitor.log`에 저장됩니다.

## 프로젝트 구조

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
├── mlruns/
├── reports/
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
├── report.pdf
└── README.md
```

## 파일 역할

| 파일 | 역할 |
|---|---|
| `notebooks/01_eda_preprocessing.ipynb` | 데이터 로딩, `head()`, `info()`, `describe()`, 타깃 분포, 결측치, 이상치 확인 및 전처리 전략 정리 |
| `src/preprocessing.py` | 원본 데이터 로딩, 중복/빈 컬럼/임상적으로 불가능한 값 처리, 타깃 이진화, 재사용 가능한 전처리 파이프라인 구성 |
| `src/train.py` | train/test split, 스케일링, 특성 선택, 3개 이상 모델 비교, 5-fold CV, GridSearchCV, MLflow 기록, 최종 모델 저장 |
| `src/inference.py` | 저장된 모델과 CSV 입력을 사용한 추론, 입력값 범위 검증, 예측 결과 CSV 저장, 추론 로그 기록 |
| `src/monitor.py` | synthetic drift 생성, KS 검정, 원본/드리프트 테스트셋 성능 비교, 시계열 그래프와 로그 저장 |
| `tests/test_pipeline.py` | 예측 shape, 확률 범위, 임상 입력 범위, 결정론적 예측을 검증하는 unittest |
| `Dockerfile` | `python:3.10-slim` 기반 이미지 생성, 의존성 설치, 코드/데이터/모델 복사, 샘플 추론 실행 |
| `.github/workflows/ci.yml` | push 또는 pull request 발생 시 의존성 설치, 필수 모델/입력 파일 확인, unittest 실행 |
| `report.pdf` | 최종 보고서 |

## 데이터와 문제 정의

사용 데이터는 UCI Heart Disease 계열 데이터입니다. 여러 원본 파일을 통합하여 사용하며, 원본 target이 다중 클래스일 수 있으므로 다음과 같이 이진화합니다.

| 원본 target | 변환 후 의미 |
|---:|---|
| `0` | 정상 |
| `1` 이상 | 심장질환 가능성 있음 |

의료 보조 시스템에서는 실제 심장질환 환자를 정상으로 예측하는 False Negative가 특히 위험합니다. 따라서 모델 평가는 accuracy만 사용하지 않고 `balanced accuracy`, `precision`, `recall`, `F1`, `confusion matrix`를 함께 사용합니다.

## EDA와 전처리 요약

EDA에서는 타깃 클래스 분포, 컬럼별 결측치, 연속형 변수 분포와 이상치를 확인했습니다. 주요 결측치 비율은 `ca` 약 66.41%, `thal` 약 52.83%, `slope` 약 33.59%로 나타났습니다. 결측치가 매우 높은 컬럼은 학습 feature에서 제외하고, 상대적으로 결측치가 적은 수치형 변수는 중앙값으로 대치했습니다.

전처리에서는 중복 행 제거, 완전히 비어 있는 컬럼 제거, target 이진화, 임상적으로 불가능한 값의 `NaN` 변환, `source` 컬럼 제거를 수행했습니다. 수치형 변수는 `SimpleImputer(strategy="median")`와 `StandardScaler()`로 처리하고, 범주형 변수는 `SimpleImputer(strategy="most_frequent")`와 `OneHotEncoder(handle_unknown="ignore")`로 처리합니다. 모든 전처리는 `ColumnTransformer`와 `sklearn.pipeline.Pipeline` 안에서 수행되어 train/test split 이후 학습 데이터에 대해서만 fit되도록 구성했습니다.

## 모델 학습과 최종 모델

`src/train.py`에서는 Logistic Regression, SVC, Random Forest, Tuned Random Forest를 비교했습니다. 실험은 MLflow에 파라미터, 평가지표, confusion matrix, 선택 feature, 모델 artifact를 기록합니다.

| 모델 | CV balanced accuracy | Test balanced accuracy | Precision | Recall | F1 | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SVC | 0.7858 | 0.7727 | 0.7905 | 0.8137 | 0.8019 | 60 | 22 | 19 | 83 |
| Logistic Regression | 0.7801 | 0.7861 | 0.8119 | 0.8039 | 0.8079 | 63 | 19 | 20 | 82 |
| Random Forest | 0.7635 | 0.7556 | 0.7736 | 0.8039 | 0.7885 | 58 | 24 | 20 | 82 |
| Tuned Random Forest | 0.7825 | 0.7873 | 0.8182 | 0.7941 | 0.8060 | 64 | 18 | 21 | 81 |

최종 저장 모델은 `models/final_model.joblib`이며, Tuned Random Forest를 사용합니다. 이 모델은 test balanced accuracy가 약 0.7873으로 가장 높고 precision도 약 0.8182로 가장 높아 전체적인 성능 균형이 좋습니다. 다만 SVC는 recall이 더 높고 FN이 19건으로 가장 적기 때문에, 실제 운영 전에는 threshold 조정과 calibration을 통해 FN을 줄이는 추가 검토가 필요합니다.

MLflow UI는 다음 명령으로 확인할 수 있습니다.

```powershell
mlflow ui --backend-store-uri mlruns
```

## 테스트

`tests/test_pipeline.py`는 다음 4가지를 검증합니다.

1. 예측 결과의 행 수가 입력 행 수와 일치하는지 확인합니다.
2. `predict_proba` 확률이 `[0, 1]` 범위에 있고 각 행의 확률 합이 1에 가까운지 확인합니다.
3. `chol=999`처럼 임상적으로 허용 범위를 벗어난 입력이 예외를 발생시키는지 확인합니다.
4. 동일 모델과 동일 입력에서 예측 결과가 반복 실행 시 동일한지 확인합니다.

실행 명령은 다음과 같습니다.

```powershell
python -m unittest discover -s tests -v
```

## Docker

Dockerfile은 `python:3.10-slim` 이미지를 기반으로 합니다. `requirements.txt`를 설치한 뒤 `src/`, `tests/`, `data/`, `models/`를 컨테이너 내부로 복사하고, 기본 명령으로 `src/inference.py`를 실행합니다.

```powershell
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

컨테이너 실행 시 `data/sample_input.csv`와 `models/final_model.joblib`을 사용하여 샘플 입력 5건의 예측 결과와 심장질환 확률을 출력합니다.

## CI

`.github/workflows/ci.yml`은 push 또는 pull request 발생 시 자동으로 실행됩니다. CI는 Python 3.10 환경을 설정하고, `requirements.txt`를 설치한 뒤, `models/final_model.joblib`과 `data/sample_input.csv` 존재 여부를 확인하고 unittest를 실행합니다.

GitHub Actions에서 main 브랜치의 CI 워크플로우가 green 상태로 통과한 것을 보고서에 기록했습니다.

## 모니터링과 드리프트

`src/monitor.py`는 저장된 모델과 `data/train_split.csv`, `data/test_split.csv`를 사용합니다. 테스트셋 복사본에 synthetic drift를 적용한 뒤, 학습 분포와 drifted test 분포를 KS 검정으로 비교합니다.

적용한 synthetic drift는 다음과 같습니다.

| 변수 | 적용한 변화 |
|---|---:|
| `chol` | +120 |
| `trestbps` | +40 |
| `thalach` | -40 |
| `oldpeak` | +2 |

KS 검정 결과는 다음과 같습니다.

| Feature | p-value | 판정 |
|---|---:|---|
| `age` | 약 0.8793 | drift 아님 |
| `trestbps` | 약 6.06e-77 | drift 탐지 |
| `chol` | 약 5.19e-76 | drift 탐지 |
| `thalach` | 약 4.35e-38 | drift 탐지 |
| `oldpeak` | 약 1.70e-87 | drift 탐지 |

성능 비교 결과는 다음과 같습니다.

| Dataset | Samples | Balanced Accuracy | 원본 대비 변화량 |
|---|---:|---:|---:|
| `original_test` | 184 | 약 0.7873 | 0.0000 |
| `drifted_test` | 184 | 약 0.7048 | 약 -0.0825 |

이는 입력 분포 변화가 통계적으로 탐지될 뿐 아니라 실제 모델 성능 저하와도 연결될 수 있음을 보여줍니다.

## 재학습과 피드백 루프 전략

재학습은 자동으로 수행하지 않습니다. 의료 도메인에서는 잘못된 라벨이나 편향된 데이터를 자동으로 다시 학습하면 기존 편향이 강화될 수 있기 때문입니다. 핵심 변수의 drift가 지속적으로 감지되고, balanced accuracy 또는 recall 하락이 관찰되며, 의료진이 검증한 정답 라벨이 충분히 수집되었을 때 재학습 후보로 올립니다.

재학습 승인에는 human-in-the-loop가 필요합니다. 의사 또는 데이터 담당자가 라벨 품질, 환자군 변화, 데이터 수집 방식의 변화 여부를 검토한 뒤 재학습 여부를 결정해야 합니다. 모델 예측값을 정답처럼 다시 학습 데이터에 넣는 방식은 폭주하는 피드백 루프를 만들 수 있으므로 사용하지 않습니다.

## 서빙 선택

본 프로젝트에서는 Model-as-a-Service 방식을 선택합니다. 심장질환 위험 예측은 밀리초 단위 응답이 필요한 응급 처치 시스템이라기보다 진료 보조와 추가 검사 판단을 위한 위험도 산출에 가깝기 때문에, 병원 내부망 또는 통제된 서버에서 API 형태로 운영하는 방식이 적절합니다.

의료 데이터에는 PHI와 같은 민감 정보가 포함될 수 있으므로, 외부 공개 서버가 아니라 병원 내부망 또는 보안 통제 환경에서 운영해야 합니다. 또한 전송 중 암호화, 접근 제어, 인증, 로그 마스킹, 최소 정보 수집 원칙이 필요합니다. 중앙 서버 방식은 모델 버전 관리, 업데이트, 롤백이 쉽다는 장점도 있습니다.

## 피처 스토어와 모델 레지스트리 예시

피처 스토어에 관리할 만한 feature 예시는 `thalach`입니다. 최대 심박수는 추론 시점에도 재사용 가치가 큰 임상 변수이며, 드리프트 감지와 모델 입력 품질 관리에도 중요합니다.

모델 레지스트리에 기록해야 할 메타데이터 예시는 `model_version`, `balanced_accuracy`, `recall`, `false_negative`, `training_data_version`입니다. 특히 의료 보조 시스템에서는 전체 정확도뿐 아니라 실제 환자를 놓치는 FN과 recall을 함께 추적해야 합니다.

## 최종 보고서

최종 보고서는 `report.pdf`입니다. 보고서에는 문제 정의, EDA 결과, 전처리 결정, MLflow 기반 모델 비교, 테스트와 Docker 패키징, CI, 드리프트 결과, 재학습/피드백 루프 계획, 서빙 선택, 한계와 윤리적 고려 사항, AI 도구 사용 공개가 포함되어 있습니다.

## 주요 명령 요약

```powershell
pip install -r requirements.txt
python src/train.py
python -m unittest discover -s tests -v
python src/inference.py
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
python src/monitor.py
mlflow ui --backend-store-uri mlruns
```
