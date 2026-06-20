from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RANDOM_STATE = 42
TEST_SIZE = 0.2

candidate_roots = [
    Path.cwd(),
    Path.cwd().parent,
]

PROJECT_ROOT = None

for root in candidate_roots:
    if (root / "data").exists():
        PROJECT_ROOT = root
        break

if PROJECT_ROOT is None:
    raise FileNotFoundError("data 폴더를 찾을 수 없습니다. 노트북 위치와 프로젝트 구조를 확인하십시오.")

DATA_DIR = PROJECT_ROOT / "data"
SRC_DIR = PROJECT_ROOT / "src"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = REPORTS_DIR / "tables"

SRC_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

print("PROJECT_ROOT:", PROJECT_ROOT)
print("DATA_DIR:", DATA_DIR)
print("SRC_DIR:", SRC_DIR)

# ## 32. 모델 학습용 라이브러리 불러오기 및 MLflow와 저장 폴더 설정하기

# 모델 학습, 실험 추적, 모델 저장에 필요한 라이브러리를 불러옵니다.
import json
import joblib

import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from IPython.display import display
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)



# MLflow 저장 폴더 설정하기
MODELS_DIR = PROJECT_ROOT / "models"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

MODELS_DIR.mkdir(exist_ok=True)
TABLES_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"file:{PROJECT_ROOT / 'mlruns'}")
mlflow.set_experiment("CardioCare_Heart_Disease_Prediction")

print("MLflow tracking URI:", mlflow.get_tracking_uri())
print("Models dir:", MODELS_DIR)

# ## 33. 학습 데이터 다시 준비하기
# 앞에서 만든 src/preprocessing.py를 사용해 데이터를 다시 불러오기

from preprocessing import (
    TARGET,
    load_heart_disease_processed_files,
    clean_dataframe,
    infer_feature_groups,
    split_features_target,
    build_preprocessor,
)

# 앞에서 만든 전처리 모듈을 사용해 학습용 데이터를 다시 준비합니다.
raw_df = load_heart_disease_processed_files(DATA_DIR)
clean_df = clean_dataframe(raw_df)

X, y = split_features_target(clean_df, drop_features=[])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

train_feature_df = X_train.copy()
train_feature_df[TARGET] = y_train.values

numeric_features, categorical_features, drop_features = infer_feature_groups(
    train_feature_df,
    missing_drop_threshold=0.5
)

X_train = X_train.drop(columns=drop_features, errors="ignore")
X_test = X_test.drop(columns=drop_features, errors="ignore")

print("사용 수치형 변수:", numeric_features)
print("사용 범주형 변수:", categorical_features)
print("제외 변수:", drop_features)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

# ## 34. 평가 함수 작성하기
# balacned accuracy, precision, recall, F1, confusion matrix 사용하여 평가지표 함수 작성하기

def evaluate_model(model, X_test, y_test):
    """PDF가 요구한 주요 평가 지표와 confusion matrix 값을 계산합니다."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, cm

# ## 35. Confusion Matrix 저장 함수 

def save_confusion_matrix(cm, model_name):
    """모델별 confusion matrix 그림을 보고서용 이미지로 저장합니다."""
    fig, ax = plt.subplots(figsize=(4, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["normal", "heart_disease"]
    )

    display.plot(ax=ax, values_format="d")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    return output_path

# ## 36. 선택된 feature name 추출 함수 

def get_selected_feature_names(fitted_pipeline):
    """SelectFromModel이 선택한 feature 이름을 추출합니다."""
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    selector = fitted_pipeline.named_steps["feature_selection"]

    feature_names = preprocessor.get_feature_names_out()
    selected_mask = selector.get_support()

    selected_features = feature_names[selected_mask].tolist()

    return selected_features

# ## 37. 모델 파이프라인 생성 함수 

def build_training_pipeline(model):
    """전처리, feature selection, 모델을 하나의 sklearn Pipeline으로 묶습니다."""
    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features
    )

    feature_selector = SelectFromModel(
        RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("feature_selection", feature_selector),
            ("model", model),
        ]
    )

    return pipeline

# ## 38. 비교할 모델 3개 정의 및 학습, 5-fold cv, MLflow 기록하기

# 최소 3개 모델 정의하기
candidate_models = {
    "logistic_regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "svc": SVC(
        probability=True,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}

candidate_models

# 모델 학습 및 5-fold cv, MLflow 기록
experiment_results = []

for model_name, base_model in candidate_models.items():
    print(f"\nTraining model: {model_name}")

    pipeline = build_training_pipeline(base_model)

    # 각 모델의 일반화 성능을 보기 위해 5-fold 교차 검증을 수행합니다.
    cv_scores = cross_val_score(
        pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="balanced_accuracy",
        n_jobs=-1
    )

    # 모델별 파라미터, 지표, 산출물을 MLflow run으로 기록합니다.
    with mlflow.start_run(run_name=model_name):
        pipeline.fit(X_train, y_train)

        metrics, cm = evaluate_model(pipeline, X_test, y_test)

        selected_features = get_selected_feature_names(pipeline)

        cm_path = save_confusion_matrix(cm, model_name)

        selected_features_path = TABLES_DIR / f"selected_features_{model_name}.json"
        with open(selected_features_path, "w", encoding="utf-8") as f:
            json.dump(selected_features, f, indent=2, ensure_ascii=False)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("numeric_features", numeric_features)
        mlflow.log_param("categorical_features", categorical_features)
        mlflow.log_param("drop_features", drop_features)
        mlflow.log_param("num_selected_features", len(selected_features))

        mlflow.log_metric("cv_balanced_accuracy_mean", cv_scores.mean())
        mlflow.log_metric("cv_balanced_accuracy_std", cv_scores.std())

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("task", "binary_classification")
        mlflow.set_tag("project", "CardioCare")

        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(selected_features_path))

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model"
        )

        result_row = {
            "model_name": model_name,
            "cv_balanced_accuracy_mean": cv_scores.mean(),
            "cv_balanced_accuracy_std": cv_scores.std(),
            **metrics,
            "num_selected_features": len(selected_features),
        }

        experiment_results.append(result_row)

results_df = pd.DataFrame(experiment_results)
display(results_df)

# ## 39. 모델 비교표 저장하기

# 임상 맥락에서 false negative를 줄이는 것이 중요하므로 recall을 우선 기준으로 정렬합니다.
results_df = results_df.sort_values(
    by=["recall", "balanced_accuracy", "f1"],
    ascending=False
)

model_comparison_path = TABLES_DIR / "model_comparison.csv"
results_df.to_csv(model_comparison_path, index=False)

display(results_df)

print("Saved:", model_comparison_path)

# ## 40. 현재 기준 최선 모델 확인하기

best_initial_model_name = results_df.iloc[0]["model_name"]

print("초기 기준 최선 모델:", best_initial_model_name)

display(
    results_df[
        [
            "model_name",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "false_negative",
            "cv_balanced_accuracy_mean"
        ]
    ]
)

# ## 41. Random Forest 하이퍼파라미터 튜닝하기

# 후보 모델 중 Random Forest에 대해 GridSearchCV로 하이퍼파라미터 탐색을 수행합니다.
rf_pipeline = build_training_pipeline(
    RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 5, 10],
    "model__min_samples_leaf": [1, 2],
}

grid_search = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=param_grid,
    scoring="balanced_accuracy",
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best CV balanced accuracy:", grid_search.best_score_)

# ## 42. 튜닝 모델 평가 및 MLflow 기록하기
# GridSearchCV를 통해 찾은 최종 후보 모델을 테스트셋에서 평가하고 MLflow에 기록하기

# 튜닝으로 선택된 최적 모델을 테스트셋에서 다시 평가합니다.
best_tuned_model = grid_search.best_estimator_

tuned_metrics, tuned_cm = evaluate_model(
    best_tuned_model,
    X_test,
    y_test
)

tuned_selected_features = get_selected_feature_names(best_tuned_model)

display(pd.DataFrame([tuned_metrics]))
print("Confusion matrix:")
print(tuned_cm)
print("선택 feature 수:", len(tuned_selected_features))

# 튜닝된 모델도 별도 MLflow run으로 기록합니다.
with mlflow.start_run(run_name="random_forest_tuned"):
    tuned_cm_path = save_confusion_matrix(
        tuned_cm,
        "random_forest_tuned"
    )

    tuned_selected_features_path = TABLES_DIR / "selected_features_random_forest_tuned.json"
    with open(tuned_selected_features_path, "w", encoding="utf-8") as f:
        json.dump(tuned_selected_features, f, indent=2, ensure_ascii=False)

    mlflow.log_param("model_name", "random_forest_tuned")
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("best_params", grid_search.best_params_)
    mlflow.log_param("numeric_features", numeric_features)
    mlflow.log_param("categorical_features", categorical_features)
    mlflow.log_param("drop_features", drop_features)
    mlflow.log_param("num_selected_features", len(tuned_selected_features))

    mlflow.log_metric("best_cv_balanced_accuracy", grid_search.best_score_)

    for metric_name, metric_value in tuned_metrics.items():
        mlflow.log_metric(metric_name, metric_value)

    mlflow.set_tag("model_family", "random_forest")
    mlflow.set_tag("model_stage", "tuned_candidate")
    mlflow.set_tag("task", "binary_classification")
    mlflow.set_tag("project", "CardioCare")

    mlflow.log_artifact(str(tuned_cm_path))
    mlflow.log_artifact(str(tuned_selected_features_path))

    mlflow.sklearn.log_model(
        sk_model=best_tuned_model,
        artifact_path="model"
    )

# ## 43. 최종 모델 비교표에 튜닝 모델 추가하기

# 기본 모델 비교표에 튜닝 모델 결과를 추가해 최종 비교표를 만듭니다.
tuned_result_row = {
    "model_name": "random_forest_tuned",
    "cv_balanced_accuracy_mean": grid_search.best_score_,
    "cv_balanced_accuracy_std": np.nan,
    **tuned_metrics,
    "num_selected_features": len(tuned_selected_features),
}

final_results_df = pd.concat(
    [
        results_df,
        pd.DataFrame([tuned_result_row])
    ],
    ignore_index=True
)

final_results_df = final_results_df.sort_values(
    by=["recall", "balanced_accuracy", "f1"],
    ascending=False
)

final_model_comparison_path = TABLES_DIR / "final_model_comparison.csv"
final_results_df.to_csv(final_model_comparison_path, index=False)

display(final_results_df)

print("Saved:", final_model_comparison_path)

# ## 44. 최종 모델 선택하기
# recall이 높고 false negative가 낮으며 balanced accuracy가 높고 f1-score가 안정적인 모델을 기준으로 최종 모델을 선택함.

# 정렬 기준상 가장 위에 있는 모델명을 최종 선택 후보로 확인합니다.
final_selected_model_name = final_results_df.iloc[0]["model_name"]

print("최종 선택 후보:", final_selected_model_name)

display(
    final_results_df[
        [
            "model_name",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "false_negative",
            "cv_balanced_accuracy_mean"
        ]
    ]
)

# ## 45. 최종 모델 객체 결정하기

# 현재 코드는 튜닝된 Random Forest를 최종 모델 파일로 저장합니다.
final_model = best_tuned_model
final_model_name = "random_forest_tuned"

final_model_path = MODELS_DIR / "final_model.joblib"

joblib.dump(final_model, final_model_path)

print("Final model saved:", final_model_path)

# ## 46. 테스트셋 저장하기

# 이후 테스트, 모니터링, 보고서 재현을 위해 train/test split을 CSV로 저장합니다.
test_set = X_test.copy()
test_set[TARGET] = y_test.values

train_set = X_train.copy()
train_set[TARGET] = y_train.values

train_set_path = DATA_DIR / "train_split.csv"
test_set_path = DATA_DIR / "test_split.csv"

train_set.to_csv(train_set_path, index=False)
test_set.to_csv(test_set_path, index=False)

print("Train split saved:", train_set_path)
print("Test split saved:", test_set_path)

# ## 47. 샘플 추론 입력 저장하기
# Docker와 inference.py에서 사용할 작은 입력 파일 만들기

# Docker와 inference.py 동작 확인에 사용할 작은 샘플 입력을 저장합니다.
sample_input = X_test.head(5).copy()

sample_input_path = DATA_DIR / "sample_input.csv"
sample_input.to_csv(sample_input_path, index=False)

display(sample_input)

print("Sample input saved:", sample_input_path)

# ## 48. 최종 모델 로드 테스트

# 저장된 모델 파일이 다시 로드되고 predict/predict_proba가 동작하는지 확인합니다.
loaded_model = joblib.load(final_model_path)

sample_pred = loaded_model.predict(sample_input)
sample_proba = loaded_model.predict_proba(sample_input)

print("Predictions:", sample_pred)
print("Probabilities:")
print(sample_proba)
