# 여러 원본 심장질환 데이터를 읽고 전처리 파이프라인 구성을 돕는 모듈
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 데이터셋별 원본 파일 이름 목록
RAW_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

# 원본 데이터 파일을 읽을 때 사용할 공통 컬럼 이름
COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

# 기본적으로 연속형으로 다룰 변수 목록
NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

# 기본적으로 범주형으로 다룰 변수 목록
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]

TARGET = "target"
SOURCE = "source"


def load_heart_disease_processed_files(data_dir="data"):
    # 여러 병원 원본 파일을 읽어 하나의 데이터프레임으로 합침
    data_dir = Path(data_dir)
    frames = []

    for source_name, file_name in RAW_FILES.items():
        file_path = data_dir / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"Missing required data file: {file_path}")

        temp_df = pd.read_csv(
            file_path,
            header=None,
            names=COLUMNS,
            na_values="?"
        )

        # 어느 원본 파일에서 왔는지 추적할 수 있도록 source 컬럼을 추가함
        temp_df[SOURCE] = source_name
        frames.append(temp_df)

    df = pd.concat(frames, ignore_index=True)

    # 숫자로 해석 가능한 컬럼은 모두 수치형으로 변환함
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_dataframe(df):
    # 중복 제거, 타깃 이진화, 비정상 수치 정리를 수행함
    cleaned = df.copy()

    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.drop_duplicates()

    # 원본 타깃의 1 이상 값을 모두 심장질환 있음으로 묶어 이진 분류 형태로 변환함
    cleaned[TARGET] = (cleaned[TARGET] > 0).astype(int)

    # 임상적으로 말이 되지 않는 값은 결측치로 바꿔 이후 imputer가 처리하게 함
    cleaned.loc[cleaned["age"] <= 0, "age"] = np.nan
    cleaned.loc[cleaned["trestbps"] <= 0, "trestbps"] = np.nan
    cleaned.loc[cleaned["chol"] <= 0, "chol"] = np.nan
    cleaned.loc[cleaned["thalach"] <= 0, "thalach"] = np.nan
    cleaned.loc[cleaned["oldpeak"] < 0, "oldpeak"] = np.nan

    return cleaned


def infer_feature_groups(df, missing_drop_threshold=0.5):
    # 결측 비율을 기준으로 제외할 변수를 고르고 숫자형/범주형 목록을 다시 만듦
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_ratio = df[feature_columns].isna().mean()

    high_missing_features = (
        missing_ratio[missing_ratio >= missing_drop_threshold]
        .index
        .tolist()
    )

    # 학습 특징으로 직접 쓰지 않을 source 컬럼도 함께 제외 목록에 넣음
    drop_features = sorted(set(high_missing_features + [SOURCE]))

    selected_numeric_features = [
        col for col in NUMERIC_FEATURES
        if col not in drop_features
    ]

    selected_categorical_features = [
        col for col in CATEGORICAL_FEATURES
        if col not in drop_features
    ]

    return selected_numeric_features, selected_categorical_features, drop_features


def split_features_target(df, drop_features):
    # 타깃과 입력 특징을 분리하고 제외 대상 컬럼을 제거함
    X = df.drop(columns=[TARGET], errors="ignore")
    X = X.drop(columns=drop_features, errors="ignore")
    y = df[TARGET].astype(int)

    return X, y


def build_preprocessor(numeric_features, categorical_features):
    # 숫자형 변수는 중앙값 대치 후 표준화함
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    # 범주형 변수는 최빈값 대치 후 원-핫 인코딩함
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # 숫자형/범주형 전처리를 하나의 ColumnTransformer로 묶음
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor
