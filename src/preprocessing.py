
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RAW_FILES = {
    "cleveland": "processed.cleveland.data",
    "hungarian": "processed.hungarian.data",
    "switzerland": "processed.switzerland.data",
    "va": "processed.va.data",
}

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

NUMERIC_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

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

        temp_df[SOURCE] = source_name
        frames.append(temp_df)

    df = pd.concat(frames, ignore_index=True)

    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_dataframe(df):
    cleaned = df.copy()

    cleaned = cleaned.dropna(axis=1, how="all")
    cleaned = cleaned.drop_duplicates()

    cleaned[TARGET] = (cleaned[TARGET] > 0).astype(int)

    cleaned.loc[cleaned["age"] <= 0, "age"] = np.nan
    cleaned.loc[cleaned["trestbps"] <= 0, "trestbps"] = np.nan
    cleaned.loc[cleaned["chol"] <= 0, "chol"] = np.nan
    cleaned.loc[cleaned["thalach"] <= 0, "thalach"] = np.nan
    cleaned.loc[cleaned["oldpeak"] < 0, "oldpeak"] = np.nan

    return cleaned


def infer_feature_groups(df, missing_drop_threshold=0.5):
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    missing_ratio = df[feature_columns].isna().mean()

    high_missing_features = (
        missing_ratio[missing_ratio >= missing_drop_threshold]
        .index
        .tolist()
    )

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
    X = df.drop(columns=[TARGET], errors="ignore")
    X = X.drop(columns=drop_features, errors="ignore")
    y = df[TARGET].astype(int)

    return X, y


def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop"
    )

    return preprocessor
