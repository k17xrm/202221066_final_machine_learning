from pathlib import Path
import logging
import argparse
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DEFAULT_TRAIN_PATH = PROJECT_ROOT / "data" / "train_split.csv"
DEFAULT_TEST_PATH = PROJECT_ROOT / "data" / "test_split.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"

DEFAULT_LOG_PATH = REPORTS_DIR / "monitor.log"

TARGET = "target"

CONTINUOUS_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
]

# Synthetic drift scenario.
# These shifts are intentionally stronger than a tiny perturbation so that
# the effect can be observed in KS statistics and model performance.
SHIFT_CONFIG = {
    "chol": 80.0,
    "trestbps": 20.0,
    "thalach": -25.0,
    "oldpeak": 1.0,
}

CLINICAL_LIMITS = {
    "age": (0, 120),
    "trestbps": (0, 250),
    "chol": (0, 600),
    "thalach": (0, 250),
    "oldpeak": (0, 10),
}


def setup_directories() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path = DEFAULT_LOG_PATH) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def get_model_version(model_path: Path) -> str:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    modified_time = datetime.fromtimestamp(model_path.stat().st_mtime)
    version = f"{model_path.stem}_{modified_time.strftime('%Y%m%d_%H%M%S')}"

    return version


def load_model(model_path: Path = DEFAULT_MODEL_PATH):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def load_train_test_data(
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train split file not found: {train_path}. "
            "Run your training notebook or src/train.py first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test split file not found: {test_path}. "
            "Run your training notebook or src/train.py first."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if TARGET not in train_df.columns:
        raise ValueError(f"Missing target column in train split: {TARGET}")

    if TARGET not in test_df.columns:
        raise ValueError(f"Missing target column in test split: {TARGET}")

    X_train = train_df.drop(columns=[TARGET])
    y_train = train_df[TARGET].astype(int)

    X_test = test_df.drop(columns=[TARGET])
    y_test = test_df[TARGET].astype(int)

    return X_train, X_test, y_train, y_test


def clip_to_clinical_limits(df: pd.DataFrame) -> pd.DataFrame:
    clipped = df.copy()

    for col, (lower, upper) in CLINICAL_LIMITS.items():
        if col in clipped.columns:
            clipped[col] = clipped[col].clip(lower=lower, upper=upper)

    return clipped


def create_drifted_dataset(
    X: pd.DataFrame,
    strength: float = 1.0,
) -> pd.DataFrame:
    drifted = X.copy()

    for col, shift_value in SHIFT_CONFIG.items():
        if col in drifted.columns:
            drifted[col] = drifted[col] + shift_value * strength

    drifted = clip_to_clinical_limits(drifted)

    return drifted


def run_prediction_logging(
    model,
    X: pd.DataFrame,
    y_true: pd.Series | None,
    model_version: str,
    dataset_name: str,
) -> np.ndarray:
    predictions = model.predict(X)

    log_payload = {
        "model_version": model_version,
        "dataset_name": dataset_name,
        "input_shape": X.shape,
        "predictions": predictions.tolist(),
    }

    if y_true is not None:
        log_payload["actual"] = y_true.tolist()

    logging.info("%s", log_payload)

    return predictions


def compute_performance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_version: str,
    dataset_name: str,
) -> dict:
    predictions = run_prediction_logging(
        model=model,
        X=X,
        y_true=y,
        model_version=model_version,
        dataset_name=dataset_name,
    )

    balanced_accuracy = balanced_accuracy_score(y, predictions)

    result = {
        "dataset": dataset_name,
        "model_version": model_version,
        "n_samples": len(y),
        "balanced_accuracy": balanced_accuracy,
    }

    return result


def run_ks_drift_test(
    X_train: pd.DataFrame,
    X_drifted: pd.DataFrame,
    continuous_features: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []

    for col in continuous_features:
        if col not in X_train.columns:
            continue

        if col not in X_drifted.columns:
            continue

        train_values = X_train[col].dropna()
        drifted_values = X_drifted[col].dropna()

        if len(train_values) == 0 or len(drifted_values) == 0:
            continue

        statistic, p_value = ks_2samp(train_values, drifted_values)

        rows.append(
            {
                "feature": col,
                "ks_statistic": statistic,
                "p_value": p_value,
                "alpha": alpha,
                "is_drifted": bool(p_value < alpha),
                "train_mean": train_values.mean(),
                "drifted_mean": drifted_values.mean(),
                "mean_difference": drifted_values.mean() - train_values.mean(),
            }
        )

    return pd.DataFrame(rows)


def create_metric_timeseries(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_version: str,
) -> pd.DataFrame:
    rows = []

    base_time = datetime.now().replace(microsecond=0)
    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for idx, strength in enumerate(strengths):
        batch_time = base_time + timedelta(hours=idx)
        batch_X = create_drifted_dataset(X_test, strength=strength)

        predictions = run_prediction_logging(
            model=model,
            X=batch_X,
            y_true=y_test,
            model_version=model_version,
            dataset_name=f"synthetic_batch_strength_{strength:.1f}",
        )

        balanced_accuracy = balanced_accuracy_score(y_test, predictions)

        rows.append(
            {
                "timestamp": batch_time.isoformat(),
                "drift_strength": strength,
                "balanced_accuracy": balanced_accuracy,
            }
        )

    return pd.DataFrame(rows)


def save_ks_plot(ks_results: pd.DataFrame) -> Path:
    output_path = FIGURES_DIR / "drift_ks_pvalues.png"

    if ks_results.empty:
        return output_path

    plt.figure(figsize=(8, 4))
    plt.bar(ks_results["feature"], ks_results["p_value"])
    plt.axhline(0.05, linestyle="--")
    plt.title("KS Test p-values by Feature")
    plt.xlabel("Feature")
    plt.ylabel("p-value")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def save_metric_timeseries_plot(metric_timeseries: pd.DataFrame) -> Path:
    output_path = FIGURES_DIR / "drift_balanced_accuracy_timeseries.png"

    plt.figure(figsize=(8, 4))
    plt.plot(
        metric_timeseries["timestamp"],
        metric_timeseries["balanced_accuracy"],
        marker="o",
    )
    plt.title("Balanced Accuracy under Synthetic Drift")
    plt.xlabel("Synthetic Timestamp")
    plt.ylabel("Balanced Accuracy")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


def run_monitoring(
    model_path: Path = DEFAULT_MODEL_PATH,
    train_path: Path = DEFAULT_TRAIN_PATH,
    test_path: Path = DEFAULT_TEST_PATH,
    log_path: Path = DEFAULT_LOG_PATH,
) -> dict:
    setup_directories()
    setup_logger(log_path)

    model_path = Path(model_path)

    model = load_model(model_path)
    model_version = get_model_version(model_path)

    X_train, X_test, y_train, y_test = load_train_test_data(
        train_path=train_path,
        test_path=test_path,
    )

    original_performance = compute_performance(
        model=model,
        X=X_test,
        y=y_test,
        model_version=model_version,
        dataset_name="original_test",
    )

    X_drifted = create_drifted_dataset(X_test, strength=1.0)

    drifted_performance = compute_performance(
        model=model,
        X=X_drifted,
        y=y_test,
        model_version=model_version,
        dataset_name="drifted_test",
    )

    performance_comparison = pd.DataFrame(
        [
            original_performance,
            drifted_performance,
        ]
    )

    original_score = original_performance["balanced_accuracy"]
    drifted_score = drifted_performance["balanced_accuracy"]

    performance_comparison["balanced_accuracy_change_vs_original"] = (
        performance_comparison["balanced_accuracy"] - original_score
    )

    ks_results = run_ks_drift_test(
        X_train=X_train,
        X_drifted=X_drifted,
        continuous_features=CONTINUOUS_FEATURES,
        alpha=0.05,
    )

    metric_timeseries = create_metric_timeseries(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_version=model_version,
    )

    performance_path = TABLES_DIR / "drift_performance_comparison.csv"
    ks_path = TABLES_DIR / "drift_ks_results.csv"
    timeseries_path = TABLES_DIR / "drift_metric_timeseries.csv"

    performance_comparison.to_csv(performance_path, index=False)
    ks_results.to_csv(ks_path, index=False)
    metric_timeseries.to_csv(timeseries_path, index=False)

    ks_plot_path = save_ks_plot(ks_results)
    timeseries_plot_path = save_metric_timeseries_plot(metric_timeseries)

    logging.info("monitoring_completed=True")
    logging.info("performance_path=%s", performance_path)
    logging.info("ks_path=%s", ks_path)
    logging.info("timeseries_path=%s", timeseries_path)
    logging.info("ks_plot_path=%s", ks_plot_path)
    logging.info("timeseries_plot_path=%s", timeseries_plot_path)

    print("\n=== Performance comparison ===")
    print(performance_comparison)

    print("\n=== KS drift test results ===")
    print(ks_results)

    print("\n=== Metric time series ===")
    print(metric_timeseries)

    print("\nSaved files:")
    print(performance_path)
    print(ks_path)
    print(timeseries_path)
    print(ks_plot_path)
    print(timeseries_plot_path)
    print(log_path)

    return {
        "performance_comparison": performance_comparison,
        "ks_results": ks_results,
        "metric_timeseries": metric_timeseries,
        "performance_path": performance_path,
        "ks_path": ks_path,
        "timeseries_path": timeseries_path,
        "ks_plot_path": ks_plot_path,
        "timeseries_plot_path": timeseries_plot_path,
        "log_path": log_path,
        "original_balanced_accuracy": original_score,
        "drifted_balanced_accuracy": drifted_score,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run CardioCare monitoring and drift detection."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
    )

    parser.add_argument(
        "--train-path",
        type=str,
        default=str(DEFAULT_TRAIN_PATH),
    )

    parser.add_argument(
        "--test-path",
        type=str,
        default=str(DEFAULT_TEST_PATH),
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default=str(DEFAULT_LOG_PATH),
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_monitoring(
        model_path=Path(args.model_path),
        train_path=Path(args.train_path),
        test_path=Path(args.test_path),
        log_path=Path(args.log_path),
    )