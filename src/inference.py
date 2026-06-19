from pathlib import Path
import argparse
import logging

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_input.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "inference_predictions.csv"
DEFAULT_LOG_PATH = PROJECT_ROOT / "reports" / "inference.log"


CLINICAL_RANGES = {
    "age": (0, 120),
    "trestbps": (0, 250),
    "chol": (0, 600),
    "thalach": (0, 250),
    "oldpeak": (0, 10),
}


def setup_logger(log_path=DEFAULT_LOG_PATH):
    """
    추론 로그 저장을 위한 logger 설정 함수.
    reports/inference.log에 추론 기록을 남긴다.
    """

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def validate_input(df: pd.DataFrame) -> None:
    """
    입력 데이터의 기본 임상 범위를 검증한다.

    이 함수는 test_pipeline.py에서도 사용된다.
    예를 들어 chol 값이 600을 초과하면 ValueError를 발생시킨다.
    """

    missing_columns = [
        col for col in CLINICAL_RANGES.keys()
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required clinical columns: {missing_columns}")

    for col, (lower, upper) in CLINICAL_RANGES.items():
        invalid_mask = df[col].notna() & (
            (df[col] < lower) | (df[col] > upper)
        )

        if invalid_mask.any():
            invalid_values = df.loc[invalid_mask, col].tolist()

            raise ValueError(
                f"Column '{col}' has values outside clinical range "
                f"[{lower}, {upper}]: {invalid_values}"
            )


def load_model(model_path=DEFAULT_MODEL_PATH):
    """
    joblib으로 저장된 최종 모델을 불러온다.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)


def run_inference(
    model_path=DEFAULT_MODEL_PATH,
    input_path=DEFAULT_INPUT_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    log_path=DEFAULT_LOG_PATH,
):
    """
    저장된 모델과 입력 CSV를 이용해 추론을 수행한다.

    출력 파일에는 원본 입력값, 예측 클래스, 심장병 확률을 함께 저장한다.
    """

    setup_logger(log_path)

    model = load_model(model_path)

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(input_path)

    validate_input(input_df)

    predictions = model.predict(input_df)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)
        heart_disease_probability = probabilities[:, 1]
    else:
        probabilities = None
        heart_disease_probability = [None] * len(predictions)

    result_df = input_df.copy()
    result_df["prediction"] = predictions
    result_df["heart_disease_probability"] = heart_disease_probability

    result_df.to_csv(output_path, index=False)

    logging.info("model_path=%s", model_path)
    logging.info("input_path=%s", input_path)
    logging.info("output_path=%s", output_path)
    logging.info("input_shape=%s", input_df.shape)
    logging.info("predictions=%s", predictions.tolist())

    if probabilities is not None:
        logging.info("probabilities=%s", probabilities.tolist())

    print(result_df)

    return result_df


def parse_args():
    """
    CLI 인자를 파싱한다.

    예시:
    python src/inference.py
    python src/inference.py --input-path data/sample_input.csv
    """

    parser = argparse.ArgumentParser(
        description="Run CardioCare heart disease inference."
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to trained joblib model.",
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to input CSV file.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to save prediction CSV file.",
    )

    parser.add_argument(
        "--log-path",
        type=str,
        default=str(DEFAULT_LOG_PATH),
        help="Path to save inference log file.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_inference(
        model_path=args.model_path,
        input_path=args.input_path,
        output_path=args.output_path,
        log_path=args.log_path,
    )