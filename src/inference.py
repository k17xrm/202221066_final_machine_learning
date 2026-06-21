from pathlib import Path
import argparse
import logging

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] # 현재 파일의 절대 경로를 구한 뒤, 부모 디렉터리 중 한 단계 위를 프로젝트 루트로 설정

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "final_model.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "sample_input.csv"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "reports" / "inference_predictions.csv"
DEFAULT_LOG_PATH = PROJECT_ROOT / "reports" / "inference.log"

# 입력 데이터에서 검사할 임상 변수들의 허용 범위를 딕셔너리로 정의하기
# key: 컬럼 이름 , value : (최소값, 최대값) 튜플
CLINICAL_RANGES = {
    "age": (0, 120),
    "trestbps": (0, 250),
    "chol": (0, 600),
    "thalach": (0, 250),
    "oldpeak": (0, 10),
}

# 로그 설정을 담당하는 함수 
def setup_logger(log_path=DEFAULT_LOG_PATH):
  
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,  # INFO 레벨 이상의 로그를 저장
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )

# 입력 DataFrame이 모델 추론에 적절한지 검사하는 함수 -> 문제 있으면 ValueError 발생시킴 / 문제 없으면 아무것도 반환 x 
def validate_input(df: pd.DataFrame) -> None:
      # CLINICAL_RANGES에 정의된 필수 컬럼들이 df에 있는지 검사 -> 없는 컬럼들만 리스트로 모음
    missing_columns = [
        col for col in CLINICAL_RANGES.keys()
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required clinical columns: {missing_columns}")

    for col, (lower, upper) in CLINICAL_RANGES.items():
        # 해당 컬럼이 결측치가 아니면서 lower보다 작거나 upper보다 큰 값이 있는지 검사 -> 있으면 invalid_mask가 True인 Series 반환
        invalid_mask = df[col].notna() & (
            (df[col] < lower) | (df[col] > upper)
        )

        if invalid_mask.any():
            invalid_values = df.loc[invalid_mask, col].tolist()

            raise ValueError(
                f"Column '{col}' has values outside clinical range "
                f"[{lower}, {upper}]: {invalid_values}"
            )

# 저장된 모델 파일을 불러오는 함수
def load_model(model_path=DEFAULT_MODEL_PATH):
    
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return joblib.load(model_path)  # joblib.load()로 저장된 모델 객체를 불러와 반환


# 실제 추론 전체 과정을 수행하는 메인 함수 -> 모델 경로, 입력 CSV 경로, 출력 CSV 경로, 로그 경로를 인자로 받음
def run_inference(
    model_path=DEFAULT_MODEL_PATH,
    input_path=DEFAULT_INPUT_PATH,
    output_path=DEFAULT_OUTPUT_PATH,
    log_path=DEFAULT_LOG_PATH,
):
    setup_logger(log_path)  # 로그 시스템을 설정함

    model = load_model(model_path)

    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    input_df = pd.read_csv(input_path)

    # 입력 데이터에 필수 컬럼이 있는지 임상 수치가 허용 범위 안에 있는지 검사함
    validate_input(input_df)

    # 모델의 predict() 메서드를 사용해 예측값을 계산
    predictions = model.predict(input_df)

    
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df) # 각 클래스에 대한 예측 확률을 계산함 -> 결과 형태 : [샘플 수, 클래스 수]임
        heart_disease_probability = probabilities[:, 1] # 두 번째 클래스의 확률만 가져옴
    else:  # 모델이 predict_proba()를 지원하지 않는 경우
        
        probabilities = None # 확률 전체 None으로 둠
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
    # ??? ??? ?? ?? ??? ??? ??? ???
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
