import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from inference import validate_input


class TestCardioCarePipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 모든 테스트에서 공통으로 사용할 모델 파일과 샘플 입력을 한 번만 불러옴
        cls.model_path = PROJECT_ROOT / "models" / "final_model.joblib"
        cls.sample_input_path = PROJECT_ROOT / "data" / "sample_input.csv"

        if not cls.model_path.exists():
            raise FileNotFoundError(
                f"Missing model file: {cls.model_path}"
            )

        if not cls.sample_input_path.exists():
            raise FileNotFoundError(
                f"Missing sample input file: {cls.sample_input_path}"
            )

        cls.model = joblib.load(cls.model_path)
        cls.sample_input = pd.read_csv(cls.sample_input_path)

    def test_prediction_shape(self):
        # 입력 행 수와 예측 결과 행 수가 정확히 일치하는지 확인함
        predictions = self.model.predict(self.sample_input)

        self.assertEqual(
            predictions.shape[0],
            self.sample_input.shape[0],
        )

    def test_prediction_probability_range(self):
        # 확률 예측값이 0 이상 1 이하이고 각 행의 합이 1인지 확인함
        probabilities = self.model.predict_proba(self.sample_input)

        self.assertTrue(
            np.all(probabilities >= 0),
            "Some probabilities are smaller than 0.",
        )

        self.assertTrue(
            np.all(probabilities <= 1),
            "Some probabilities are greater than 1.",
        )

        row_sums = probabilities.sum(axis=1)

        self.assertTrue(
            np.allclose(row_sums, 1.0, atol=1e-6),
            "Each probability row should sum to 1.",
        )

    def test_clinical_value_range(self):
        # 정상 입력은 통과하고 비정상 임상값은 예외가 발생하는지 확인함
        valid_input = self.sample_input.copy()

        try:
            validate_input(valid_input)
        except ValueError as error:
            self.fail(
                f"validate_input raised ValueError unexpectedly: {error}"
            )

        invalid_input = self.sample_input.copy()
        invalid_input.loc[invalid_input.index[0], "chol"] = 999

        with self.assertRaises(ValueError):
            validate_input(invalid_input)

    def test_deterministic_prediction(self):
        # 같은 입력에 대해 예측 결과가 항상 동일한지 확인함
        first_prediction = self.model.predict(self.sample_input)
        second_prediction = self.model.predict(self.sample_input)

        np.testing.assert_array_equal(
            first_prediction,
            second_prediction,
        )


if __name__ == "__main__":
    unittest.main()
