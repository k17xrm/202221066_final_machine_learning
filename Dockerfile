# 저장된 모델로 추론을 실행하기 위한 가벼운 런타임 이미지
# 컨테이너 안에서는 재학습하지 않고, 저장된 코드와 데이터, 모델만 사용함
FROM python:3.10-slim

# 컨테이너 내부 작업 디렉터리를 /app으로 고정함
WORKDIR /app

COPY requirements.txt .

# 나머지 파일을 복사하기 전에 Python 의존성을 먼저 설치함
RUN pip install --no-cache-dir -r requirements.txt

# 추론과 검증에 필요한 소스 코드와 아티팩트를 복사함
COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/

# 추론 결과가 저장될 reports/ 폴더를 미리 생성함
RUN mkdir -p reports

# 기본 실행 명령으로 샘플 입력에 대한 추론을 수행하고 결과와 로그를 저장함
CMD ["python", "src/inference.py", "--input-path", "data/sample_input.csv", "--model-path", "models/final_model.joblib", "--output-path", "reports/inference_predictions.csv", "--log-path", "reports/inference.log"]
