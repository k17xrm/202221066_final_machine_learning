FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY tests/ ./tests/
COPY data/ ./data/
COPY models/ ./models/

RUN mkdir -p reports

CMD ["python", "src/inference.py", "--input-path", "data/sample_input.csv", "--model-path", "models/final_model.joblib", "--output-path", "reports/inference_predictions.csv", "--log-path", "reports/inference.log"]