# 202221066_final_machine_learning

CardioCare final project scaffold for an end-to-end heart disease prediction system.

## Environment

Recommended Python version: 3.10.13

```powershell
pip install -r requirements.txt
```

## Data

The UCI Heart Disease dataset is stored under `data/`.

Start with `data/processed.cleveland.data` for the first implementation pass. It uses the standard 14-column format and should be loaded with `na_values="?"`.

## Planned Reproduction Flow

```powershell
python src/train.py
python -m unittest discover -s tests
docker build -t cardiocare:1.0 .
python src/monitor.py
```

## Project Structure

- `data/`
- `notebooks/01_eda_preprocessing.ipynb`
- `src/preprocessing.py`
- `src/train.py`
- `src/inference.py`
- `src/monitor.py`
- `tests/test_pipeline.py`
- `mlruns/`
- `Dockerfile`
- `requirements.txt`
- `.github/workflows/ci.yml`
- `report.pdf`
