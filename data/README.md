# Data

This folder contains the UCI Heart Disease dataset files.

For the first implementation pass, use `processed.cleveland.data` because it has the standard 14-column format, 303 rows, and only a small number of missing values.

Expected column order:

```text
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
```

The original target is multiclass. For this project, convert it to binary:

```text
0 -> no heart disease
1, 2, 3, 4 -> heart disease
```

Missing values are marked as `?` in the raw file and should be loaded with `na_values="?"`.
