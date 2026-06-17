# CardioCare Notebook Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the repository structure and notebook skeleton required by the CardioCare final project so the full ML pipeline can be filled in cell by cell.

**Architecture:** The repo will start with a single EDA/preprocessing notebook and a project tree that matches the PDF submission checklist. The notebook will contain sectioned markdown and code cells for loading data, EDA, preprocessing design, and reusable pipeline hooks, while the remaining required files will exist as lightweight stubs ready for implementation.

**Tech Stack:** Jupyter notebook JSON, Python 3.10+, pandas, numpy, scikit-learn, mlflow, unittest, scipy, logging, Docker, GitHub Actions.

---

### Task 1: Create the submission tree

**Files:**
- Create: `notebooks/01_eda_preprocessing.ipynb`
- Create: `src/preprocessing.py`
- Create: `src/train.py`
- Create: `src/inference.py`
- Create: `src/monitor.py`
- Create: `tests/test_pipeline.py`
- Create: `data/.gitkeep`
- Create: `docs/.gitkeep`

- [ ] **Step 1: Create the empty project folders and placeholder files**

```text
notebooks/
src/
tests/
data/
docs/
```

- [ ] **Step 2: Verify the tree exists**

Run: `Get-ChildItem -Recurse | Select-Object FullName`
Expected: the listed folders and starter files are present.

- [ ] **Step 3: Commit**

```bash
git add notebooks src tests data docs
git commit -m "chore: scaffold cardiocare project tree"
```

### Task 2: Add the EDA notebook skeleton

**Files:**
- Create: `notebooks/01_eda_preprocessing.ipynb`

- [ ] **Step 1: Add a notebook with section headers and starter cells**

```json
{
  "cells": [
    {"cell_type": "markdown", "metadata": {}, "source": ["# 01. EDA and Preprocessing"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## 1) Project goal and dataset"]},
    {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["from pathlib import Path\nimport pandas as pd\nimport numpy as np\n"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## 2) Load data"]},
    {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["# Load the selected heart disease dataset here\n"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## 3) EDA: head, info, describe, target balance"]},
    {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["# head(), info(), describe(), value_counts(normalize=True)\n"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## 4) Missing values and outliers"]},
    {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["# Missing-value and outlier checks\n"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## 5) Preprocessing pipeline design"]},
    {"cell_type": "code", "metadata": {}, "execution_count": null, "outputs": [], "source": ["# Build reusable sklearn preprocessing components\n"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["## 6) EDA summary and preprocessing decisions"]},
    {"cell_type": "markdown", "metadata": {}, "source": ["Write the short conclusion here.\n"]}
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.10"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

- [ ] **Step 2: Verify the notebook opens in Jupyter**

Run: `jupyter notebook notebooks/01_eda_preprocessing.ipynb`
Expected: the notebook shows the section layout without JSON errors.

- [ ] **Step 3: Commit**

```bash
git add notebooks/01_eda_preprocessing.ipynb
git commit -m "docs: add cardiocare EDA notebook skeleton"
```

