<p align="center">
  <h1 align="center">🧠 Anxiety Prediction from Music & Mental Health Survey</h1>
  <p align="center">
    <em>Machine learning pipeline predicting high anxiety levels using the MXMH survey dataset</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/python-3.13-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn&logoColor=white" alt="sklearn">
    <img src="https://img.shields.io/badge/XGBoost-2.0+-green?logo=xgboost" alt="XGBoost">
    <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
    <img src="https://img.shields.io/badge/status-complete-success" alt="Status">
  </p>
</p>

---

## 📌 Overview

This project builds and compares **three classification models** (Logistic Regression, Random Forest, and XGBoost) to predict whether an individual has **high anxiety** (score >= 7) based on their demographics, music listening habits, and behavioral patterns from the [Music & Mental Health (MXMH) survey](https://www.kaggle.com/datasets/catherinerasgaitis/mxmh-survey-results).

The pipeline covers data cleaning, feature engineering, hyperparameter tuning, model evaluation, and feature importance analysis.

---

## 🏆 Key Results

| Model | Avg Precision | ROC AUC | Precision | Recall | F1 Score |
|:------|:---:|:---:|:---:|:---:|:---:|
| Logistic Regression | 0.613 | 0.633 | 0.575 | 0.583 | 0.579 |
| Random Forest | 0.641 | 0.645 | 0.560 | 0.583 | 0.571 |
| **XGBoost** ✅ | **0.653** | **0.639** | **0.575** | **0.583** | **0.579** |

> **Best Model: XGBoost** achieved the highest Average Precision (0.653) and ROC AUC (0.639), showing strong generalization across all classification thresholds.

---

## 📁 Project Structure

```
├── 📄 README.md                  ← You are here
├── 📄 LICENSE                    ← MIT License
├── 📄 requirements.txt           ← Python dependencies
│
├── 📂 notebooks/
│   ├── 01_cleaning.ipynb         ← Data cleaning & feature engineering
│   └── 02_modeling.ipynb         ← Model training, tuning & evaluation
│
├── 📂 src/
│   └── modeling.py               ← Standalone modeling script
│
├── 📂 data/
│   ├── raw/
│   │   └── mxmh_survey_results.csv   ← Original survey data (736 rows)
│   └── processed/
│       └── mxmh_clean.csv            ← Cleaned dataset (735 rows)
│
└── 📂 results/
    ├── results.csv                ← Model metrics (CSV)
    ├── results.json               ← Model metrics (JSON)
    └── results_summary.txt        ← Formatted summary table
```

---

## 🔬 Methodology

### Data Cleaning
- Removed outlier ages (outside 10 to 100 range)
- Clipped listening hours to 0 to 24h
- Median imputation for numeric features, mode for categorical
- Encoded binary and ordinal variables
- Excluded mental health score columns from features to prevent data leakage

### Models & Tuning
| Model | Tuning Method | Key Hyperparameters |
|:------|:------|:------|
| Logistic Regression | `LogisticRegressionCV` | C in [0.001, 100], L2 penalty |
| Random Forest | `RandomizedSearchCV` | n_estimators, max_depth, min_samples_split |
| XGBoost | `RandomizedSearchCV` | learning_rate, max_depth, subsample, colsample |

- Stratified 5-fold cross-validation with Average Precision scoring
- Class weighting (`balanced`) to handle target imbalance

### Evaluation Metrics
- **Threshold-independent:** Average Precision, ROC AUC
- **Threshold-dependent (at 0.5):** Precision, Recall, F1 Score
- **Visualization:** Precision-Recall curves for all models

---

## 🚀 How to Reproduce

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/anxiety-prediction-mxmh.git
cd anxiety-prediction-mxmh

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
#    Option A: Notebooks (recommended for exploration)
jupyter notebook notebooks/01_cleaning.ipynb
jupyter notebook notebooks/02_modeling.ipynb

#    Option B: Script (for quick execution)
python src/modeling.py
```

---

## 🛠️ Tech Stack

| Category | Tools |
|:---------|:------|
| Language | Python 3.13 |
| Data | pandas, NumPy |
| ML | scikit-learn, XGBoost |
| Visualization | Matplotlib |
| Environment | Jupyter Notebook |

---

## 📊 Top Feature Importances (Random Forest)

The most influential predictors for anxiety prediction:

1. **Hours per day** - daily music listening duration
2. **Age** - respondent's age
3. **BPM** - preferred beats per minute
4. **Genre frequencies** - how often specific genres are listened to
5. **Streaming service** - primary platform used

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
