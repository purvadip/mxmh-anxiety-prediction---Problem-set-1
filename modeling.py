# =============================================================================
# Machine Learning Pipeline: Anxiety Prediction
# Builds Logistic Regression, Random Forest, and XGBoost classifiers
# to predict high anxiety levels from the MXMH survey data.
# =============================================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. Data Loading and Target Definition
# ─────────────────────────────────────────────
df = pd.read_csv("mxmh_clean.csv")

target = "high_anxiety"
if target not in df.columns:
    raise ValueError(f"{target} not found. Create it in cleaning or add it here.")

# Avoid leakage: drop all mental health score columns from features
leak_cols = ["anxiety", "depression", "insomnia", "ocd", target]
leak_cols = [c for c in leak_cols if c in df.columns]

X = df.drop(columns=leak_cols)
y = df[target].astype(int)

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True)}")

# ─────────────────────────────────────────────
# 2. Preprocessing
# ─────────────────────────────────────────────
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "bool"]).columns.tolist()

print(f"\nNumeric features ({len(numeric_features)}): {numeric_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# For Linear Models (scaling required)
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_linear = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# For Tree Models (no scaling needed)
numeric_transformer_tree = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocess_tree = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_tree, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ─────────────────────────────────────────────
# 3. Model Training and Tuning
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# Models
log_clf = LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs")

rf_clf = RandomForestClassifier(
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

xgb_clf = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

# Pipelines
pipe_log = Pipeline(steps=[("prep", preprocess_linear), ("model", log_clf)])
pipe_rf = Pipeline(steps=[("prep", preprocess_tree), ("model", rf_clf)])
pipe_xgb = Pipeline(steps=[("prep", preprocess_tree), ("model", xgb_clf)])

# CV + Tuning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

param_log = {
    "model__C": np.logspace(-3, 2, 20)
}

param_rf = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [5, 10, 15, 20, None],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2", None]
}

param_xgb = {
    "model__n_estimators": [200, 400, 600],
    "model__max_depth": [2, 3, 4, 5],
    "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.85, 1.0],
    "model__colsample_bytree": [0.7, 0.85, 1.0],
    "model__min_child_weight": [1, 3, 5]
}

# Use Average Precision for imbalanced classification
search_log = RandomizedSearchCV(
    pipe_log, param_distributions=param_log,
    n_iter=20, scoring="average_precision",
    cv=cv, random_state=42, n_jobs=-1
)

search_rf = RandomizedSearchCV(
    pipe_rf, param_distributions=param_rf,
    n_iter=25, scoring="average_precision",
    cv=cv, random_state=42, n_jobs=-1
)

search_xgb = RandomizedSearchCV(
    pipe_xgb, param_distributions=param_xgb,
    n_iter=25, scoring="average_precision",
    cv=cv, random_state=42, n_jobs=-1
)

print("\n" + "="*60)
print("TRAINING MODELS (with hyperparameter tuning)")
print("="*60)

print("\n[1/3] Tuning Logistic Regression...")
search_log.fit(X_train, y_train)

print("[2/3] Tuning Random Forest...")
search_rf.fit(X_train, y_train)

print("[3/3] Tuning XGBoost...")
search_xgb.fit(X_train, y_train)

print("\n--- Best Hyperparameters ---")
print("Logistic Regression:", search_log.best_params_)
print("Random Forest:", search_rf.best_params_)
print("XGBoost:", search_xgb.best_params_)

print("\n--- Cross-Validation Average Precision ---")
print(f"Logistic Regression: {search_log.best_score_:.4f}")
print(f"Random Forest:       {search_rf.best_score_:.4f}")
print(f"XGBoost:             {search_xgb.best_score_:.4f}")

# ─────────────────────────────────────────────
# 4. Evaluation on Test Set
# ─────────────────────────────────────────────
def eval_model(model, X_te, y_te, name="model"):
    proba = model.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)

    ap = average_precision_score(y_te, proba)
    auc = roc_auc_score(y_te, proba)
    prec = precision_score(y_te, preds, zero_division=0)
    rec = recall_score(y_te, preds, zero_division=0)
    f1 = f1_score(y_te, preds, zero_division=0)
    cm = confusion_matrix(y_te, preds)

    print(f"\n{'='*60}")
    print(f" {name}")
    print(f"{'='*60}")
    print(f"  Average Precision : {ap:.4f}")
    print(f"  ROC AUC           : {auc:.4f}")
    print(f"  Precision         : {prec:.4f}")
    print(f"  Recall            : {rec:.4f}")
    print(f"  F1 Score          : {f1:.4f}")
    print(f"  Confusion Matrix  :\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_te, preds, zero_division=0)}")

    return {"model": name, "ap": ap, "roc_auc": auc, "precision": prec, "recall": rec, "f1": f1}

best_log = search_log.best_estimator_
best_rf = search_rf.best_estimator_
best_xgb = search_xgb.best_estimator_

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

results = []
results.append(eval_model(best_log, X_test, y_test, "Logistic Regression"))
results.append(eval_model(best_rf, X_test, y_test, "Random Forest"))
results.append(eval_model(best_xgb, X_test, y_test, "XGBoost"))

# ─────────────────────────────────────────────
# 5. Summary Comparison Table
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
print("\n" + "="*60)
print("MODEL COMPARISON SUMMARY")
print("="*60)
print(results_df.to_string(index=False))

# Find overall best model
best_idx = results_df["f1"].idxmax()
print(f"\n>>> Best model by F1 score: {results_df.loc[best_idx, 'model']} "
      f"(F1 = {results_df.loc[best_idx, 'f1']:.4f})")

# ─────────────────────────────────────────────
# 6. Feature Importance (Random Forest)
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("TOP 15 FEATURE IMPORTANCES (Random Forest)")
print("="*60)

rf_model = best_rf.named_steps['model']
preprocessor = best_rf.named_steps['prep']

feature_names = []
feature_names.extend(numeric_features)
if len(categorical_features) > 0:
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    feature_names.extend(cat_encoder.get_feature_names_out(categorical_features))

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1][:15]

for rank, idx in enumerate(indices, 1):
    print(f"  {rank:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")

print("\n[DONE] All models trained and tested successfully!")
