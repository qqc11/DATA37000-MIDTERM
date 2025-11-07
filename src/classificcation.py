#%%
# ==============================================================
# Classification Models with Timing (Logistic, LinearSVC, RF)
# ==============================================================
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# 1. Load dataset
df = pd.read_csv("medical_insurance.csv")

# 2. Define features and target
y = df["is_high_risk"]
X = df.drop(columns=["annual_medical_cost", "is_high_risk", "person_id"])

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocessing: standardize numeric features, encode categorical
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# 5. Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "Linear SVM": LinearSVC(max_iter=3000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, n_jobs=-1, random_state=42)
}

# 6. Train, time, and evaluate models
results = []
for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("clf", model)])
    start = time.time()
    pipe.fit(X_train, y_train)
    end = time.time()
    
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    duration = end - start
    
    results.append((name, acc, duration))
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Training + Prediction Time: {duration:.2f} sec")

# 7. Summary table
print("\n=== Summary ===")
for name, acc, t in results:
    print(f"{name:<20} | Accuracy: {acc:.4f} | Time: {t:.2f} sec")


#%%
# ==============================================================
# Task: Predict Top-10% High-Cost Members (Harder Binary Task)
# - Label = 1 if annual_medical_cost >= 90th percentile
# - Drop potential leakage features (claims/cost-derived)
# - Metrics: ROC-AUC, PR-AUC, best-threshold F1; timing
# ==============================================================

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

# 1) Load data
df = pd.read_csv("medical_insurance.csv")

# 2) Build hard label: top-10% by annual_medical_cost
q90 = df["annual_medical_cost"].quantile(0.90)
y = (df["annual_medical_cost"] >= q90).astype(int)

# 3) Remove outcome/leakage-like features:
#    Anything that is directly cost/claims related is excluded from predictors.
leak_like = [
    "annual_medical_cost", "total_claims_paid", "claims_count",
    "proc_imaging_count", "proc_surgery_count", "proc_physio_count",
    "proc_consult_count", "proc_lab_count"
]
# Optionally remove engineered risk summaries that might be too close to outcome
maybe_leaky_scores = ["risk_score", "had_major_procedure", "is_high_risk"]
drop_cols = [c for c in leak_like + maybe_leaky_scores if c in df.columns]

X = df.drop(columns=drop_cols + ["person_id"])  # keep demographic/behavior/labs, etc.

# 4) Train/test split (stratify to preserve positive rate)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5) Preprocess: scale numeric + one-hot categorical
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# 6) Define compact, fast models
models = {
    "Logistic": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
    "LinearSVM": LinearSVC(max_iter=3000, random_state=42),  # no predict_proba
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=12, n_jobs=-1, random_state=42
    )
}

def eval_binary(pipe, name):
    """Fit, time, and compute ROC-AUC, PR-AUC, and best-threshold F1."""
    start = time.time()
    pipe.fit(X_train, y_train)
    fit_time = time.time() - start

    # Scores/probabilities for metrics
    # For LinearSVC, use decision_function; for others, use predict_proba
    if hasattr(pipe.named_steps["clf"], "predict_proba"):
        scores = pipe.predict_proba(X_test)[:, 1]
    else:
        scores = pipe.decision_function(X_test)

    # Threshold sweep for best F1
    precision, recall, thr = precision_recall_curve(y_test, scores)
    f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = thr[best_idx]
    y_pred_best = (scores >= best_thr).astype(int)

    # Metrics
    roc = roc_auc_score(y_test, scores)
    pr = average_precision_score(y_test, scores)  # PR-AUC
    f1 = f1_score(y_test, y_pred_best)

    return {
        "name": name,
        "roc_auc": roc,
        "pr_auc": pr,
        "best_f1": f1,
        "best_thr": float(best_thr),
        "time_sec": fit_time
    }

# 7) Run and summarize
results = []
for name, clf in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
    res = eval_binary(pipe, name)
    results.append(res)
    print(f"\n=== {name} ===")
    print(f"ROC-AUC: {res['roc_auc']:.4f} | PR-AUC: {res['pr_auc']:.4f} | "
          f"Best-F1: {res['best_f1']:.4f} @ thr={res['best_thr']:.3f} | "
          f"Time: {res['time_sec']:.2f}s")

# Optional: pretty dataframe
summary = pd.DataFrame(results).sort_values(by="pr_auc", ascending=False)
print("\nSummary (sorted by PR-AUC):")
print(summary)

#%%
# ==============================================================
# Task: 3-Class Cost Tiering (Low / Mid / High)
# - Tiers by quantiles: <=33%, 33~66%, >=66%
# - Metrics: macro-F1 (class-balanced), timing
# ==============================================================

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("medical_insurance.csv")

# Create 3-class label from annual cost
q33, q66 = df["annual_medical_cost"].quantile([0.33, 0.66])
def to_tier(v):
    if v <= q33: return 0  # Low
    if v >= q66: return 2  # High
    return 1               # Mid

y = df["annual_medical_cost"].apply(to_tier)

# Drop leakage-like columns
leak_like = [
    "annual_medical_cost", "total_claims_paid", "claims_count",
    "proc_imaging_count", "proc_surgery_count", "proc_physio_count",
    "proc_consult_count", "proc_lab_count"
]
maybe_leaky_scores = ["risk_score", "had_major_procedure", "is_high_risk"]
drop_cols = [c for c in leak_like + maybe_leaky_scores if c in df.columns]
X = df.drop(columns=drop_cols + ["person_id"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

models = {
    "Multinomial Logistic": LogisticRegression(
        multi_class="multinomial", max_iter=1000, n_jobs=-1, random_state=42
    ),
    "KNN (k=9)": KNeighborsClassifier(n_neighbors=9, weights="distance"),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=14, n_jobs=-1, random_state=42
    )
}

for name, clf in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("clf", clf)])
    start = time.time()
    pipe.fit(X_train, y_train)
    secs = time.time() - start

    y_pred = pipe.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"\n=== {name} ===")
    print(f"Macro-F1: {macro_f1:.4f} | Time: {secs:.2f}s")
    print(classification_report(y_test, y_pred, digits=3))


#%%
# ==============================================================
# Markers only: no per-point text labels
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data (same as前面)
data = {
    "Task Type": [
        "Regression", "Regression", "Regression",
        "Binary Classification", "Binary Classification", "Binary Classification",
        "Multi-Class Classification", "Multi-Class Classification", "Multi-Class Classification"
    ],
    "Model": [
        "Linear Regression", "Random Forest Regressor", "XGBoost Regressor",
        "Logistic Regression", "Linear SVM", "Random Forest Classifier",
        "Multinomial Logistic", "KNN (k=9)", "Random Forest (Multi-Class)"
    ],
    "Metric Type": [
        "R²", "R²", "R²",
        "ROC-AUC", "ROC-AUC", "ROC-AUC",
        "Macro-F1", "Macro-F1", "Macro-F1"
    ],
    "Metric Value": [
        0.9664, 0.9982, 0.9759,
        1.0000, 1.0000, 0.9998,
        0.9964, 0.6208, 0.9765
    ],
    "Time (s)": [0.0, 0.0, 0.0, 1.34, 0.35, 2.55, 1.71, 0.12, 4.42]
}
df = pd.DataFrame(data)

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df,
    x="Time (s)", y="Metric Value",
    hue="Task Type",          # color by task
    style="Metric Type",      # marker shape by metric
    s=120, edgecolor="black"
)

# No per-point text labels — markers only
# (Do NOT call plt.text for any point)

plt.title("Model Performance vs Training Time", fontsize=14)
plt.xlabel("Training + Prediction Time (seconds)")
plt.ylabel("Metric Value (R² / ROC-AUC / Macro-F1)")
plt.xlim(-0.1, df["Time (s)"].max() + 0.5)
plt.ylim(0.55, 1.05)
plt.grid(True, alpha=0.3)

# If you also want to hide the legend entirely, uncomment the next line:
# plt.legend().remove()

plt.tight_layout()
plt.show()


