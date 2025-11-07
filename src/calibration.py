# ==============================================================
# Calibration Curve for Classification Models
# ==============================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

# 1. Load dataset
df = pd.read_csv("medical_insurance.csv")

# 2. Define features and target
y = df["is_high_risk"]
X = df.drop(columns=["annual_medical_cost", "is_high_risk", "person_id"])

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Preprocess: scale numeric + encode categorical
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# 5. Define models (SVM wrapped with calibration for probability)
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
    "Linear SVM": CalibratedClassifierCV(LinearSVC(max_iter=3000, random_state=42))
}

# 6. Fit and plot calibration curves
plt.figure(figsize=(6, 6))

for name, model in models.items():
    pipe = Pipeline([("preprocess", preprocessor), ("clf", model)])
    pipe.fit(X_train, y_train)

    # Predict calibrated probabilities
    y_prob = pipe.predict_proba(X_test)[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)

    plt.plot(prob_pred, prob_true, marker="o", label=name)

# Reference line: perfect calibration
plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

plt.title("Calibration Curve for Classification Models")
plt.xlabel("Predicted Probability")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()