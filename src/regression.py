# ==============================================================
# Regression Models: Predict Annual Medical Cost (Full Version)
# ==============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# 1. Load dataset
df = pd.read_csv("medical_insurance.csv")

# 2. Define features and target
y = df["annual_medical_cost"]
X = df.drop(columns=["annual_medical_cost", "is_high_risk", "person_id"])

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Preprocessing
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

# =========================
# 5. Linear Regression
# =========================
lin_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", LinearRegression())
])
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)

mse_lin = mean_squared_error(y_test, y_pred_lin)
rmse_lin = np.sqrt(mse_lin)

print("=== Linear Regression ===")
print("R²:", r2_score(y_test, y_pred_lin))
print("RMSE:", rmse_lin)

# =========================
# 6. Random Forest Regressor
# =========================
rf_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))
])
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

mse_rf = mean_squared_error(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

print("\n=== Random Forest Regressor ===")
print("R²:", r2_score(y_test, y_pred_rf))
print("RMSE:", rmse_rf)

# =========================
# 7. XGBoost Regressor
# =========================
xgb_model = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, random_state=42))
])
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mse_xgb)

print("\n=== XGBoost Regressor ===")
print("R²:", r2_score(y_test, y_pred_xgb))
print("RMSE:", rmse_xgb)
# ==============================================================
# Regression Performance Comparison: Predicted vs Actual
# ==============================================================

import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot comparing three regression models
plt.figure(figsize=(7, 6))

sns.scatterplot(x=y_test, y=y_pred_lin, alpha=0.3, label="Linear Regression", color="skyblue")
sns.scatterplot(x=y_test, y=y_pred_rf, alpha=0.3, label="Random Forest", color="orange")
sns.scatterplot(x=y_test, y=y_pred_xgb, alpha=0.3, label="XGBoost", color="green")

# Plot reference line y = x
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         'r--', lw=2, label="Perfect Fit")

plt.xlabel("Actual Annual Medical Cost")
plt.ylabel("Predicted Annual Medical Cost")
plt.title("Predicted vs Actual Annual Medical Cost (Three Models)")
plt.legend()
plt.tight_layout()
plt.show()