# ==============================================================
# Simplified EDA Visualization for Medical Insurance Dataset
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
df = pd.read_csv("medical_insurance.csv")

# 2. Select categorical and key numeric features
cat_cols = df.select_dtypes(include=['object']).columns

# 3. Plot bar charts for first few categorical variables
for c in cat_cols[:5]:  # Show first 5 categorical columns
    plt.figure(figsize=(6,3))
    sns.countplot(y=c, data=df, order=df[c].value_counts().index)
    plt.title(f"Distribution of {c}", fontsize=12)
    plt.tight_layout()
    plt.show()

# 4. Scatter plot: Risk score vs annual medical cost
plt.figure(figsize=(6,4))
sns.scatterplot(x="risk_score", y="annual_medical_cost", data=df, alpha=0.4)
plt.title("Risk Score vs Annual Medical Cost", fontsize=12)
plt.xlabel("Risk Score")
plt.ylabel("Annual Medical Cost")
plt.tight_layout()
plt.show()