# Medical Insurance ML Project

## Overview
Classical ML for medical cost prediction and risk classification on a 100,000 × 54 dataset.

## Tasks
- Regression: Predict annual medical cost
- Binary classification: High-risk membership
- Multi-class classification: Cost tiers (low/mid/high)
- Calibration & interpretability

## Key Results
| Task | Best Model | Metric | Score |
|-----|------------|--------|-------|
| Regression | Random Forest | R² | 0.9982 |
| Binary Classification | Linear SVM | ROC-AUC | 1.0000 |
| Multi-Class | Multinomial Logistic | Macro-F1 | 0.9964 |

## How to Run
```bash
pip install -r requirements.txt
python src/regression_models.py
python src/classification_models.py
python src/calibration_analysis.py
