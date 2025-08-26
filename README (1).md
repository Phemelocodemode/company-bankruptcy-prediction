# Company Bankruptcy Prediction (Classification)

This project builds machine learning models to predict **company bankruptcy** using a tabular dataset.
It demonstrates a full workflow: data cleaning, train/test split, ML pipelines, model comparison, evaluation,
and presentation-ready charts and outputs.

## Dataset
- Source: Kaggle (uploaded locally as `data.csv`)
- Rows: 6819
- Features: 95
- Detected target column: `Bankrupt?` (converted to 0/1)

## Methods
- **Logistic Regression** (with scaling)
- **Random Forest** (class_weight balanced)
- Stratified train/test split
- Missing values handled (median/mode)

## Key Results (Test Set)
- **Logistic Regression** — accuracy: 0.962, precision: 0.300, recall: 0.136, f1: 0.187, roc_auc: 0.875
- **Random Forest** — accuracy: 0.971, precision: 0.692, recall: 0.205, f1: 0.316, roc_auc: 0.951

> Class distribution in the full dataset: {0: 6599, 1: 220}

## Files
- `bankruptcy_prediction.py` — full training & evaluation script
- `charts/`
  - `class_distribution.png`
  - `confusion_matrix_rf.png`
  - `feature_importances_top15.png`
- `outputs/`
  - `predictions_rf.csv`
  - `evaluation_metrics.json`
  - `classification_report_lr.txt`
  - `classification_report_rf.txt`

## How to Run (Locally)
1. Ensure you have Python 3.9+ and install packages:
   ```bash
   pip install -r requirements.txt
   ```
   or
   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```
2. Place `data.csv` in the project root (same folder as the script).
3. Run:
   ```bash
   python bankruptcy_prediction.py
   ```
4. Check the `charts/` and `outputs/` folders for results.

## Notes
- Logistic Regression is simple and interpretable.
- Random Forest typically provides stronger performance and exposes feature importances (see chart).
- For imbalanced datasets, consider threshold tuning or metrics like ROC-AUC and PR-AUC.

---
*Generated on: 2025-08-26 20:30:59*
