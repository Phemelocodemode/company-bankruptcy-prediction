# Company Bankruptcy Prediction  

This project uses machine learning to predict company bankruptcy from financial indicators.

## Dataset
- Source: Kaggle "Company Bankruptcy Prediction"
- 6,819 companies × 96 financial indicators.
- Target: `Bankrupt?` (0 = Not Bankrupt, 1 = Bankrupt).

## Methodology
- Preprocessed data (imputed missing values, removed constant columns).
- Train/test split with stratification.
- Built two models:
  - Logistic Regression
  - Random Forest (with balanced class weights)

## Key Results
- Random Forest achieved higher ROC-AUC than Logistic Regression.
- Feature importances highlighted the top 20 financial predictors.
- Predictions and evaluation metrics are saved in `outputs/`.

## Visualizations
- `charts/class_distribution.png` – distribution of bankrupt vs non-bankrupt companies  
- `charts/confusion_matrix_rf.png` – confusion matrix of Random Forest  
- `charts/roc_curve_models.png` – ROC curve comparison  
- `charts/feature_importances_top20.png` – top 20 important features  

## How to Run
1. Clone this repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/company-bankruptcy-prediction.git
   cd company-bankruptcy-prediction
