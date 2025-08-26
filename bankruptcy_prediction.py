
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

def main():
    # Paths
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(project_root, "data.csv")
    outputs_dir = os.path.join(project_root, "outputs")
    charts_dir = os.path.join(project_root, "charts")
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)

    # Load data
    df = pd.read_csv(data_path)

    # Detect target
    possible_targets = ["Bankrupt?", "bankrupt?", "bankrupt", "Bankruptcy", "bankruptcy", "Class", "class", "Target", "target", "label", "Label", "Y"]
    target_col = None
    for c in df.columns:
        if c.strip() in possible_targets:
            target_col = c
            break
    if target_col is None:
        for c in df.columns:
            unique_vals = pd.unique(df[c].dropna())
            if len(unique_vals) == 2:
                lowered = {str(v).strip().lower() for v in unique_vals}
                if lowered.issubset({"0","1","true","false","yes","no"}):
                    target_col = c
                    break
    if target_col is None:
        target_col = df.columns[-1]

    y_raw = df[target_col]

    def to_binary(series):
        m = series.astype(str).str.strip().str.lower()
        if set(m.unique()).issubset({"0","1"}):
            return m.astype(int).to_numpy()
        if set(m.unique()).issubset({"true","false"}):
            return (m == "true").astype(int).to_numpy()
        if set(m.unique()).issubset({"yes","no"}):
            return (m == "yes").astype(int).to_numpy()
        try:
            nums = pd.to_numeric(series, errors="coerce")
            unique_nums = set(pd.unique(nums.dropna()))
            if unique_nums.issubset({0,1}):
                return nums.fillna(0).astype(int).to_numpy()
        except Exception:
            pass
        vc = series.value_counts(dropna=False)
        if len(vc) == 2:
            minority = vc.index[-1]
            return (series == minority).astype(int).to_numpy()
        try:
            nums = pd.to_numeric(series, errors="coerce")
            median_val = np.nanmedian(nums)
            return (nums > median_val).astype(int)
        except Exception:
            return np.zeros(len(series), dtype=int)

    y = to_binary(y_raw)
    X = df.drop(columns=[target_col])

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocessor_for_lr = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols)
        ],
        remainder="drop"
    )

    numeric_transformer_rf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])
    categorical_transformer_rf = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    preprocessor_for_rf = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer_rf, num_cols),
            ("cat", categorical_transformer_rf, cat_cols)
        ],
        remainder="drop"
    )

    stratify_y = y if len(np.unique(y)) == 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=stratify_y
    )

    lr_clf = Pipeline(steps=[
        ("preprocess", preprocessor_for_lr),
        ("clf", LogisticRegression(max_iter=1000))
    ])
    rf_clf = Pipeline(steps=[
        ("preprocess", preprocessor_for_rf),
        ("clf", RandomForestClassifier(
            n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
        ))
    ])

    lr_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)

    def safe_predict_proba(model, X):
        try:
            proba = model.predict_proba(X)
            if proba.shape[1] == 2:
                return proba[:, 1]
            return proba[:, 0]
        except Exception:
            return None

    y_pred_lr = lr_clf.predict(X_test)
    y_proba_lr = safe_predict_proba(lr_clf, X_test)

    y_pred_rf = rf_clf.predict(X_test)
    y_proba_rf = safe_predict_proba(rf_clf, X_test)

    def compute_metrics(y_true, y_pred, y_proba):
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        try:
            if y_proba is not None and len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
        return metrics

    metrics_lr = compute_metrics(y_test, y_pred_lr, y_proba_lr)
    metrics_rf = compute_metrics(y_test, y_pred_rf, y_proba_rf)

    report_lr = classification_report(y_test, y_pred_lr, zero_division=0)
    report_rf = classification_report(y_test, y_pred_rf, zero_division=0)

    cm = confusion_matrix(y_test, y_pred_rf)

    # Plots — each on its own figure, no styles/colors set
    # 1) Class distribution
    plt.figure()
    class_counts = pd.Series(y).value_counts().sort_index()
    class_counts.plot(kind="bar")
    plt.title("Class Distribution (0 = Not Bankrupt, 1 = Bankrupt)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "class_distribution.png"))
    plt.close()

    # 2) Confusion matrix (RF)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Random Forest)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Not Bankrupt", "Bankrupt"])
    plt.yticks(tick_marks, ["Not Bankrupt", "Bankrupt"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     verticalalignment="center")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "confusion_matrix_rf.png"))
    plt.close()

    # 3) Feature importances (RF)
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    # Impute only, to align with raw column names
    if len(num_cols) > 0:
        imputer_num = SimpleImputer(strategy="median")
        X_train_num_imp = pd.DataFrame(imputer_num.fit_transform(X_train[num_cols]), columns=num_cols, index=X_train.index)
    else:
        X_train_num_imp = pd.DataFrame(index=X_train.index)

    if len(cat_cols) > 0:
        imputer_cat = SimpleImputer(strategy="most_frequent")
        X_train_cat_imp = pd.DataFrame(imputer_cat.fit_transform(X_train[cat_cols]), columns=cat_cols, index=X_train.index)
    else:
        X_train_cat_imp = pd.DataFrame(index=X_train.index)

    X_train_imp = pd.concat([X_train_num_imp, X_train_cat_imp], axis=1)
    rf_raw = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced")
    rf_raw.fit(X_train_imp, y_train)
    importances = rf_raw.feature_importances_

    imp_df = pd.DataFrame({"feature": X_train_imp.columns, "importance": importances}).sort_values("importance", ascending=False)
    top_k = imp_df.head(15)

    plt.figure()
    plt.barh(top_k["feature"][::-1], top_k["importance"][::-1])
    plt.title("Top 15 Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "feature_importances_top15.png"))
    plt.close()

    # Save predictions
    test_index = getattr(X_test, "index", pd.RangeIndex(start=0, stop=len(X_test)))
    preds_df = pd.DataFrame({
        "index": test_index,
        "y_true": y_test,
        "y_pred_rf": y_pred_rf,
    })
    if y_proba_rf is not None:
        preds_df["y_proba_rf"] = y_proba_rf
    preds_df.to_csv(os.path.join(outputs_dir, "predictions_rf.csv"), index=False)

    # Save metrics and reports
    metrics_summary = {
        "target_column_detected": target_col,
        "logistic_regression": metrics_lr,
        "random_forest": metrics_rf,
        "n_samples": int(df.shape[0]),
        "n_features": int(X.shape[1]),
        "class_counts": class_counts.to_dict(),
        "created_at": datetime.now().isoformat()
    }
    with open(os.path.join(outputs_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)
    with open(os.path.join(outputs_dir, "classification_report_lr.txt"), "w") as f:
        f.write(report_lr)
    with open(os.path.join(outputs_dir, "classification_report_rf.txt"), "w") as f:
        f.write(report_rf)

if __name__ == "__main__":
    main()
