"""
Train the M2 churn-propensity model on customers.csv.

Design decisions (defended in the Architecture Design Document):
  * nationality_region is EXCLUDED from features - the data was generated so
    churn is independent of region; a model leaning on it has found bias, not
    signal. Region is used only afterwards, to AUDIT the fairness of scores.
  * Primary model: logistic regression - coefficients give the per-customer
    "reasons" the dashboard shows. XGBoost is trained as a comparison and its
    metrics reported honestly.

Run:  python training/train_churn_model.py
Outputs:
  models/churn_model.joblib   - sklearn pipeline (primary, logistic regression)
  models/churn_eval.json      - metrics for both models + fairness audit
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

NUMERIC = ["tenure_months", "products_held", "avg_balance_aed",
           "balance_trend_3m", "remittance_count_3m", "complaints_6m",
           "branch_visits_trend", "clv_estimate_aed"]
BOOLEAN = ["salary_credit_active", "intl_transfer_spike"]
CATEG = ["segment"]
EXCLUDED = ["nationality_region"]  # audit-only, never a feature


def main() -> None:
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (classification_report, confusion_matrix,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    df = pd.read_csv(ROOT / "data" / "customers.csv")
    df[BOOLEAN] = df[BOOLEAN].astype(int)
    y = df["churned"].astype(int)
    X = df[NUMERIC + BOOLEAN + CATEG]

    X_tr, X_te, y_tr, y_te, region_tr, region_te = train_test_split(
        X, y, df["nationality_region"], test_size=0.25,
        random_state=SEED, stratify=y)

    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC + BOOLEAN),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEG),
    ])
    logreg = Pipeline([("pre", pre),
                       ("clf", LogisticRegression(max_iter=2000, C=1.0))])
    logreg.fit(X_tr, y_tr)
    p_lr = logreg.predict_proba(X_te)[:, 1]

    from xgboost import XGBClassifier
    xgb = Pipeline([("pre", pre), ("clf", XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.1,
        random_state=SEED, eval_metric="logloss"))])
    xgb.fit(X_tr, y_tr)
    p_xgb = xgb.predict_proba(X_te)[:, 1]

    def evaluate(name: str, p: np.ndarray) -> dict:
        pred = (p >= 0.5).astype(int)
        return {
            "auc": float(roc_auc_score(y_te, p)),
            "report": classification_report(y_te, pred, output_dict=True,
                                            zero_division=0),
            "confusion": confusion_matrix(y_te, pred).tolist(),
        }

    # fairness audit: mean predicted risk per nationality region (test set).
    # Scores should be roughly level; a skewed region means the model found
    # a proxy for nationality despite its exclusion.
    audit = {}
    for region in sorted(region_te.unique()):
        mask = (region_te == region).to_numpy()
        audit[region] = {
            "n": int(mask.sum()),
            "mean_predicted_risk": float(p_lr[mask].mean()),
            "actual_churn_rate": float(y_te[mask].mean()),
        }

    coef = logreg.named_steps["clf"].coef_[0]
    feat_names = logreg.named_steps["pre"].get_feature_names_out().tolist()

    report = {
        "primary": "logistic_regression",
        "logistic_regression": evaluate("logreg", p_lr),
        "xgboost": evaluate("xgb", p_xgb),
        "fairness_audit_by_region": audit,
        "coefficients": dict(zip(feat_names, coef.round(4).tolist())),
        "excluded_features": EXCLUDED,
    }

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(logreg, models_dir / "churn_model.joblib")
    (models_dir / "churn_eval.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print(f"logreg  AUC: {report['logistic_regression']['auc']:.3f}")
    print(f"xgboost AUC: {report['xgboost']['auc']:.3f}")
    print("fairness audit (mean predicted risk by region):")
    for r, a in audit.items():
        print(f"  {r:<15} risk={a['mean_predicted_risk']:.3f} "
              f"actual={a['actual_churn_rate']:.3f} (n={a['n']})")
    print("saved -> models/churn_model.joblib")


if __name__ == "__main__":
    main()
