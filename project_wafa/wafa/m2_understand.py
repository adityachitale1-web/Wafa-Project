"""
M2 - UNDERSTAND: fuse what the customer SAYS with what they DO.

  behaviour_score : churn probability from the trained tabular model
  text_score      : churn signal from M1 mapped to 0..1
  fused_risk      : 0.55 * behaviour + 0.45 * text   (transparent, tunable)

Overrides (readable on purpose - they are part of the decision audit):
  * leaving_confirmed floors fused_risk at 0.85 - a customer telling us they
    are leaving outranks any model.
  * low classifier confidence (< 0.5) sets needs_human_triage instead of
    silently trusting a shaky prediction.

Reasons combine the logistic-regression drivers for THIS customer with the
text signal, so the dashboard can answer "why is this person at risk?".

Contract in : contracts.ListenSignals + a customers.csv row
Contract out: contracts.RiskAssessment
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .contracts import ListenSignals, RiskAssessment

ROOT = Path(__file__).resolve().parent.parent
W_BEHAVIOUR, W_TEXT = 0.55, 0.45
TEXT_SCORE = {"High": 0.9, "Medium": 0.55, "Low": 0.15}
CONFIDENCE_FLOOR = 0.5

NUMERIC = ["tenure_months", "products_held", "avg_balance_aed",
           "balance_trend_3m", "remittance_count_3m", "complaints_6m",
           "branch_visits_trend", "clv_estimate_aed"]
BOOLEAN = ["salary_credit_active", "intl_transfer_spike"]
CATEG = ["segment"]

# behaviour features -> plain-English reason, shown when the feature pushes
# risk up for this specific customer
_REASON_RULES = [
    ("balance_trend_3m", lambda v: v < -0.15,
     lambda v: f"balance draining ({v:+.0%} over 3 months)"),
    ("salary_credit_active", lambda v: v == 0,
     lambda v: "salary credits have stopped"),
    ("intl_transfer_spike", lambda v: v == 1,
     lambda v: "spike in international transfers"),
    ("complaints_6m", lambda v: v >= 3,
     lambda v: f"{int(v)} complaints in 6 months"),
    ("branch_visits_trend", lambda v: v < -0.4,
     lambda v: "branch engagement falling sharply"),
    ("tenure_months", lambda v: v <= 12,
     lambda v: f"new relationship ({int(v)} months tenure)"),
    ("remittance_count_3m", lambda v: v >= 8,
     lambda v: f"{int(v)} remittances in 3 months"),
]

_customers: pd.DataFrame | None = None
_model = None


def _load() -> tuple[pd.DataFrame, object | None]:
    global _customers, _model
    if _customers is None:
        _customers = pd.read_csv(ROOT / "data" / "customers.csv")
        _customers[BOOLEAN] = _customers[BOOLEAN].astype(int)
        _customers = _customers.set_index("customer_id", drop=False)
    if _model is None:
        path = ROOT / "models" / "churn_model.joblib"
        if path.exists():
            _model = joblib.load(path)
    return _customers, _model


def get_customer(customer_id: str) -> dict | None:
    customers, _ = _load()
    if customer_id in customers.index:
        return customers.loc[customer_id].to_dict()
    return None


def _band(risk: float) -> str:
    if risk >= 0.75:
        return "Critical"
    if risk >= 0.55:
        return "High"
    if risk >= 0.35:
        return "Medium"
    return "Low"


def assess_risk(signals: ListenSignals) -> RiskAssessment:
    customers, model = _load()
    cid = signals["customer_id"]
    row = customers.loc[cid] if cid in customers.index else None
    profile_found = row is not None

    reasons: list[str] = []
    if profile_found and model is not None:
        X = pd.DataFrame([row[NUMERIC + BOOLEAN + CATEG]])
        behaviour = float(model.predict_proba(X)[0, 1])
        for feat, cond, phrase in _REASON_RULES:
            v = row[feat]
            if cond(v):
                reasons.append(phrase(v))
    else:
        behaviour = 0.5  # unknown customer: neutral prior, text carries the call
        reasons.append("no profile on file - assessment based on message only")

    text_score = TEXT_SCORE[signals["churn_signal"]]
    fused = W_BEHAVIOUR * behaviour + W_TEXT * text_score

    if signals["leaving_confirmed"]:
        fused = max(fused, 0.85)
        reasons.insert(0, "customer states they are leaving the UAE")
    elif signals["churn_signal"] == "High":
        reasons.insert(0, "message expresses high churn intent")

    needs_triage = (signals["issue_confidence"] < CONFIDENCE_FLOOR
                    or signals["churn_signal_confidence"] < CONFIDENCE_FLOOR)

    return {
        "customer_id": cid,
        "behaviour_score": round(behaviour, 3),
        "text_score": text_score,
        "fused_risk": round(min(fused, 1.0), 3),
        "risk_band": _band(fused),
        "reasons": reasons[:5],
        "needs_human_triage": needs_triage,
        "segment": str(row["segment"]) if profile_found else "Unknown",
        "clv_estimate_aed": float(row["clv_estimate_aed"]) if profile_found else 0.0,
        "profile_found": profile_found,
    }
