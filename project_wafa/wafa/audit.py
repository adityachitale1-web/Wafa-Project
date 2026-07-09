"""Append-only audit log: every automated decision and every human verdict
is recorded to logs/audit_log.jsonl (an ethics requirement made visible)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "audit_log.jsonl"


def log_decision(signals: dict, risk: dict, plan: dict,
                 human_verdict: str, final_message: str) -> dict:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "message_id": signals["message_id"],
        "customer_id": signals["customer_id"],
        "language": signals["language"],
        "issue_type": signals["issue_type"],
        "churn_signal": signals["churn_signal"],
        "leaving_confirmed": signals["leaving_confirmed"],
        "fused_risk": risk["fused_risk"],
        "risk_band": risk["risk_band"],
        "action": plan["action"],
        "offer_value_aed": plan["offer_value_aed"],
        "draft_source": plan["draft_source"],
        "guardrail_flags": plan["guardrail_flags"],
        "human_verdict": human_verdict,   # approved | edited | rejected
        "final_message": final_message,
    }
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    with LOG_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
