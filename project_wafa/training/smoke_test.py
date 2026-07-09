"""Thin-thread smoke test: the three demo scenarios from the brief,
end to end (M1 -> M2 -> M3 -> audit). Run with WAFA_USE_LLM=0 for the
template path, or =1 to exercise Qwen."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wafa import audit, m1_listen, m2_understand, m3_act

SCENARIOS = [
    ("HIGH churn, Arabic",
     "سأغادر الإمارات الشهر القادم. لا يصلني رمز التحقق.", "FB1000"),
    ("Genuine leaver, English",
     "I have already resigned and am moving back home. how do I close my salary account.",
     "FB1027"),
    ("Routine query, Tagalog",
     "ano ang oras ng branch. salamat po.", "FB1001"),
]

for name, text, cid in SCENARIOS:
    print("=" * 72)
    print(f"SCENARIO: {name}  ({cid})")
    print(f"  text: {text}")
    signals = m1_listen.listen(text, cid)
    print(f"  M1 -> lang={signals['language']} issue={signals['issue_type']} "
          f"({signals['issue_confidence']:.2f}) signal={signals['churn_signal']} "
          f"({signals['churn_signal_confidence']:.2f}) "
          f"leaving={signals['leaving_confirmed']} clf={signals['classifier']}")
    print(f"        entities={json.dumps(signals['entities'], ensure_ascii=False)}")
    risk = m2_understand.assess_risk(signals)
    print(f"  M2 -> fused={risk['fused_risk']} band={risk['risk_band']} "
          f"(behaviour={risk['behaviour_score']}, text={risk['text_score']})")
    for r in risk["reasons"]:
        print(f"        why: {r}")
    plan = m3_act.decide_and_act(signals, risk)
    print(f"  M3 -> action={plan['action']} offer=AED {plan['offer_value_aed']:.0f} "
          f"source={plan['draft_source']} flags={plan['guardrail_flags']}")
    for r in plan["rationale"]:
        print(f"        rule: {r}")
    print(f"  DRAFT: {plan['draft_message'][:200]}")
    entry = audit.log_decision(signals, risk, plan, "approved (smoke test)",
                               plan["draft_message"])
    print(f"  AUDIT: logged at {entry['timestamp']}")

print("=" * 72)
print("smoke test complete - thin thread is live")
