"""
Own-language outreach quality check - evidence for the ethics statement.

For each of the four languages, runs two realistic scenarios (a fee-waiver
retention offer and a dignified goodbye) through the REAL drafting path
(Qwen2.5-0.5B + guardrails + language check + template fallback) and records:

  * whether the LLM produced the customer's language at all
  * which guardrails tripped
  * what was ultimately served (llm draft vs curated template) and in
    what language

The point is fairness measurement, not a pass/fail: if the small LLM cannot
write Tagalog, the customer must still be answered in Tagalog - by the
template. Saves models/outreach_language_check.json.

Run:  python training/check_outreach_quality.py   (needs WAFA_USE_LLM=1; ~5 min CPU)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from wafa.contracts import mock_listen_signals, mock_risk_assessment
from wafa.m1_listen import detect_language
from wafa.m3_act import draft_outreach

SCENARIOS = {
    "fee_waiver": {
        "en": "why was I charged this maintenance fee. I am very frustrated with the service lately.",
        "ar": "لماذا خصمتم هذه الرسوم. أنا محبط جدا من الخدمة.",
        "hi": "yeh charges kyun lage hain. main service se pareshan hoon.",
        "tl": "bakit may bayad na ganito. dismayado ako sa serbisyo.",
    },
    "dignified_goodbye": {
        "en": "I am leaving the UAE next month. I want to close my account.",
        "ar": "سأغادر الإمارات الشهر القادم. أريد إغلاق حسابي.",
        "hi": "main agle mahine UAE chhod raha hoon. mujhe apna account band karna hai.",
        "tl": "aalis na ako ng UAE sa susunod na buwan. gusto ko nang isara ang account ko.",
    },
}
OFFERS = {"fee_waiver": 200.0, "dignified_goodbye": 0.0}

results: dict = {}
for action, texts in SCENARIOS.items():
    for lang, text in texts.items():
        signals = mock_listen_signals(
            text=text, language=lang,
            issue_type="Fees_Charges" if action == "fee_waiver" else "Account_Closure",
            churn_signal="Medium" if action == "fee_waiver" else "High",
            leaving_confirmed=(action == "dignified_goodbye"))
        risk = mock_risk_assessment(risk_band="High" if action == "fee_waiver" else "Critical")
        rationale = [f"quality-check scenario: {action}"]
        t0 = time.time()
        draft, draft_lang, source, flags, _ = draft_outreach(
            signals, risk, action, rationale, OFFERS[action])
        detected, _ = detect_language(draft)
        results[f"{action}/{lang}"] = {
            "llm_wrote_customer_language": not any("language mismatch" in f for f in flags),
            "guardrail_flags": flags,
            "served_by": source,
            "served_language": draft_lang,
            "served_language_verified": detected == lang,
            "seconds": round(time.time() - t0, 1),
            "draft_preview": draft[:140],
        }
        print(f"{action}/{lang}: llm_ok={results[f'{action}/{lang}']['llm_wrote_customer_language']} "
              f"served_by={source} in {draft_lang} "
              f"({results[f'{action}/{lang}']['seconds']}s) flags={flags}")

summary = {
    "llm_language_success": {
        lang: sum(1 for k, r in results.items()
                  if k.endswith("/" + lang) and r["llm_wrote_customer_language"])
        for lang in ["en", "ar", "hi", "tl"]},
    "customer_always_answered_in_own_language": all(
        r["served_language_verified"] for r in results.values()),
}
out = ROOT / "models" / "outreach_language_check.json"
out.write_text(json.dumps({"summary": summary, "runs": results}, indent=2,
                          ensure_ascii=False), encoding="utf-8")
print("\nsummary:", json.dumps(summary, indent=1))
print(f"saved -> {out}")
