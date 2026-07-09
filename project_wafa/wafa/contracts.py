"""
Project Wafa - Interface contracts between modules.

Every arrow in the architecture diagram carries exactly one of these
dictionaries. Modules are built and tested against these contracts, so
teammates can work in parallel: if your module consumes a contract, test it
against the mock factories at the bottom of this file, never against a
teammate's unfinished code.

Pipeline:  raw message + customer_id
             -> M1 listen()      -> ListenSignals
             -> M2 assess_risk() -> RiskAssessment   (consumes ListenSignals + customer row)
             -> M3 decide_act()  -> ActionPlan       (consumes RiskAssessment)
             -> Dashboard renders everything, human approves -> audit log
"""
from typing import TypedDict, Optional


# ---------------------------------------------------------------- M1: LISTEN
class Entities(TypedDict, total=False):
    amounts_aed: list[float]      # monetary amounts mentioned
    dates: list[str]              # date-ish phrases ("next month", "March")
    destinations: list[str]       # countries/regions money or people move to
    products: list[str]           # bank products mentioned (card, loan, ...)


class ListenSignals(TypedDict):
    message_id: str               # "M0001" or "LIVE" for pasted text
    customer_id: str              # "FB1042"
    text: str                     # the raw message
    language: str                 # "en" | "ar" | "hi" | "tl"
    language_confidence: float    # 0..1
    issue_type: str               # one of the 7 issue classes
    issue_confidence: float       # 0..1 (softmax max prob)
    churn_signal: str             # "High" | "Medium" | "Low"
    churn_signal_confidence: float
    leaving_confirmed: bool       # explicit relocation statement detected
    entities: Entities
    classifier: str               # which model produced this ("lstm" | "finetuned" | "heuristic")


# ------------------------------------------------------------ M2: UNDERSTAND
class RiskAssessment(TypedDict):
    customer_id: str
    behaviour_score: float        # churn probability from the tabular model, 0..1
    text_score: float             # churn signal mapped to 0..1
    fused_risk: float             # the combined score, 0..1
    risk_band: str                # "Critical" | "High" | "Medium" | "Low"
    reasons: list[str]            # human-readable drivers, most important first
    needs_human_triage: bool      # True when classifier confidence is low
    segment: str                  # Mass | Premium | Private
    clv_estimate_aed: float       # customer lifetime value, caps offer spend
    profile_found: bool           # False -> scores based on text only


# ------------------------------------------------------------------- M3: ACT
class ActionPlan(TypedDict):
    customer_id: str
    action: str                   # action code, e.g. "fee_waiver", "dignified_goodbye"
    action_label: str             # human-readable action name
    rationale: list[str]          # which rules fired, line by line
    offer_value_aed: float        # 0 when no monetary offer
    offer_within_budget: bool     # offer_value <= cap derived from CLV
    draft_message: str            # the outreach text for human review
    draft_language: str           # language the draft is written in
    draft_source: str             # "llm" | "template"
    guardrail_flags: list[str]    # non-empty when the LLM draft violated a check
    prompt_used: Optional[str]    # the exact LLM prompt (ethics: shown on demand)


# ------------------------------------------------------------- mock factories
def mock_listen_signals(**overrides) -> ListenSignals:
    base: ListenSignals = {
        "message_id": "MOCK1", "customer_id": "FB1000",
        "text": "I am leaving the UAE next month. I want to close my account.",
        "language": "en", "language_confidence": 0.99,
        "issue_type": "Account_Closure", "issue_confidence": 0.94,
        "churn_signal": "High", "churn_signal_confidence": 0.91,
        "leaving_confirmed": True,
        "entities": {"dates": ["next month"], "products": ["account"]},
        "classifier": "heuristic",
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def mock_risk_assessment(**overrides) -> RiskAssessment:
    base: RiskAssessment = {
        "customer_id": "FB1000", "behaviour_score": 0.82, "text_score": 0.9,
        "fused_risk": 0.86, "risk_band": "Critical",
        "reasons": ["balance draining over last 3 months", "salary credits stopped"],
        "needs_human_triage": False, "segment": "Premium",
        "clv_estimate_aed": 99627.0, "profile_found": True,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def mock_action_plan(**overrides) -> ActionPlan:
    base: ActionPlan = {
        "customer_id": "FB1000", "action": "dignified_goodbye",
        "action_label": "Dignified goodbye - help them leave well",
        "rationale": ["Rule 1: customer confirmed they are leaving the UAE"],
        "offer_value_aed": 0.0, "offer_within_budget": True,
        "draft_message": "Thank you for banking with us...",
        "draft_language": "en", "draft_source": "template",
        "guardrail_flags": [], "prompt_used": None,
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base
