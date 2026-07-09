"""
Project Wafa - retention dashboard (Gradio).

Run:  python app.py          (add WAFA_USE_LLM=0 to skip loading Qwen)

The dashboard walks one message through the whole platform:
  message in -> understanding -> fused risk + why -> action -> draft ->
  human approve / edit / reject -> audit log.
"""
from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import pandas as pd

from wafa import audit, m1_listen, m2_understand, m3_act

ROOT = Path(__file__).resolve().parent
MESSAGES = pd.read_csv(ROOT / "data" / "messages.csv")
CUSTOMERS = pd.read_csv(ROOT / "data" / "customers.csv")

BAND_COLOR = {"Critical": "#a32d2d", "High": "#ba7517",
              "Medium": "#185fa5", "Low": "#0f6e56"}
LANG_NAMES = {"en": "English", "ar": "Arabic", "hi": "Hindi (romanized)",
              "tl": "Tagalog"}

_sample_labels = [
    f"{r.message_id} | {r.language} | {r.text[:70]}"
    for r in MESSAGES.itertuples()
]
_state: dict = {}


def _pick_sample(label: str):
    mid = label.split(" | ")[0]
    row = MESSAGES[MESSAGES.message_id == mid].iloc[0]
    return row.text, row.customer_id, mid


def analyze(text: str, customer_id: str, message_id: str):
    text = (text or "").strip()
    customer_id = (customer_id or "").strip()
    if not text:
        raise gr.Error("Paste or select a customer message first.")
    signals = m1_listen.listen(text, customer_id or "UNKNOWN",
                               message_id or "LIVE")
    risk = m2_understand.assess_risk(signals)
    plan = m3_act.decide_and_act(signals, risk)
    _state.update(signals=signals, risk=risk, plan=plan)

    ents = signals["entities"]
    ent_lines = "".join(
        f"\n- {k.replace('_', ' ')}: {', '.join(map(str, v))}"
        for k, v in ents.items()) or "\n- none detected"
    understanding = (
        f"**Language:** {LANG_NAMES[signals['language']]} "
        f"({signals['language_confidence']:.0%})\n\n"
        f"**Issue:** {signals['issue_type'].replace('_', ' / ')} "
        f"({signals['issue_confidence']:.0%}, model: {signals['classifier']})\n\n"
        f"**Churn signal in text:** {signals['churn_signal']} "
        f"({signals['churn_signal_confidence']:.0%})\n\n"
        f"**Leaving confirmed:** {'YES' if signals['leaving_confirmed'] else 'no'}\n\n"
        f"**Entities:**{ent_lines}")

    color = BAND_COLOR[risk["risk_band"]]
    reasons = "".join(f"\n- {r}" for r in risk["reasons"])
    triage = ("\n\n**Low model confidence - routed to human triage.**"
              if risk["needs_human_triage"] else "")
    risk_md = (
        f"## <span style='color:{color}'>{risk['risk_band']} - "
        f"{risk['fused_risk']:.0%}</span>\n\n"
        f"behaviour score {risk['behaviour_score']:.0%} x 0.55  +  "
        f"text signal {risk['text_score']:.0%} x 0.45\n\n"
        f"**Segment:** {risk['segment']} | **CLV:** AED "
        f"{risk['clv_estimate_aed']:,.0f}\n\n"
        f"**Why:**{reasons}{triage}")

    rationale = "".join(f"\n{i+1}. {r}" for i, r in enumerate(plan["rationale"]))
    offer = (f"AED {plan['offer_value_aed']:.0f} "
             f"({'within' if plan['offer_within_budget'] else 'EXCEEDS'} CLV budget cap)"
             if plan["offer_value_aed"] > 0 else "none")
    flags = ("\n\n**Guardrail flags:** " + "; ".join(plan["guardrail_flags"])
             if plan["guardrail_flags"] else "")
    action_md = (
        f"### {plan['action_label']}\n\n"
        f"**Offer:** {offer}\n\n"
        f"**Rules fired:**{rationale}{flags}\n\n"
        f"*Draft written by: {plan['draft_source']} in "
        f"{LANG_NAMES[plan['draft_language']]}*")

    prompt_text = plan["prompt_used"] or "(template draft - no LLM prompt used)"
    return understanding, risk_md, action_md, plan["draft_message"], prompt_text


def _verdict(verdict: str, final_text: str):
    if not _state:
        raise gr.Error("Analyze a message first.")
    audit.log_decision(_state["signals"], _state["risk"], _state["plan"],
                       verdict, final_text)
    return (f"Recorded: **{verdict}** - logged to the audit trail. "
            f"No message reaches a customer without this step."), _audit_df()


def _audit_df():
    rows = audit.read_log()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "customer_id", "risk_band",
                                     "action", "human_verdict"])
    df = pd.DataFrame(rows)
    return df[["timestamp", "customer_id", "language", "issue_type",
               "risk_band", "action", "offer_value_aed", "draft_source",
               "human_verdict"]].iloc[::-1]


def _segment_view():
    df = CUSTOMERS.copy()
    try:
        import joblib
        model = joblib.load(ROOT / "models" / "churn_model.joblib")
        feats = ["tenure_months", "products_held", "avg_balance_aed",
                 "balance_trend_3m", "remittance_count_3m", "complaints_6m",
                 "branch_visits_trend", "clv_estimate_aed",
                 "salary_credit_active", "intl_transfer_spike", "segment"]
        X = df[feats].copy()
        X[["salary_credit_active", "intl_transfer_spike"]] = (
            X[["salary_credit_active", "intl_transfer_spike"]].astype(int))
        df["risk"] = model.predict_proba(X)[:, 1]
    except Exception:
        df["risk"] = float("nan")
    g = (df.groupby("segment")
           .agg(customers=("customer_id", "count"),
                mean_risk=("risk", "mean"),
                at_risk=("risk", lambda s: int((s >= 0.55).sum())),
                clv_at_risk_aed=("clv_estimate_aed",
                                 lambda s: float(s[df.loc[s.index, "risk"] >= 0.55].sum())))
           .reset_index())
    g["mean_risk"] = g["mean_risk"].round(3)
    g["clv_at_risk_aed"] = g["clv_at_risk_aed"].round(0)
    return g


with gr.Blocks(title="Project Wafa - Falcon Bank UAE") as demo:
    gr.Markdown("# Project Wafa\n**Customer retention intelligence - "
                "Falcon Bank UAE.** Wafa means loyalty, and it runs both ways.")

    with gr.Tab("Triage a message"):
        with gr.Row():
            with gr.Column(scale=1):
                sample = gr.Dropdown(_sample_labels, label="Pick a sample message",
                                     value=None)
                text_in = gr.Textbox(label="Customer message", lines=4,
                                     placeholder="...or paste a live message here")
                cust_in = gr.Textbox(label="Customer ID", value="FB1000")
                msg_id = gr.Textbox(visible=False, value="LIVE")
                go = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=2):
                with gr.Row():
                    understanding = gr.Markdown(label="What the platform understood")
                    risk_md = gr.Markdown(label="Fused risk")
                action_md = gr.Markdown(label="Decision")
        draft = gr.Textbox(label="Drafted outreach (edit before approving)",
                           lines=6, interactive=True)
        with gr.Accordion("Show the LLM prompt (ethics: tone constraints)",
                          open=False):
            prompt_view = gr.Textbox(label="Exact prompt", lines=8,
                                     interactive=False)
        with gr.Row():
            approve = gr.Button("Approve & send", variant="primary")
            edit = gr.Button("Approve with edits")
            reject = gr.Button("Reject", variant="stop")
        verdict_md = gr.Markdown()

    with gr.Tab("Segment risk overview"):
        gr.Markdown("Churn-model risk aggregated per segment "
                    "(who is at risk, and how much value is exposed).")
        seg_table = gr.Dataframe(value=_segment_view)

    with gr.Tab("Audit log"):
        gr.Markdown("Every decision and every human verdict, append-only.")
        audit_table = gr.Dataframe(value=_audit_df)
        refresh = gr.Button("Refresh")

    sample.change(_pick_sample, sample, [text_in, cust_in, msg_id])
    go.click(analyze, [text_in, cust_in, msg_id],
             [understanding, risk_md, action_md, draft, prompt_view])
    approve.click(lambda t: _verdict("approved", t), draft,
                  [verdict_md, audit_table])
    edit.click(lambda t: _verdict("edited", t), draft,
               [verdict_md, audit_table])
    reject.click(lambda t: _verdict("rejected", ""), draft,
                 [verdict_md, audit_table])
    refresh.click(_audit_df, None, audit_table)


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1",
                server_port=int(os.environ.get("WAFA_PORT", "7860")))
