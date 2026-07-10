"""
Project Wafa - retention dashboard (Streamlit port of project_wafa/app.py).

Runs the same pipeline as the original Gradio app:
  message in -> understanding -> fused risk + why -> action -> draft ->
  human approve / edit / reject -> audit log.

Deployed on Streamlit Community Cloud with the entrypoint at the repo root;
the original Gradio app is untouched in project_wafa/.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "project_wafa"
sys.path.insert(0, str(ROOT))

# The Qwen drafting LLM is too heavy for Streamlit Cloud's free tier;
# m3_act falls back to its bilingual template drafts.
os.environ.setdefault("WAFA_USE_LLM", "0")

import pandas as pd
import streamlit as st

from wafa import audit, m1_listen, m2_understand, m3_act

st.set_page_config(page_title="Project Wafa - Falcon Bank UAE",
                   page_icon="🤝", layout="wide")

BAND_COLOR = {"Critical": "#a32d2d", "High": "#ba7517",
              "Medium": "#185fa5", "Low": "#0f6e56"}
LANG_NAMES = {"en": "English", "ar": "Arabic", "hi": "Hindi (romanized)",
              "tl": "Tagalog"}


@st.cache_data
def load_data():
    messages = pd.read_csv(ROOT / "data" / "messages.csv")
    customers = pd.read_csv(ROOT / "data" / "customers.csv")
    return messages, customers


MESSAGES, CUSTOMERS = load_data()

SAMPLE_LABELS = [
    f"{r.message_id} | {r.language} | {r.text[:70]}"
    for r in MESSAGES.itertuples()
]


def _apply_sample():
    label = st.session_state.get("sample")
    if not label:
        return
    mid = label.split(" | ")[0]
    row = MESSAGES[MESSAGES.message_id == mid].iloc[0]
    st.session_state["text_in"] = row.text
    st.session_state["cust_in"] = row.customer_id
    st.session_state["msg_id"] = mid


def _audit_df() -> pd.DataFrame:
    rows = audit.read_log()
    if not rows:
        return pd.DataFrame(columns=["timestamp", "customer_id", "risk_band",
                                     "action", "human_verdict"])
    df = pd.DataFrame(rows)
    return df[["timestamp", "customer_id", "language", "issue_type",
               "risk_band", "action", "offer_value_aed", "draft_source",
               "human_verdict"]].iloc[::-1]


@st.cache_data
def _segment_view() -> pd.DataFrame:
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


def _verdict(verdict: str, final_text: str):
    if "plan" not in st.session_state:
        st.error("Analyze a message first.")
        return
    audit.log_decision(st.session_state["signals"], st.session_state["risk"],
                       st.session_state["plan"], verdict, final_text)
    st.session_state["verdict_msg"] = (
        f"Recorded: **{verdict}** - logged to the audit trail. "
        f"No message reaches a customer without this step.")


st.title("Project Wafa")
st.markdown("**Customer retention intelligence - Falcon Bank UAE.** "
            "Wafa means loyalty, and it runs both ways.")

tab_triage, tab_segments, tab_audit = st.tabs(
    ["Triage a message", "Segment risk overview", "Audit log"])

with tab_triage:
    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.selectbox("Pick a sample message", SAMPLE_LABELS, index=None,
                     key="sample", on_change=_apply_sample)
        text = st.text_area("Customer message", key="text_in", height=140,
                            placeholder="...or paste a live message here")
        cust = st.text_input("Customer ID", value="FB1000", key="cust_in")
        go = st.button("Analyze", type="primary")

    if go:
        text = (text or "").strip()
        if not text:
            st.error("Paste or select a customer message first.")
        else:
            with st.spinner("Running the pipeline..."):
                signals = m1_listen.listen(
                    text, (cust or "UNKNOWN").strip(),
                    st.session_state.get("msg_id", "LIVE"))
                risk = m2_understand.assess_risk(signals)
                plan = m3_act.decide_and_act(signals, risk)
            st.session_state.update(signals=signals, risk=risk, plan=plan)
            st.session_state["draft"] = plan["draft_message"]
            st.session_state.pop("verdict_msg", None)

    if "plan" in st.session_state:
        signals = st.session_state["signals"]
        risk = st.session_state["risk"]
        plan = st.session_state["plan"]

        with col_out:
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("What the platform understood")
                ents = signals["entities"]
                ent_lines = "".join(
                    f"\n- {k.replace('_', ' ')}: {', '.join(map(str, v))}"
                    for k, v in ents.items()) or "\n- none detected"
                st.markdown(
                    f"**Language:** {LANG_NAMES[signals['language']]} "
                    f"({signals['language_confidence']:.0%})\n\n"
                    f"**Issue:** {signals['issue_type'].replace('_', ' / ')} "
                    f"({signals['issue_confidence']:.0%}, model: {signals['classifier']})\n\n"
                    f"**Churn signal in text:** {signals['churn_signal']} "
                    f"({signals['churn_signal_confidence']:.0%})\n\n"
                    f"**Leaving confirmed:** "
                    f"{'YES' if signals['leaving_confirmed'] else 'no'}\n\n"
                    f"**Entities:**{ent_lines}")
            with c2:
                st.subheader("Fused risk")
                color = BAND_COLOR[risk["risk_band"]]
                reasons = "".join(f"\n- {r}" for r in risk["reasons"])
                triage = ("\n\n**Low model confidence - routed to human triage.**"
                          if risk["needs_human_triage"] else "")
                st.markdown(
                    f"## <span style='color:{color}'>{risk['risk_band']} - "
                    f"{risk['fused_risk']:.0%}</span>\n\n"
                    f"behaviour score {risk['behaviour_score']:.0%} x 0.55  +  "
                    f"text signal {risk['text_score']:.0%} x 0.45\n\n"
                    f"**Segment:** {risk['segment']} | **CLV:** AED "
                    f"{risk['clv_estimate_aed']:,.0f}\n\n"
                    f"**Why:**{reasons}{triage}",
                    unsafe_allow_html=True)

            st.subheader("Decision")
            rationale = "".join(f"\n{i+1}. {r}"
                                for i, r in enumerate(plan["rationale"]))
            offer = (f"AED {plan['offer_value_aed']:.0f} "
                     f"({'within' if plan['offer_within_budget'] else 'EXCEEDS'} "
                     f"CLV budget cap)"
                     if plan["offer_value_aed"] > 0 else "none")
            flags = ("\n\n**Guardrail flags:** " + "; ".join(plan["guardrail_flags"])
                     if plan["guardrail_flags"] else "")
            st.markdown(
                f"### {plan['action_label']}\n\n"
                f"**Offer:** {offer}\n\n"
                f"**Rules fired:**{rationale}{flags}\n\n"
                f"*Draft written by: {plan['draft_source']} in "
                f"{LANG_NAMES[plan['draft_language']]}*")

        st.divider()
        draft = st.text_area("Drafted outreach (edit before approving)",
                             key="draft", height=160)
        with st.expander("Show the LLM prompt (ethics: tone constraints)"):
            st.text(plan["prompt_used"] or "(template draft - no LLM prompt used)")

        b1, b2, b3, _ = st.columns([1, 1, 1, 3])
        if b1.button("Approve & send", type="primary"):
            _verdict("approved", draft)
        if b2.button("Approve with edits"):
            _verdict("edited", draft)
        if b3.button("Reject"):
            _verdict("rejected", "")
        if st.session_state.get("verdict_msg"):
            st.success(st.session_state["verdict_msg"])

with tab_segments:
    st.markdown("Churn-model risk aggregated per segment "
                "(who is at risk, and how much value is exposed).")
    st.dataframe(_segment_view(), use_container_width=True)

with tab_audit:
    st.markdown("Every decision and every human verdict, append-only.")
    st.dataframe(_audit_df(), use_container_width=True)
    st.button("Refresh")  # any click reruns the script and reloads the log
