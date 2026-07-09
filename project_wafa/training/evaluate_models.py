"""
Model bake-off: trained LSTM vs fine-tuned DistilmBERT vs keyword heuristic
vs (optionally) zero-shot Qwen2.5 - on BOTH evaluation sets:

  held_out : the 20% random split of messages.csv (same seed/stratify as
             training) - shares generator templates with training, so treat
             as an upper bound
  unseen   : data/messages_unseen.csv - brand-new templates the models never
             saw; the honest number for the report

Run:  python training/evaluate_models.py            (fast: no LLM)
      python training/evaluate_models.py --with-llm (adds zero-shot Qwen; slow on CPU)
Saves models/bakeoff_results.json and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

ISSUES = ["Account_Closure", "Remittance_Transfer", "Loan_Mortgage",
          "Fees_Charges", "Card_Services", "App_Technical", "General_Query"]
SIGNALS = ["High", "Medium", "Low"]
SEED = 42


# ------------------------------------------------------------------ models
def lstm_predict(texts: list[str]) -> tuple[list[str], list[str]]:
    from wafa.m1_listen import _get_lstm
    model = _get_lstm()
    if model is None:
        raise RuntimeError("LSTM not trained - run train_lstm_classifier.py")
    x = model.vectorizer(np.array(texts))
    p_issue, p_signal = model.model.predict(x, verbose=0)
    return ([ISSUES[i] for i in p_issue.argmax(1)],
            [SIGNALS[i] for i in p_signal.argmax(1)])


def finetuned_predict(texts: list[str]) -> tuple[list[str], list[str]] | None:
    d_issue = ROOT / "models" / "wafa_finetuned_issue_type"
    d_signal = ROOT / "models" / "wafa_finetuned_churn_signal"
    if not (d_issue.exists() and d_signal.exists()):
        return None
    from transformers import pipeline
    out = []
    for d in (d_issue, d_signal):
        pipe = pipeline("text-classification", model=str(d), device=-1)
        preds = pipe(texts, batch_size=16, truncation=True, max_length=64)
        out.append([p["label"] for p in preds])
    return out[0], out[1]


def heuristic_predict(texts: list[str]) -> tuple[list[str], list[str]]:
    from wafa.m1_listen import _heuristic_classify
    issues, signals = [], []
    for t in texts:
        i, _, s, _ = _heuristic_classify(t)
        issues.append(i)
        signals.append(s)
    return issues, signals


def llm_predict(texts: list[str]) -> tuple[list[str], list[str]]:
    """Zero-shot Qwen2.5-0.5B-Instruct - the 'just prompt an LLM' baseline."""
    from transformers import pipeline
    pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")

    def classify(text: str, labels: list[str], what: str) -> str:
        msgs = [
            {"role": "system",
             "content": f"You are a precise classifier for bank customer "
                        f"messages in English, Arabic, Hindi or Tagalog. "
                        f"Classify the {what}. Reply with EXACTLY one label "
                        f"from this list and nothing else: {', '.join(labels)}"},
            {"role": "user", "content": text},
        ]
        out = pipe(msgs, max_new_tokens=12, do_sample=False)
        reply = out[0]["generated_text"][-1]["content"]
        for lab in labels:
            if lab.lower() in reply.lower():
                return lab
        return labels[-1]  # unparseable -> least-harmful default

    issues = [classify(t, ISSUES, "banking issue type") for t in texts]
    signals = [classify(t, SIGNALS,
                        "churn risk the customer expresses (High = clearly "
                        "leaving, Medium = uncertain or frustrated, Low = "
                        "routine)") for t in texts]
    return issues, signals


# ---------------------------------------------------------------- evaluation
def score(df: pd.DataFrame, pred_issue: list[str], pred_signal: list[str]) -> dict:
    res = {
        "n": len(df),
        "issue_acc": float((df.issue_type.to_numpy() == np.array(pred_issue)).mean()),
        "signal_acc": float((df.churn_signal.to_numpy() == np.array(pred_signal)).mean()),
        "per_language": {},
    }
    for lang in ["en", "ar", "hi", "tl"]:
        m = (df.language == lang).to_numpy()
        if m.sum():
            res["per_language"][lang] = {
                "n": int(m.sum()),
                "issue_acc": float((df.issue_type.to_numpy()[m] == np.array(pred_issue)[m]).mean()),
                "signal_acc": float((df.churn_signal.to_numpy()[m] == np.array(pred_signal)[m]).mean()),
            }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-llm", action="store_true",
                    help="include zero-shot Qwen (slow on CPU)")
    args = ap.parse_args()

    from sklearn.model_selection import train_test_split

    df = pd.read_csv(ROOT / "data" / "messages.csv")
    strat = df["issue_type"] + "_" + df["churn_signal"]
    _, idx_test = train_test_split(np.arange(len(df)), test_size=0.2,
                                   random_state=SEED, stratify=strat)
    held_out = df.iloc[idx_test].reset_index(drop=True)
    unseen = pd.read_csv(ROOT / "data" / "messages_unseen.csv")

    models: dict = {"lstm": lstm_predict, "heuristic": heuristic_predict}
    if ((ROOT / "models" / "wafa_finetuned_issue_type").exists()
            and (ROOT / "models" / "wafa_finetuned_churn_signal").exists()):
        models["finetuned_distilmbert"] = finetuned_predict
    if args.with_llm:
        models["zeroshot_qwen2.5_0.5b"] = llm_predict

    results: dict = {"note": ("held_out shares generator templates with "
                              "training (upper bound); unseen uses brand-new "
                              "templates (honest number)")}
    for name, fn in models.items():
        results[name] = {}
        for ds_name, ds in (("held_out", held_out), ("unseen", unseen)):
            print(f"evaluating {name} on {ds_name} ({len(ds)} messages)...")
            pi, ps = fn(ds["text"].tolist())
            results[name][ds_name] = score(ds, pi, ps)

    out = ROOT / "models" / "bakeoff_results.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"\n{'model':<24} {'set':<9} {'issue':>6} {'signal':>7}")
    for name in models:
        for ds_name in ("held_out", "unseen"):
            r = results[name][ds_name]
            print(f"{name:<24} {ds_name:<9} {r['issue_acc']:>6.2%} {r['signal_acc']:>7.2%}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
