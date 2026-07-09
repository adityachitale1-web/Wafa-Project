"""
Fine-tune DistilmBERT on the Wafa messages - runs on FREE Google Colab (T4).

How to use (teammate owning M1b):
  1. Open colab.research.google.com -> New notebook -> Runtime -> T4 GPU.
  2. Upload this file and data/messages.csv (left sidebar -> Files).
  3. Run:  !pip -q install transformers datasets accelerate
           !python finetune_distilmbert_colab.py --task churn_signal
           !python finetune_distilmbert_colab.py --task issue_type
  4. Download the two wafa_finetuned_* folders (zip them first:
     !zip -r finetuned.zip wafa_finetuned_*) and place them under
     project_wafa/models/ on your laptop. Inference runs fine on CPU.

Why DistilmBERT: multilingual (104 languages incl. Arabic and Tagalog),
half the size of mBERT, fine-tunes on 252 messages in ~2 minutes on a T4.
The romanized-Hindi messages are the known weak spot - check the
per-language report this script prints and put the finding in the report.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

MODEL_NAME = "distilbert-base-multilingual-cased"
SEED = 42

LABELS = {
    "issue_type": ["Account_Closure", "Remittance_Transfer", "Loan_Mortgage",
                   "Fees_Charges", "Card_Services", "App_Technical",
                   "General_Query"],
    "churn_signal": ["High", "Medium", "Low"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=list(LABELS), default="churn_signal")
    ap.add_argument("--csv", default="messages.csv")
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import (AutoModelForSequenceClassification,
                              AutoTokenizer, Trainer, TrainingArguments)

    labels = LABELS[args.task]
    df = pd.read_csv(args.csv)
    df["label"] = df[args.task].map(labels.index)
    strat = df["issue_type"] + "_" + df["churn_signal"]
    tr, te = train_test_split(df, test_size=0.2, random_state=SEED,
                              stratify=strat)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)

    def encode(batch):
        return tok(batch["text"], truncation=True, max_length=64,
                   padding="max_length")

    ds_tr = Dataset.from_pandas(tr[["text", "label"]]).map(encode, batched=True)
    ds_te = Dataset.from_pandas(te[["text", "label"]]).map(encode, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)})

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=f"ft_{args.task}_ckpt", seed=SEED,
            num_train_epochs=args.epochs, per_device_train_batch_size=16,
            learning_rate=3e-5, eval_strategy="epoch", logging_steps=10,
            report_to=[]),
        train_dataset=ds_tr, eval_dataset=ds_te)
    trainer.train()

    # ------------------------------------------------ evaluate per language
    from sklearn.metrics import classification_report

    logits = trainer.predict(ds_te).predictions
    pred = np.argmax(logits, axis=1)
    y = te["label"].to_numpy()
    report = {
        "task": args.task,
        "overall": classification_report(y, pred, target_names=labels,
                                         output_dict=True, zero_division=0),
        "per_language_accuracy": {},
        "note": ("Random split shares generator templates between train/test;"
                 " compare against models/lstm_eval.json under the same split"
                 " seed for a fair bake-off."),
    }
    langs = te["language"].to_numpy()
    for lang in ["en", "ar", "hi", "tl"]:
        m = langs == lang
        if m.sum():
            report["per_language_accuracy"][lang] = {
                "n": int(m.sum()), "acc": float((pred[m] == y[m]).mean())}

    out_dir = f"wafa_finetuned_{args.task}"
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    with open(f"{out_dir}/eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report["per_language_accuracy"], indent=1))
    print(f"overall acc: {(pred == y).mean():.3f}  -> saved to {out_dir}/")


if __name__ == "__main__":
    main()
