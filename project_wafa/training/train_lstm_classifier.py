"""
Train the M1 text classifier: a bidirectional LSTM with two heads
(issue_type, churn_signal) over a shared embedding - built from scratch,
exactly the architecture family covered in the course.

Run:  python training/train_lstm_classifier.py
Outputs:
  models/lstm_listen.keras       - trained model (int-sequence input)
  models/lstm_listen_meta.json   - vocabulary + label maps for inference
  models/lstm_eval.json          - held-out metrics incl. per-language accuracy

Honest-evaluation note: the provided messages are template-generated, so a
random split shares templates between train and test and scores near-perfect.
We report the random-split metrics AND flag this in the eval file; the
unseen-template test (extending the generator) is the follow-up experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS = 2000
SEQ_LEN = 32
SEED = 42

ISSUES = ["Account_Closure", "Remittance_Transfer", "Loan_Mortgage",
          "Fees_Charges", "Card_Services", "App_Technical", "General_Query"]
SIGNALS = ["High", "Medium", "Low"]


def main() -> None:
    import keras
    from keras import layers
    from sklearn.model_selection import train_test_split

    keras.utils.set_random_seed(SEED)

    df = pd.read_csv(ROOT / "data" / "messages.csv")
    y_issue = df["issue_type"].map(ISSUES.index).to_numpy()
    y_signal = df["churn_signal"].map(SIGNALS.index).to_numpy()
    strat = df["issue_type"] + "_" + df["churn_signal"]

    idx_train, idx_test = train_test_split(
        np.arange(len(df)), test_size=0.2, random_state=SEED, stratify=strat)

    vectorizer = layers.TextVectorization(
        max_tokens=MAX_TOKENS, output_sequence_length=SEQ_LEN)
    vectorizer.adapt(df["text"].iloc[idx_train].to_numpy())
    X = vectorizer(df["text"].to_numpy()).numpy()

    inp = keras.Input(shape=(SEQ_LEN,), dtype="int32")
    x = layers.Embedding(MAX_TOKENS, 64, mask_zero=True)(inp)
    x = layers.Bidirectional(layers.LSTM(64))(x)
    x = layers.Dropout(0.3)(x)
    out_issue = layers.Dense(len(ISSUES), activation="softmax", name="issue")(x)
    out_signal = layers.Dense(len(SIGNALS), activation="softmax", name="signal")(x)
    model = keras.Model(inp, [out_issue, out_signal])
    model.compile(
        optimizer="adam",
        loss={"issue": "sparse_categorical_crossentropy",
              "signal": "sparse_categorical_crossentropy"},
        metrics={"issue": ["accuracy"], "signal": ["accuracy"]},
    )

    model.fit(
        X[idx_train], {"issue": y_issue[idx_train], "signal": y_signal[idx_train]},
        validation_split=0.15, epochs=60, batch_size=16, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True)],
    )

    # ------------------------------------------------------------- evaluate
    from sklearn.metrics import classification_report, confusion_matrix

    p_issue, p_signal = model.predict(X[idx_test], verbose=0)
    pred_issue = p_issue.argmax(axis=1)
    pred_signal = p_signal.argmax(axis=1)

    report = {
        "note": ("Random split shares generator templates between train/test; "
                 "treat these numbers as an upper bound. Unseen-template eval "
                 "requires extending the generator (see README)."),
        "issue_report": classification_report(
            y_issue[idx_test], pred_issue, target_names=ISSUES,
            output_dict=True, zero_division=0),
        "signal_report": classification_report(
            y_signal[idx_test], pred_signal, target_names=SIGNALS,
            output_dict=True, zero_division=0),
        "issue_confusion": confusion_matrix(y_issue[idx_test], pred_issue).tolist(),
        "signal_confusion": confusion_matrix(y_signal[idx_test], pred_signal).tolist(),
        "per_language_accuracy": {},
    }
    # multilingual fairness: accuracy per language on the test set
    langs_test = df["language"].iloc[idx_test].to_numpy()
    for lang in ["en", "ar", "hi", "tl"]:
        mask = langs_test == lang
        if mask.sum() == 0:
            continue
        report["per_language_accuracy"][lang] = {
            "n": int(mask.sum()),
            "issue_acc": float((pred_issue[mask] == y_issue[idx_test][mask]).mean()),
            "signal_acc": float((pred_signal[mask] == y_signal[idx_test][mask]).mean()),
        }

    # ---------------------------------------------------------------- save
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    model.save(models_dir / "lstm_listen.keras")
    vocab = [str(v) for v in vectorizer.get_vocabulary()]
    (models_dir / "lstm_listen_meta.json").write_text(json.dumps({
        "max_tokens": MAX_TOKENS, "seq_len": SEQ_LEN, "vocab": vocab,
        "issues": ISSUES, "signals": SIGNALS,
    }, ensure_ascii=False), encoding="utf-8")
    (models_dir / "lstm_eval.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\nissue acc:  {(pred_issue == y_issue[idx_test]).mean():.3f}")
    print(f"signal acc: {(pred_signal == y_signal[idx_test]).mean():.3f}")
    print("per-language:", json.dumps(report["per_language_accuracy"], indent=1))
    print("saved -> models/lstm_listen.keras")


if __name__ == "__main__":
    main()
