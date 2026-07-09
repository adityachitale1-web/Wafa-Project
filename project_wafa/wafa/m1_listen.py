"""
M1 - LISTEN: turn a raw multilingual message into structured signals.

Components:
  * Language ID     - transparent heuristic (Arabic script + romanized Hindi /
                      Tagalog keyword scoring). Romanized Hindi defeats most
                      off-the-shelf detectors, which is why this is rule-based.
  * Classifiers     - the team-TRAINED Keras LSTM (models/lstm_listen.keras)
                      predicts issue_type and churn_signal with confidences.
                      A keyword heuristic is the always-works fallback so the
                      pipeline never breaks while models are retrained.
  * Entities        - regex + gazetteers: amounts, dates, destinations, products.
  * Leaving check   - explicit relocation statements (multilingual phrase list)
                      set leaving_confirmed, which routes to the dignified
                      goodbye downstream. Kept transparent on purpose.

Contract out: contracts.ListenSignals
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .contracts import Entities, ListenSignals

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

ISSUES = ["Account_Closure", "Remittance_Transfer", "Loan_Mortgage",
          "Fees_Charges", "Card_Services", "App_Technical", "General_Query"]
SIGNALS = ["High", "Medium", "Low"]

# ------------------------------------------------------------- language ID
_AR_RE = re.compile(r"[؀-ۿ]")
_HI_WORDS = {"mera", "mujhe", "apna", "apne", "hai", "hain", "nahi", "kya",
             "karna", "karo", "band", "zyada", "bahut", "gaya", "raha",
             "rahe", "bhejne", "paise", "wapas", "kaise", "hoga", "doon",
             "lage", "pooch", "mahine", "chhod", "hamesha", "pareshan",
             "dhanyavaad", "baar", "aa", "ho", "ke", "liye", "se", "ja"}
_TL_WORDS = {"ko", "kong", "ang", "ng", "sa", "po", "ako", "hindi", "paano",
             "gusto", "kailangan", "salamat", "bakit", "aalis", "kami",
             "lilipat", "isara", "ipadala", "pera", "nabigo", "laging",
             "dumarating", "masyadong", "mataas", "singil", "oras", "lang",
             "nang", "may", "mga", "na"}


def detect_language(text: str) -> tuple[str, float]:
    if _AR_RE.search(text):
        return "ar", 0.99
    tokens = re.findall(r"[a-z']+", text.lower())
    if not tokens:
        return "en", 0.5
    hi = sum(t in _HI_WORDS for t in tokens) / len(tokens)
    tl = sum(t in _TL_WORDS for t in tokens) / len(tokens)
    if hi > 0.12 and hi >= tl:
        return "hi", min(0.99, 0.6 + hi)
    if tl > 0.12:
        return "tl", min(0.99, 0.6 + tl)
    return "en", 0.9


# ------------------------------------------------- leaving-confirmed check
_LEAVING_PHRASES = [
    # en
    "leaving the uae", "relocating for good", "moving back home",
    "final month here", "already resigned", "we are relocating",
    # ar
    "سأغادر الإمارات", "ننتقل نهائيا",
    # hi (romanized)
    "uae chhod", "hamesha ke liye ja",
    # tl
    "aalis na ako ng uae", "lilipat na kami",
]


def leaving_confirmed(text: str) -> bool:
    low = text.lower()
    return any(p in low for p in _LEAVING_PHRASES)


# ------------------------------------------------------- entity extraction
_COUNTRIES = {"india": "India", "pakistan": "Pakistan", "philippines": "Philippines",
              "pilipinas": "Philippines", "egypt": "Egypt", "uk": "UK",
              "home country": "home country", "abroad": "abroad",
              "الخارج": "abroad", "الإمارات": "UAE", "uae": "UAE"}
_PRODUCTS = {"account": "account", "salary account": "salary account",
             "card": "card", "loan": "loan", "mortgage": "mortgage",
             "emi": "loan EMI", "app": "mobile app", "otp": "OTP",
             "statement": "statements", "remittance": "remittance",
             "transfer": "transfer", "fee": "fees", "charge": "charges",
             "حساب": "account", "بطاقتي": "card", "قرضي": "loan",
             "الرهن": "mortgage", "الرسوم": "fees", "التطبيق": "mobile app",
             "hulog": "loan EMI"}
_AMOUNT_RE = re.compile(r"(?:aed|dhs?\.?)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:aed|dirhams?)",
                        re.IGNORECASE)
_DATE_PHRASES = ["next month", "next week", "this month", "tomorrow",
                 "الشهر القادم", "agle mahine", "susunod na buwan"]


def extract_entities(text: str) -> Entities:
    low = text.lower()
    ents: Entities = {}
    amounts = []
    for m in _AMOUNT_RE.finditer(low):
        raw = (m.group(1) or m.group(2) or "").replace(",", "")
        if raw:
            amounts.append(float(raw))
    if amounts:
        ents["amounts_aed"] = amounts
    dates = [p for p in _DATE_PHRASES if p in low]
    if dates:
        ents["dates"] = dates
    dests = sorted({v for k, v in _COUNTRIES.items() if k in low})
    if dests:
        ents["destinations"] = dests
    prods = sorted({v for k, v in _PRODUCTS.items() if k in low})
    if prods:
        ents["products"] = prods
    return ents


# ------------------------------------------------------------- classifiers
_KEYWORDS = {  # heuristic fallback: transparent, never crashes
    "Account_Closure": ["close my account", "close accounts", "band karna",
                        "band hoga", "isara", "إغلاق", "أغلق"],
    "Remittance_Transfer": ["transfer", "remittance", "bhejne", "ipadala",
                            "pera", "تحويل", "حوالتي"],
    "Loan_Mortgage": ["loan", "mortgage", "emi", "hulog", "قرضي", "الرهن"],
    "Fees_Charges": ["fee", "charge", "singil", "bayad", "الرسوم", "خصمتم"],
    "Card_Services": ["card", "declined", "block", "بطاقتي"],
    "App_Technical": ["app", "otp", "crash", "log in", "statements",
                      "التطبيق", "رمز التحقق"],
}
_SIGNAL_KEYWORDS = {
    "High": _LEAVING_PHRASES,
    "Medium": ["not sure", "uncertain", "frustrated", "big decisions",
               "pareshan", "pata nahi", "dismayado", "sigurado", "محبط",
               "لست متأكدا"],
}


def _heuristic_classify(text: str) -> tuple[str, float, str, float]:
    low = text.lower()
    issue, i_conf = "General_Query", 0.4
    for cls, words in _KEYWORDS.items():
        if any(w in low for w in words):
            issue, i_conf = cls, 0.7
            break
    signal, s_conf = "Low", 0.6
    for cls, words in _SIGNAL_KEYWORDS.items():
        if any(w.lower() in low for w in words):
            signal, s_conf = cls, 0.75
            break
    return issue, i_conf, signal, s_conf


class _LstmClassifier:
    """Loads the team-trained LSTM (see training/train_lstm_classifier.py)."""

    def __init__(self) -> None:
        import keras  # imported lazily so the fallback works without TF

        meta = json.loads((MODELS_DIR / "lstm_listen_meta.json").read_text(encoding="utf-8"))
        self.model = keras.models.load_model(MODELS_DIR / "lstm_listen.keras")
        from keras import layers
        self.vectorizer = layers.TextVectorization(
            max_tokens=meta["max_tokens"],
            output_sequence_length=meta["seq_len"],
            vocabulary=meta["vocab"],
        )
        self.issues = meta["issues"]
        self.signals = meta["signals"]

    def predict(self, text: str) -> tuple[str, float, str, float]:
        import numpy as np

        x = self.vectorizer(np.array([text]))
        issue_p, signal_p = self.model.predict(x, verbose=0)
        i = int(issue_p[0].argmax())
        s = int(signal_p[0].argmax())
        return (self.issues[i], float(issue_p[0][i]),
                self.signals[s], float(signal_p[0][s]))


class _FinetunedClassifier:
    """Loads the Colab fine-tuned DistilmBERT pair (issue + signal). Serve it
    with WAFA_CLASSIFIER=finetuned once the bake-off says it earns the slot."""

    def __init__(self) -> None:
        from transformers import pipeline

        self.p_issue = pipeline(
            "text-classification",
            model=str(MODELS_DIR / "wafa_finetuned_issue_type"), device=-1)
        self.p_signal = pipeline(
            "text-classification",
            model=str(MODELS_DIR / "wafa_finetuned_churn_signal"), device=-1)

    def predict(self, text: str) -> tuple[str, float, str, float]:
        ri = self.p_issue(text, truncation=True, max_length=64)[0]
        rs = self.p_signal(text, truncation=True, max_length=64)[0]
        return ri["label"], float(ri["score"]), rs["label"], float(rs["score"])


_lstm: _LstmClassifier | None = None
_lstm_failed = False
_finetuned: _FinetunedClassifier | None = None
_finetuned_failed = False


def _get_lstm() -> _LstmClassifier | None:
    global _lstm, _lstm_failed
    if _lstm is None and not _lstm_failed:
        try:
            _lstm = _LstmClassifier()
        except Exception:
            _lstm_failed = True
    return _lstm


def _get_finetuned() -> _FinetunedClassifier | None:
    global _finetuned, _finetuned_failed
    if _finetuned is None and not _finetuned_failed:
        try:
            _finetuned = _FinetunedClassifier()
        except Exception:
            _finetuned_failed = True
    return _finetuned


# -------------------------------------------------------------- entry point
def listen(text: str, customer_id: str, message_id: str = "LIVE") -> ListenSignals:
    lang, lang_conf = detect_language(text)
    model = None
    clf = "heuristic"
    if os.environ.get("WAFA_CLASSIFIER", "lstm") == "finetuned":
        model = _get_finetuned()
        clf = "finetuned"
    if model is None:
        model = _get_lstm()
        clf = "lstm" if model is not None else "heuristic"
    if model is not None:
        issue, i_conf, signal, s_conf = model.predict(text)
    else:
        issue, i_conf, signal, s_conf = _heuristic_classify(text)
    return {
        "message_id": message_id, "customer_id": customer_id, "text": text,
        "language": lang, "language_confidence": lang_conf,
        "issue_type": issue, "issue_confidence": i_conf,
        "churn_signal": signal, "churn_signal_confidence": s_conf,
        "leaving_confirmed": leaving_confirmed(text),
        "entities": extract_entities(text),
        "classifier": clf,
    }
