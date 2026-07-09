# Project Wafa — Customer Retention Intelligence Platform

Falcon Bank UAE · MAIB AI 115 · Final Group Project

Wafa (وفاء) means loyalty — and it runs both ways. The platform listens to
customer messages in four languages, fuses what customers *say* with what
they *do*, decides a retention action through transparent rules, drafts the
outreach with an open LLM, and puts a human in front of every send.

## Quick start

```
pip install -r requirements.txt
python training/train_lstm_classifier.py    # trains the M1 LSTM  (~1 min, CPU)
python training/train_churn_model.py        # trains the M2 churn model (seconds)
python app.py                               # launches the Gradio dashboard
```

Set `WAFA_USE_LLM=0` to run without Qwen (curated templates draft instead —
this is also the demo-insurance path). First LLM run downloads
`Qwen/Qwen2.5-0.5B-Instruct` (~1 GB) and caches it locally.

Smoke test (three demo scenarios end to end):
`python training/smoke_test.py`

Full evaluation pipeline:

```
python training/make_unseen_testset.py       # unseen-template test set (honest eval)
python training/finetune_distilmbert_colab.py --task churn_signal --csv data/messages.csv
python training/finetune_distilmbert_colab.py --task issue_type  --csv data/messages.csv
python training/evaluate_models.py --with-llm  # bake-off -> models/bakeoff_results.json
```

The fine-tune runs on free Colab T4 (~2 min/task) or locally on CPU (~7 min/task);
move the `wafa_finetuned_*` folders into `models/`. Serve the fine-tuned pair in
the app with `WAFA_CLASSIFIER=finetuned` (default stays `lstm`).

## Deliverables (docs/)

| Deliverable | File |
|---|---|
| Architecture Design Document | `docs/Wafa_Architecture_Design_Document.docx` (+ `architecture_flow.png`, regenerate via `docs/make_flow_diagram.py`) |
| Business report | `docs/Wafa_Business_Report.docx` (rebuild with live numbers: `node docs/build_business_report.js`) |
| Ethics statement | `docs/Wafa_Ethics_Statement.docx` |
| Demo runbook + insurance checklist | `docs/DEMO_RUNBOOK.md` |

## Architecture

```
customer message ─┐                       customer profile ─┐
                  ▼                                         ▼
   M1 LISTEN (wafa/m1_listen.py)            M2a churn model (trained, LogReg;
   language ID · LSTM classifiers            nationality EXCLUDED, audit-only)
   (trained) · entities · leaving check                    │
                  │ ListenSignals                          │ P(churn)
                  └────────────► M2 UNDERSTAND ◄───────────┘
                                 (wafa/m2_understand.py)
                                 fused_risk = 0.55·behaviour + 0.45·text
                                 overrides: confirmed leaver, low confidence
                                      │ RiskAssessment
                                      ▼
                          M3 ACT (wafa/m3_act.py)
                          numbered decision rules → offer economics cap
                          → Qwen2.5-0.5B drafts (tone-constrained prompt)
                          → guardrails (no urgency / no invented amounts)
                                      │ ActionPlan
                                      ▼
                          M4 DASHBOARD (app.py, Gradio)
                          understanding · risk+why · action · draft
                          human approve / edit / reject → audit log
```

Interface contracts live in [wafa/contracts.py](wafa/contracts.py) — every
module is testable against the mock factories there, so nobody is blocked by
anybody (brief §4).

## Modules and ownership

| Module | Files | Owner |
|---|---|---|
| M1 Listen — language ID, LSTM classifiers, entities | `wafa/m1_listen.py`, `training/train_lstm_classifier.py` | Rakshanda Dhote, Krishn Patel |
| M1b Fine-tuned transformer + model bake-off | `training/finetune_distilmbert_colab.py`, `training/evaluate_models.py` | Atharva Soundekar |
| M2 Understand — churn model, fusion, fairness audit | `wafa/m2_understand.py`, `training/train_churn_model.py` | Sarth Malankar, Aditya Chitale |
| M3 Act — decision rules, LLM drafting, guardrails | `wafa/m3_act.py`, `training/check_outreach_quality.py` | Advait Dalvi, Darshit |
| M4 Dashboard + audit log | `app.py`, `wafa/audit.py` | Anurag Devarkonda |
| Data & evaluation — generator extension, unseen-template tests | `data/generate_wafa_data.py`, `training/make_unseen_testset.py` | Karan Baid |

## Design decisions (defended in the Architecture Design Document)

- **Multilingual-native, no translation step.** Hindi arrives romanized
  ("mera card decline ho gaya") — off-the-shelf hi→en translators expect
  Devanagari and standard language detectors misclassify it. The LSTM learns
  all four languages directly from the data; language ID is a transparent
  script/keyword heuristic.
- **`nationality_region` is never a model feature.** The data is generated
  with churn independent of region; a model using it learns bias. It is used
  only to audit score fairness (see `models/churn_eval.json`).
- **Logistic regression is the primary churn model** because its coefficients
  become the per-customer "why" on the dashboard; XGBoost is the benchmarked
  alternative.
- **Transparent rule table decides actions** — every fired rule is logged.
  Confirmed leavers route to the dignified goodbye; no retention pressure.
- **Offer economics:** monetary offers are capped at 2% of CLV, else the rule
  downgrades to a relationship-manager call.
- **Guardrailed drafting:** Qwen2.5-0.5B-Instruct writes under an explicit
  tone-constrained prompt (shown in the dashboard); drafts failing checks
  (urgency language, invented amounts, retention pressure in goodbyes) fall
  back to curated templates. The platform runs fully without the LLM.

## Honest findings (the marks live here)

- Template leakage, proven: every model scores ~perfectly on the random
  split, then collapses on the unseen-template set (`models/bakeoff_results.json`):
  LSTM 100% → 32% (issue), fine-tuned DistilmBERT 100% → 54%, vs chance 14%.
  The from-scratch LSTM memorized templates; pretrained multilingual
  representations generalize better.
- Zero-shot Qwen2.5-0.5B loses to every trained model on BOTH sets
  (~20–36% throughout) — at this scale, prompting is not a substitute for
  training. It also rated "I am leaving the UAE next month" as Low churn
  intent in testing.
- Serving recommendation: `WAFA_CLASSIFIER=finetuned` for novel/live inputs
  (best generalization); `lstm` stays the default for the scripted demo
  (instant load, perfect on the reference data).
- LSTM per-language check: 1 of 4 romanized-Hindi test messages missed its
  churn signal — small-sample fragility in the lowest-resource language;
  exactly what the §7 fairness check is meant to surface.
- Qwen wrote "Dear [Customer's Name]" during draft testing → placeholder
  guardrail added (plus urgency/invented-amount checks).
- Own-language outreach, measured (`models/outreach_language_check*.json`):
  Qwen2.5-0.5B wrote English for every non-English customer (0/6) and
  placeholders in 8/8 drafts — before AND after prompt hardening. Conclusion:
  at 0.5B, reliability comes from architecture, not prompting. All 8 outreach
  actions are now templated in all four languages, and a language-mismatch
  guardrail guarantees customers are answered in their own language.

## Feasibility budget

| Component | Size | Runs on |
|---|---|---|
| LSTM classifier (Keras) | < 2 MB | laptop CPU, trains in ~1 min |
| Churn model (sklearn) | < 1 MB | laptop CPU, trains in seconds |
| Qwen2.5-0.5B-Instruct | ~1 GB | laptop CPU (fallback: templates) |
| DistilmBERT fine-tune | ~540 MB | free Colab GPU (train), laptop CPU (infer) |
| Gradio dashboard | — | laptop |
