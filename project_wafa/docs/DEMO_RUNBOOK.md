# Project Wafa — Demo Runbook (12–15 minutes)

The presentation IS the running dashboard. Slides exist only for the opening
framing and the evidence section. This runbook is the script.

## Pre-demo checklist (day before + morning of)

- [ ] `pip install -r requirements.txt` verified on the DEMO laptop (not just your own)
- [ ] Models trained and present: `models/lstm_listen.keras`, `models/churn_model.joblib`
- [ ] Qwen cached locally: run `python training/smoke_test.py` once with `WAFA_USE_LLM=1`
      (first run downloads ~1 GB — never let this happen live)
- [ ] Decide LLM mode for the live demo:
      - `WAFA_USE_LLM=1` — real Qwen drafts, 30–60 s per draft on CPU. Fill the wait by
        narrating the decision rules panel (rehearse this).
      - `WAFA_USE_LLM=0` — instant template drafts. Safe mode if the machine is slow;
        show one pre-generated Qwen draft from the audit log instead.
- [ ] `python app.py` launches clean; all three tabs load
- [ ] If you expect the panel to type NOVEL messages in Q&A, launch with
      `WAFA_CLASSIFIER=finetuned` — the bake-off shows it generalizes far better
      than the LSTM on unseen phrasings (54% vs 32% issue accuracy). For the
      scripted demo messages, the default `lstm` is perfect and loads instantly.
- [ ] Wifi OFF test: the whole demo must work offline once models are cached
- [ ] Backup recording made (see bottom) and copied to a second device/phone
- [ ] `logs/audit_log.jsonl` seeded with a few approved entries so the audit tab isn't empty

## The flow

### 1. Business framing (2 min, slides)
- The scenario: closures +40% YoY, four languages flooding in, CX team of 18.
- The architecture diagram (docs/architecture_flow.png) — name the module owners.
- One sentence on the spine: learned models for language, transparent rules for money,
  human approval before anything reaches a customer.

### 2. Live demo (~7 min, dashboard)

**Message A — high churn signal, Arabic (the platform hears what English-only monitoring misses)**
Paste: `سأغادر الإمارات الشهر القادم. لماذا خصمتم هذه الرسوم`
Customer: `FB1000`
- Point at: language Arabic 99%, issue Fees/Charges, churn signal High, leaving confirmed YES.
- Point at: fused risk CRITICAL — behaviour score ~100% (balance draining, salary stopped,
  transfer spike) AGREES with the text.
- Rule fired: dignified goodbye (leaving confirmed beats everything). Show the draft, approve it.

**Message B — the invisible leaver (why text + behaviour beats either alone)**
Paste: `I have already resigned and am moving back home. how do I close my salary account.`
Customer: `FB1027`
- Point at: behaviour score ~1% — this customer looks HEALTHY on paper. No behaviour-only
  system would ever flag them.
- The text says they're leaving; fusion overrides to Critical; dignified goodbye pathway.
- Say the line: "75% of high-intent messages in our data came from customers whose
  behaviour still looked calm. Text is the early-warning system."
- Show the dignified-goodbye draft: help them close smoothly, thank you, door stays open.
  NO retention pressure — open the prompt accordion and show the hard rules.

**Message C — routine query, Tagalog (the platform knows when to do nothing)**
Paste: `ano ang oras ng branch. salamat po.`
Customer: `FB1001`
- Low risk, answer-the-query action, no offer. The platform doesn't harass happy customers.
- Approve; flip to the Audit log tab — every decision, every verdict, append-only.

Optional if time: edit a draft before approving (human-in-the-loop is real), or show the
Segment risk overview tab (AED 5.5M CLV at risk, concentrated in 28 Premium/Private customers).

### 3. Evidence (~3 min, slides or JSON on screen)
- LSTM confusion matrices + per-language table (`models/lstm_eval.json`).
- The honest part (this earns marks — do not skip):
  - Random-split scores are ~perfect because the data is template-generated → we built an
    unseen-template test set (`data/messages_unseen.csv`) and report those numbers instead
    (`models/bakeoff_results.json`: LSTM vs fine-tuned DistilmBERT vs zero-shot Qwen).
  - Romanized Hindi is the weakest language — say what we did about it.
  - The LLM wrote "Dear [Customer's Name]" in testing → we added the placeholder guardrail;
    show the guardrail_flags mechanism.
- Fairness: nationality excluded from the churn model; audit shows predicted risk tracks
  actual outcomes per region (`models/churn_eval.json`).

### 4. Business close (~2 min)
- AED 5.46M of a 19.9M book at risk; 85% of that value in 28 customers → 18 people CAN
  cover this, if triage is automatic.
- Capacity math: ~10 min/message manual multilingual triage vs ~1 min draft review.
- The dignified goodbye is brand strategy: the expat who leaves respected returns, and refers.
- Next: own-language outreach at quality (Qwen 1.5B), CRM integration, live message streams.

## Break-glass procedures

| Failure | Action |
|---|---|
| App won't start | `WAFA_USE_LLM=0 python app.py` (templates only, instant) |
| LLM hangs mid-demo | Narrate rules panel; if >90 s, say "template fallback is exactly for this", restart analyze with LLM off |
| Laptop dies / projector chaos | Play the backup recording, keep narrating over it |

## Backup recording (record AFTER rehearsal, ~2 min)

Windows: Win+G (Game Bar) → record the browser window while running Message A and B
end-to-end (use `WAFA_USE_LLM=0` so the recording is snappy). Save as `demo_backup.mp4`
in this folder AND on a phone. A recording after a genuine failure costs little;
having nothing costs the demo marks.
