const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
        AlignmentType, LevelFormat, BorderStyle, WidthType, ShadingType,
        HeadingLevel } = require("docx");

const DOCS = __dirname;
const CONTENT_W = 9360;

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const HEAD_FILL = "E8E4F5";

function cell(text, w, opts = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: opts.head ? { fill: HEAD_FILL, type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ children: [new TextRun({
      text, bold: !!opts.head, size: 18,
      font: opts.mono ? "Consolas" : "Arial" })] })],
  });
}

function row(cells, widths, opts = {}) {
  return new TableRow({ children: cells.map((c, i) =>
    cell(c, widths[i], { head: opts.head, mono: opts.monoFirst && i === 0 })) });
}

function table(header, rows, widths, monoFirst = false) {
  return new Table({
    width: { size: CONTENT_W, type: WidthType.DXA },
    columnWidths: widths,
    rows: [row(header, widths, { head: true }),
           ...rows.map(r => row(r, widths, { monoFirst }))],
  });
}

function h1(text) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(text)] }); }
function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function p(children, opts = {}) {
  const runs = typeof children === "string" ? [new TextRun({ text: children, size: 20 })] : children;
  return new Paragraph({ children: runs, spacing: { after: opts.after ?? 120 }, ...opts.extra });
}
function bullet(text) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 },
    spacing: { after: 60 },
    children: [new TextRun({ text, size: 20 })] });
}
function boldRun(t) { return new TextRun({ text: t, bold: true, size: 20 }); }
function run(t) { return new TextRun({ text: t, size: 20 }); }

const contractsIntro =
  "Every arrow in the flow diagram carries exactly one of the dictionaries below " +
  "(defined with mock factories in wafa/contracts.py). Each module is developed and " +
  "tested against these mocks, so no member is ever blocked by another’s unfinished code.";

const listenFields = [
  ["message_id / customer_id", "str", "message ref; customer_id joins to profile"],
  ["text", "str", "the raw message"],
  ["language", "en | ar | hi | tl", "detected language + language_confidence (0–1)"],
  ["issue_type", "7-class str", "issue class + issue_confidence (0–1)"],
  ["churn_signal", "High | Medium | Low", "text churn intent + churn_signal_confidence"],
  ["leaving_confirmed", "bool", "explicit relocation statement detected"],
  ["entities", "dict", "amounts_aed, dates, destinations, products"],
  ["classifier", "lstm | finetuned | heuristic", "which model produced the prediction"],
];

const riskFields = [
  ["behaviour_score", "float 0–1", "churn probability from the tabular model"],
  ["text_score", "float 0–1", "churn signal mapped High=0.9 / Med=0.55 / Low=0.15"],
  ["fused_risk / risk_band", "float / str", "0.55·behaviour + 0.45·text; Critical≥.75, High≥.55, Med≥.35"],
  ["reasons", "list[str]", "plain-English drivers, most important first"],
  ["needs_human_triage", "bool", "true when classifier confidence < 0.5"],
  ["segment / clv_estimate_aed", "str / float", "customer value context; CLV caps offer spend"],
  ["profile_found", "bool", "false → neutral behaviour prior, text carries the call"],
];

const actionFields = [
  ["action / action_label", "str", "action code + human-readable name"],
  ["rationale", "list[str]", "every fired rule, line by line (auditable)"],
  ["offer_value_aed", "float", "0 when no monetary offer; offer_within_budget flag"],
  ["draft_message", "str", "outreach text awaiting human review"],
  ["draft_language / draft_source", "str", "language of draft; llm or template"],
  ["guardrail_flags", "list[str]", "non-empty when the LLM draft violated a check"],
  ["prompt_used", "str | null", "the exact LLM prompt (shown in dashboard, ethics)"],
];

const modelRows = [
  ["Issue + churn-signal classifier (trained #1)", "Bidirectional LSTM, Keras, from scratch",
   "Multilingual-native incl. romanized Hindi; the exact architecture family built in the course; <2 MB, trains in ~1 min on CPU"],
  ["Issue + churn-signal classifier (trained #2)", "DistilmBERT fine-tune",
   "104 languages incl. Arabic and Tagalog; ~2 min on free Colab T4 (also CPU-feasible); bake-off vs the LSTM decides which serves the app"],
  ["Language identification", "Rule-based (script range + keyword scoring)",
   "Romanized Hindi defeats off-the-shelf detectors; rules are transparent and testable; fasttext-lid is the upgrade path"],
  ["Entity extraction", "Regex + gazetteers",
   "Transparent and reliable at 252-message scale; spaCy NER optional for English"],
  ["Churn propensity", "Logistic regression (primary), XGBoost benchmarked",
   "Coefficients give the per-customer ‘why’ shown on the dashboard; nationality_region excluded by design (fairness)"],
  ["Outreach generation", "Qwen2.5-0.5B-Instruct",
   "Brief’s own guidance: best small-model Arabic; runs on laptop CPU; tone-locked prompt + guardrails; curated-template fallback means the demo never depends on it"],
];

const budgetRows = [
  ["BiLSTM classifier", "< 2 MB", "laptop CPU", "train ~1 min; infer < 0.1 s", "keyword heuristic (built in)"],
  ["DistilmBERT fine-tune", "~540 MB", "train: free Colab T4; infer: laptop CPU", "train ~2 min (T4); infer ~0.2 s", "LSTM remains the serving model"],
  ["Churn model (sklearn)", "< 1 MB", "laptop CPU", "train in seconds", "—"],
  ["Qwen2.5-0.5B-Instruct", "~1 GB cached", "laptop CPU", "30–60 s per draft", "curated templates (WAFA_USE_LLM=0)"],
  ["Gradio dashboard", "—", "laptop, localhost", "instant", "screen-recording backup"],
];

const img = fs.readFileSync(path.join(DOCS, "architecture_flow.png"));

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 200, after: 120 }, outlineLevel: 1 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1080, right: 1440, bottom: 1080, left: 1440 } } },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "Project Wafa — Architecture Design Document", bold: true, size: 36 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
        children: [new TextRun({ text: "A Customer Retention Intelligence Platform for Falcon Bank UAE", italics: true, size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
        children: [new TextRun({ text: "MAIB AI 115 · NLP and Models · Final Group Project — Deliverable 1 · July 2026", size: 20, color: "555555" })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: [new TextRun({ text: "Rakshanda Dhote · Krishn Patel · Atharva Soundekar · Sarth Malankar · Aditya Chitale · Advait Dalvi · Darshit · Anurag Devarkonda · Karan Baid", size: 18, color: "555555" })] }),

      h1("1. Design philosophy"),
      p([run("The platform follows the course’s spine: "),
         boldRun("learned models for fuzzy language, transparent rules where money and trust are decided, and a human in the loop where stakes are high."),
         run(" Text and behaviour are fused because either alone misleads: a customer can look healthy on paper while telling us they are leaving (our smoke tests surfaced exactly this case — behaviour score 0.8%, message: “I have already resigned and am moving back home”). We chose a "),
         boldRun("multilingual-native pipeline with no translation step"),
         run(": Hindi arrives romanized (“mera card decline ho gaya”), which breaks off-the-shelf hi→en translators and language detectors; training directly on the four languages side-steps that failure mode and handles code-switched text for free.")]),

      h1("2. End-to-end flow"),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
        children: [new ImageRun({ type: "png", data: img,
          transformation: { width: 620, height: 430 },
          altText: { title: "Architecture flow", description: "Flow from message and profile to dashboard", name: "flow" } })] }),

      h1("3. Modules and ownership"),
      table(["Module", "Key files", "Owner"], [
        ["M1 Listen — language ID, LSTM classifiers, entities", "wafa/m1_listen.py; training/train_lstm_classifier.py", "Rakshanda Dhote, Krishn Patel"],
        ["M1b Fine-tuned transformer + model bake-off", "training/finetune_distilmbert_colab.py; training/evaluate_models.py", "Atharva Soundekar"],
        ["M2 Understand — churn model, fusion, fairness audit", "wafa/m2_understand.py; training/train_churn_model.py", "Sarth Malankar, Aditya Chitale"],
        ["M3 Act — decision rules, LLM drafting, guardrails", "wafa/m3_act.py; training/check_outreach_quality.py", "Advait Dalvi, Darshit"],
        ["M4 Dashboard + audit log", "app.py; wafa/audit.py", "Anurag Devarkonda"],
        ["Data & evaluation — generator extension, unseen-template tests", "data/generate_wafa_data.py; training/make_unseen_testset.py", "Karan Baid"],
      ], [3600, 3560, 2200]),
      p("Nine members across six modules: the heavy modules (M1, M2, M3) are pair-owned; every member owns at least one module, and the same owners appear on the flow diagram above.", { after: 60 }),

      h1("4. Interface contracts"),
      p(contractsIntro),
      h2("M1 → M2: ListenSignals"),
      table(["Field", "Type / values", "Meaning"], listenFields, [2600, 2600, 4160], true),
      h2("M2 → M3: RiskAssessment"),
      table(["Field", "Type / values", "Meaning"], riskFields, [2600, 2600, 4160], true),
      h2("M3 → M4: ActionPlan"),
      table(["Field", "Type / values", "Meaning"], actionFields, [2600, 2600, 4160], true),

      h1("5. Model selection and justification"),
      table(["Task", "Choice", "Why (quality · size · languages · hardware)"], modelRows, [2300, 2560, 4500]),
      p([run("Two models are trained by the team (LSTM from scratch, DistilmBERT fine-tuned), exceeding the one-trained-model minimum. The bake-off (training/evaluate_models.py) compares both against a zero-shot Qwen classifier on the same held-out split "),
         boldRun("and on an unseen-template test set"),
         run(", because the provided messages are template-generated and random-split scores are upper bounds. Results and per-language quality go in the business report, honestly, whichever model wins.")]),

      h1("6. Feasibility budget — proof it runs free"),
      table(["Component", "Size", "Runs on", "Time", "Fallback"], budgetRows, [2160, 1300, 2400, 1900, 1600]),
      p("Total disk footprint ≈ 2 GB, total cost AED 0. No paid API is required at any point; every component has a fallback that keeps the live demo running offline once models are cached.", { after: 60 }),

      h1("7. Risks and fallbacks"),
      bullet("Live-demo failure (wifi, re-download, timeout): all models pre-cached locally; WAFA_USE_LLM=0 gives an instant template-only mode; a 2-minute screen recording is the break-glass backup."),
      bullet("Template leakage: near-perfect random-split metrics are reported as upper bounds; the honest evaluation runs on generator-extended unseen phrasings."),
      bullet("LLM hallucination and language failure: guardrails catch urgency language, invented amounts, unfilled placeholders and drafts in the wrong language (all observed in testing — see models/outreach_language_check*.json) and fall back to curated templates in the customer's language; no draft reaches a customer without human approval."),
      bullet("Fairness: nationality_region is never a model feature; it is used solely to audit that predicted risk does not skew by region (audit passing; see models/churn_eval.json)."),
      bullet("Draft quality at 0.5B: acceptable but plain; if demo hardware allows, Qwen2.5-1.5B is a drop-in upgrade — the bake-off template covers this comparison."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(DOCS, "Wafa_Architecture_Design_Document.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out);
});
