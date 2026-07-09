const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, BorderStyle, WidthType, ShadingType, HeadingLevel } = require("docx");

const ROOT = path.join(__dirname, "..");
const CONTENT_W = 9360;
const lstmEval = JSON.parse(fs.readFileSync(path.join(ROOT, "models", "lstm_eval.json"), "utf-8"));
const perLang = lstmEval.per_language_accuracy;

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, w, head = false) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 50, bottom: 50, left: 100, right: 100 },
    shading: head ? { fill: "E4EFE9", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold: head, size: 17 })] })],
  });
}

const langNames = { en: "English", ar: "Arabic", hi: "Hindi (romanized)", tl: "Tagalog" };
const widths = [2960, 1600, 2400, 2400];
const fairnessRows = Object.entries(perLang).map(([lang, r]) =>
  new TableRow({ children: [
    cell(langNames[lang], widths[0]),
    cell(String(r.n), widths[1]),
    cell((r.issue_acc * 100).toFixed(0) + "%", widths[2]),
    cell((r.signal_acc * 100).toFixed(0) + "%", widths[3]),
  ] }));

function h2(text) { return new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(text)] }); }
function p(runs, after = 100) {
  return new Paragraph({ children: typeof runs === "string" ? [new TextRun({ text: runs, size: 19 })] : runs, spacing: { after } });
}
function run(t, o = {}) { return new TextRun({ text: t, size: 19, ...o }); }

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 19 } } },
    paragraphStyles: [
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 1 } },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 },
      margin: { top: 1000, right: 1440, bottom: 1000, left: 1440 } } },
    children: [
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
        children: [new TextRun({ text: "Project Wafa — Ethics Statement", bold: true, size: 30 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 160 },
        children: [new TextRun({ text: "MAIB AI 115 · Final Group Project · July 2026 — Dhote · Patel · Soundekar · Malankar · Chitale · Dalvi · Darshit · Devarkonda · Baid", size: 16, color: "555555" })] }),

      p([run("This platform automates decisions about people under stress, in a context involving displacement. Our design principle is the course's spine: "),
         run("learned models for language, transparent rules wherever money or trust is decided, and a human before anything reaches a customer.", { bold: true }),
         run(" The four required safeguards below are implemented in code, not policy documents — each can be shown running in the demo.")], 140),

      h2("1. The dignified goodbye"),
      p([run("Rule 1 of the decision table short-circuits everything else: any customer whose message confirms they are leaving the UAE is routed to a leaving-well pathway — smooth closure, transfer help, a genuine thank-you — and the retention-offer branches are "),
         run("structurally unreachable", { bold: true }),
         run(" for them (wafa/m3_act.py). The goodbye guardrail additionally rejects any draft containing retention pressure (“stay with us”, “reconsider”, “special offer”). Demonstrated live with a confirmed leaver in the demo.")]),

      h2("2. No dark patterns"),
      p([run("The outreach LLM writes under an explicit tone-locked prompt, quoted verbatim from the code and viewable in the dashboard: ")]),
      p([run("“NO false urgency, NO pressure, NO ‘limited time’ language. Mention ONLY the offer given below. Never invent offers, amounts, rates, or conditions. Be respectful of a customer who may be under stress; plain language, no marketing clichés.”", { italics: true })], 80),
      p([run("An independent guardrail layer re-checks every draft for urgency phrases and invented monetary amounts. Offers are honest by construction: the amount shown to the customer is exactly the amount the rule approved, and it is capped at 2% of the customer's lifetime value.")]),

      h2("3. Multilingual fairness"),
      p([run("The pipeline is multilingual-native — no translate-to-English step whose errors would silently degrade non-English service. We measured per-language quality of the trained classifier (held-out set):")], 80),
      new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
        rows: [new TableRow({ children: [cell("Language", widths[0], true), cell("n", widths[1], true),
               cell("Issue accuracy", widths[2], true), cell("Churn-signal accuracy", widths[3], true)] }),
               ...fairnessRows] }),
      p([run("Reported honestly: romanized Hindi is our weakest language — the fine-tuned DistilmBERT shows the same gap, and the held-out sample is tiny (n=4). Mitigations: we extended the data generator with new Hindi phrasings for an unseen-template evaluation, rule-based language ID was chosen specifically because off-the-shelf detectors misclassify romanized Hindi, and low-confidence predictions route to human triage instead of being trusted.")], 80),
      p([run("Outreach fairness, measured: ", { bold: true }),
         run("we tested whether the drafting LLM actually writes the customer's language. Qwen2.5-0.5B produced English in every non-English run (0 of 6), before and after a prompt-hardening iteration (evidence: models/outreach_language_check*.json). Own-language service is therefore guaranteed structurally, not hopefully: a language-check guardrail detects the mismatch and serves a curated template in the customer's own language — all 8 outreach actions are templated in all four languages, and every quality-check run ended with the customer answered in their language.")], 100),

      h2("4. Human in the loop, hallucination-guarded"),
      p([run("No LLM-drafted message can reach a customer without a human verdict — the dashboard's approve / edit / reject step writes to an append-only audit log (logs/audit_log.jsonl) recording every automated decision and every human override. "),
         run("Observed hallucination: ", { bold: true }),
         run("in testing, Qwen2.5-0.5B produced unfilled placeholders (“Dear [Customer's Name]”) in 8 of 8 quality-check drafts — and kept doing so after we added an explicit no-placeholder rule to the prompt. One goodbye draft also contained retention pressure, which the goodbye guardrail caught. Our conclusion, stated plainly: at 0.5B scale, prompt engineering does not fix these failure modes — reliability comes from the architecture. Guardrails detect placeholders, urgency language, invented amounts, retention pressure and language mismatches; any violation is flagged to the reviewer and the draft falls back to a curated template in the customer's language. The platform runs fully with the LLM disabled.")]),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "Wafa_Ethics_Statement.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out);
});
