const fs = require("fs");
const path = require("path");
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, LevelFormat, BorderStyle, WidthType, ShadingType,
        HeadingLevel } = require("docx");

const ROOT = path.join(__dirname, "..");
const CONTENT_W = 9360;

let bakeoff = null;
try {
  bakeoff = JSON.parse(fs.readFileSync(path.join(ROOT, "models", "bakeoff_results.json"), "utf-8"));
} catch (e) { /* table rendered as pending if bake-off not yet run */ }

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, w, head = false) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 55, bottom: 55, left: 100, right: 100 },
    shading: head ? { fill: "E6EEF7", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold: head, size: 18 })] })],
  });
}
function trow(cells, widths, head = false) {
  return new TableRow({ children: cells.map((c, i) => cell(c, widths[i], head)) });
}
function table(header, rows, widths) {
  return new Table({ width: { size: CONTENT_W, type: WidthType.DXA }, columnWidths: widths,
    rows: [trow(header, widths, true), ...rows.map(r => trow(r, widths))] });
}
function h1(t) { return new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)] }); }
function p(runs, after = 110) {
  return new Paragraph({ children: typeof runs === "string" ? [new TextRun({ text: runs, size: 20 })] : runs, spacing: { after } });
}
function run(t, o = {}) { return new TextRun({ text: t, size: 20, ...o }); }
function bullet(runs) {
  return new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 60 },
    children: typeof runs === "string" ? [new TextRun({ text: runs, size: 20 })] : runs });
}

const pct = x => (x * 100).toFixed(0) + "%";
let bakeoffRows = [["(run training/evaluate_models.py to populate)", "", "", "", ""]];
if (bakeoff) {
  bakeoffRows = Object.entries(bakeoff)
    .filter(([k]) => k !== "note")
    .map(([name, r]) => [
      name.replace(/_/g, " "),
      pct(r.held_out.issue_acc), pct(r.held_out.signal_acc),
      pct(r.unseen.issue_acc), pct(r.unseen.signal_acc),
    ]);
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial" },
        paragraph: { spacing: { before: 240, after: 140 }, outlineLevel: 0 } },
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
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 50 },
        children: [new TextRun({ text: "Retention Strategy for Falcon Bank UAE", bold: true, size: 34 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 40 },
        children: [new TextRun({ text: "Findings and recommendations from Project Wafa", italics: true, size: 22 })] }),
      new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
        children: [new TextRun({ text: "Prepared for the Chief Customer Officer · MAIB AI 115 Final Group Project · July 2026 — Dhote · Patel · Soundekar · Malankar · Chitale · Dalvi · Darshit · Devarkonda · Baid", size: 16, color: "555555" })] }),

      h1("Executive summary"),
      bullet([run("AED 5.46M of the AED 19.9M customer-lifetime-value book — 27% — sits with customers the churn model flags as at risk.", { bold: true })]),
      bullet([run("That value is concentrated: 85% of at-risk CLV belongs to just 28 Premium and Private customers. "),
              run("Eighteen people can cover 28 phone calls.", { bold: true })]),
      bullet([run("Behaviour data alone is a lagging indicator: 75% of the messages expressing clear leaving intent came from customers whose behaviour still looked calm. "),
              run("The inbox is the early-warning system — if we can read it in four languages.", { bold: true })]),
      bullet([run("Project Wafa triages every message in seconds — language, issue, churn intent, recommended action, drafted reply — with a human approving every send and every decision audit-logged.")]),
      bullet([run("Customers confirmed to be leaving get a dignified goodbye, never retention pressure. In a shrinking market, the expatriate who leaves respected is the one who returns — and refers.")]),

      h1("1. What the data says"),
      p([run("Behaviour: ", { bold: true }),
         run("the churn model's strongest drivers are exactly the CCO's instincts, now quantified: a draining 3-month balance trend, stopped salary credits, and a spike in international transfers. Historical churn in the sample book is 29% (69 of 240 customers).")]),
      p([run("Voice of the customer: ", { bold: true }),
         run("45% of the 252 sampled messages are not in English. A third carry a High churn signal — and those leavers write about everything: fees, cards, OTP failures and loan worries in equal measure, not just account closures. "),
         run("Every service queue is a retention queue.", { bold: true })]),
      p([run("Why fusion matters: ", { bold: true }),
         run("of 84 high-intent messages, 63 (75%) came from customers whose behaviour-based risk score was under 35% — invisible to any system watching balances alone. The reverse also holds: quiet customers whose salaries stop never write in. Only the fused view sees both populations.")]),

      h1("2. Segment risk and the play for each"),
      table(["Segment", "Customers", "Avg CLV (AED)", "At risk", "CLV at risk (AED)", "The play"], [
        ["Mass", "149", "24,059", "38", "829,742", "Automated: fee waivers ≤ 2% of CLV, priority service fixes, drafted in the customer's language"],
        ["Premium", "70", "92,403", "23", "2,302,983", "Relationship-manager call within one working day; platform pre-briefs the RM with the why"],
        ["Private", "21", "467,892", "5", "2,330,873", "Named senior ownership immediately — five conversations protect AED 2.3M"],
      ], [1150, 1150, 1450, 950, 1750, 2910]),
      p([run("Offer economics are enforced in code: no monetary offer may exceed 2% of the customer's lifetime value; above that, the rule downgrades to a call. We never spend more retaining a customer than the relationship is worth.")], 60),

      h1("3. What the platform changes operationally"),
      p([run("Manual triage of a multilingual inbox — read, translate, judge intent, decide, draft — costs roughly 10 minutes per message; 252 messages is a full agent-week. With Wafa, triage is instant and the human role shrinks to reviewing a pre-drafted action (~1 minute): the same inbox takes about four hours. "),
         run("For an 18-person team, that converts roughly one FTE of translation-and-triage into proactive retention calls to the 28 customers who matter most.", { bold: true })]),
      p([run("Every automated decision is explainable (the rules that fired are listed line by line), every draft is approved by a person, and both are recorded in an append-only audit log — the operational posture a bank's risk function will actually sign off on.")]),

      h1("4. The evidence, honestly"),
      p([run("Two classifiers were trained by the team (a from-scratch LSTM and a fine-tuned DistilmBERT) and compared with a zero-shot prompted LLM. Because the project data is template-generated, random-split scores are near-perfect for any competent model; we therefore also built an unseen-template test set with entirely new phrasings. Those are the numbers we stand behind:")], 80),
      table(["Model", "Held-out: issue", "Held-out: signal", "Unseen: issue", "Unseen: signal"],
        bakeoffRows, [3160, 1550, 1550, 1550, 1550]),
      p([run("Reading the table honestly: ", { bold: true }),
         run("the held-out column flatters every model; the unseen column is the truth. The from-scratch LSTM largely memorized templates (100% collapsing to ~32%), while the fine-tuned multilingual transformer generalizes best — its pretrained representations transfer to phrasings it never saw (against chance baselines of 14% for issue and 33% for signal). The zero-shot prompted LLM loses to every trained model on both sets: at this scale, prompting is not a substitute for training. None of these numbers justify autonomy, which is why every decision passes a human — and the path to better numbers is known: train on a broader template pool, exactly what the extended generator enables.")], 80),
      p([run("Known limitations, stated plainly: the data is synthetic; message-to-customer linkage is simulated; churn drivers were planted by the generator (hence near-perfect propensity scores — real-world AUC will be materially lower); romanized Hindi is the weakest language for all models and is flagged for human triage at low confidence. The fairness audit confirms predicted risk does not proxy nationality — nationality is excluded from the model and used only to audit outcomes.")], 60),

      h1("5. Recommendations"),
      bullet("Deploy Wafa in the triage line now, in shadow mode for two weeks: agents see its output next to their manual process; measure agreement and time saved."),
      bullet("Stand up the segment plays above; give the five at-risk Private customers named senior owners this week."),
      bullet("Start collecting real outcome labels (saved / left after outreach) from day one — the propensity model's honest accuracy on real data depends on it."),
      bullet("Upgrade outreach to the customer's own language at quality: Qwen2.5-1.5B on one GPU workstation, with the same guardrails and human approval."),
      bullet("Track a new KPI for the dignified-goodbye pathway: returned-customer and referral rate of well-departed expatriates over 24 months. Loyalty runs both ways — that is the bet, and it should be measured."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, "Wafa_Business_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("written", out);
});
