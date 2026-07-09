"""
M3 - ACT: decide the retention action, then draft the outreach.

The decision layer is a numbered rule table - every rule that fires is
recorded in `rationale`, so a reviewer can re-derive any decision by hand.
This transparency is a hard course requirement wherever money and trust are
decided.

Offer economics: a monetary offer may not exceed OFFER_CAP_PCT of the
customer's CLV. If it would, the rule downgrades to a relationship-manager
call - we never spend more retaining a customer than they are worth.

Drafting: Qwen2.5-0.5B-Instruct (free, CPU-feasible) drafts the outreach,
conditioned on the decided action and constrained by an explicit tone prompt
(no urgency, no invented offers - the ethics statement quotes this prompt).
Guardrails validate the draft; on any violation, or when the LLM is not
available, a curated template takes over. The demo therefore never breaks.

Contract in : ListenSignals + RiskAssessment
Contract out: contracts.ActionPlan
"""
from __future__ import annotations

import os
import re

from .contracts import ActionPlan, ListenSignals, RiskAssessment

OFFER_CAP_PCT = 0.02          # max offer = 2% of CLV
FEE_WAIVER_AED = 200.0
REMIT_DISCOUNT_AED = 100.0

ACTION_LABELS = {
    "dignified_goodbye": "Dignified goodbye - help them leave well",
    "fee_waiver": "Waive fees + apology",
    "remittance_support": "Remittance fee reduction + transfer help",
    "loan_restructure_info": "Loan guidance + relationship-manager call",
    "rm_call": "Relationship-manager call",
    "service_fix": "Priority service fix + follow-up",
    "proactive_checkin": "Proactive check-in",
    "answer_query": "Answer the query",
    "human_triage": "Route to human triage",
}


# ------------------------------------------------------------ decision rules
def decide(signals: ListenSignals, risk: RiskAssessment) -> tuple[str, list[str], float]:
    """Returns (action, rationale, offer_value_aed). Rules read top-down;
    the first terminal rule wins. Every fired rule lands in the rationale."""
    rationale: list[str] = []
    offer = 0.0

    if risk["needs_human_triage"]:
        rationale.append("Rule 0: classifier confidence below 0.5 - a human "
                         "must read this message first")
        return "human_triage", rationale, offer

    if signals["leaving_confirmed"]:
        rationale.append("Rule 1: customer confirmed they are leaving the UAE "
                         "- dignified goodbye, never retention pressure")
        return "dignified_goodbye", rationale, offer

    issue = signals["issue_type"]
    band = risk["risk_band"]
    high_risk = band in ("Critical", "High")

    if high_risk:
        rationale.append(f"Rule 2: fused risk is {band} "
                         f"({risk['fused_risk']:.2f}) - retention action required")
        if issue == "Fees_Charges":
            offer = FEE_WAIVER_AED
            cap = OFFER_CAP_PCT * risk["clv_estimate_aed"]
            if offer <= cap:
                rationale.append(f"Rule 2a: fee complaint + offer AED {offer:.0f} "
                                 f"within budget cap AED {cap:.0f} (2% of CLV)")
                return "fee_waiver", rationale, offer
            rationale.append(f"Rule 2a': fee waiver AED {offer:.0f} exceeds "
                             f"budget cap AED {cap:.0f} - downgrade to RM call")
            return "rm_call", rationale, 0.0
        if issue == "Remittance_Transfer":
            offer = REMIT_DISCOUNT_AED
            cap = OFFER_CAP_PCT * risk["clv_estimate_aed"]
            if offer <= cap:
                rationale.append("Rule 2b: remittance friction at high risk - "
                                 "reduce transfer fees, offer transfer help")
                return "remittance_support", rationale, offer
            rationale.append("Rule 2b': offer exceeds budget cap - RM call instead")
            return "rm_call", rationale, 0.0
        if issue == "Loan_Mortgage":
            rationale.append("Rule 2c: loan anxiety at high risk - guidance "
                             "call, no monetary offer needed")
            return "loan_restructure_info", rationale, 0.0
        if risk["segment"] in ("Premium", "Private"):
            rationale.append(f"Rule 2d: {risk['segment']} customer at {band} "
                             "risk - personal relationship-manager call")
            return "rm_call", rationale, 0.0
        rationale.append("Rule 2e: high risk, service issue - fix the issue "
                         "with priority and follow up personally")
        return "service_fix", rationale, 0.0

    if issue in ("App_Technical", "Card_Services"):
        rationale.append("Rule 3: service problem at manageable risk - "
                         "priority fix prevents escalation")
        return "service_fix", rationale, 0.0

    if band == "Medium":
        rationale.append("Rule 4: medium fused risk - proactive check-in "
                         "before it grows")
        return "proactive_checkin", rationale, 0.0

    rationale.append("Rule 5: routine query at low risk - answer well, "
                     "no retention action")
    return "answer_query", rationale, 0.0


# --------------------------------------------------------------- templates
# Curated per-language templates: the own-language fallback that guarantees a
# customer is answered in their language even when the LLM cannot manage it.
# Arabic is formal-warm; Hindi is romanized to match how customers write.
LANG_NAMES = {"en": "English", "ar": "Arabic", "hi": "Hindi (romanized)",
              "tl": "Tagalog"}

_TEMPLATES: dict[str, dict[str, str]] = {
    "dignified_goodbye": {
        "en": ("Dear customer, thank you for the years you have banked with "
               "Falcon Bank. We understand you are relocating, and we want to "
               "make this easy: we will help you close your accounts smoothly, "
               "transfer your funds safely, and provide any documents you "
               "need. Wherever life takes you, our door stays open. "
               "- Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، شكرًا لسنوات ثقتكم في بنك فالكون. نتفهم أنكم "
               "تستعدون لمغادرة الإمارات، ونريد أن نجعل هذه الخطوة سهلة: "
               "سنساعدكم في إغلاق حساباتكم بسلاسة وتحويل أموالكم بأمان وتوفير "
               "أي مستندات تحتاجونها. أينما أخذتكم الحياة، يبقى بابنا مفتوحًا "
               "لكم. - خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, Falcon Bank ke saath itne saalon ke bharose ke "
               "liye dhanyavaad. Hum samajhte hain ki aap UAE chhod rahe hain, "
               "aur hum yeh safar aasaan banana chahte hain: aapke accounts "
               "smoothly band karne, paise surakshit transfer karne aur zaroori "
               "documents dene mein poori madad karenge. Zindagi aapko jahan "
               "bhi le jaaye, hamara darwaza aapke liye khula rahega. "
               "- Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, maraming salamat po sa inyong "
               "pagtitiwala sa Falcon Bank sa loob ng maraming taon. "
               "Nauunawaan namin na aalis na kayo ng UAE, at gusto naming "
               "gawing madali ito: tutulungan namin kayong isara nang maayos "
               "ang inyong mga account, ilipat nang ligtas ang inyong pera, at "
               "ibigay ang anumang dokumentong kailangan ninyo. Saan man kayo "
               "dalhin ng buhay, bukas ang aming pinto para sa inyo. "
               "- Falcon Bank Customer Care"),
    },
    "fee_waiver": {
        "en": ("Dear customer, you are right to raise this, and we are sorry "
               "for the frustration. We have reviewed your account and will "
               "waive the fee of AED {offer:.0f}. Your relationship matters to "
               "us - if anything else feels unclear, reply here and a person "
               "will pick it up. - Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، أنتم محقون في ملاحظتكم ونعتذر عن الإزعاج. "
               "راجعنا حسابكم وسنعفيكم من الرسوم البالغة {offer:.0f} درهم. "
               "علاقتكم تهمنا — إذا كان هناك أي شيء آخر غير واضح، راسلونا هنا "
               "وسيتابع معكم أحد موظفينا. - خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, aapki baat sahi hai aur pareshani ke liye hum "
               "maafi chahte hain. Humne aapka account check kiya hai aur AED "
               "{offer:.0f} ki fee waive kar di jayegi. Aapka rishta hamare "
               "liye important hai - koi aur sawaal ho to yahin reply kariye. "
               "- Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, tama po kayo at humihingi kami ng "
               "paumanhin sa abala. Nasuri na namin ang inyong account at "
               "iwe-waive namin ang bayarin na AED {offer:.0f}. Mahalaga po sa "
               "amin ang inyong pagtitiwala - kung may iba pang hindi malinaw, "
               "mag-reply lang po dito. - Falcon Bank Customer Care"),
    },
    "remittance_support": {
        "en": ("Dear customer, we understand how important your transfers home "
               "are. We have reduced your international transfer fees by AED "
               "{offer:.0f} and a specialist can walk you through the fastest "
               "option for your corridor. - Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، نتفهم أهمية تحويلاتكم إلى عائلاتكم. خفضنا رسوم "
               "التحويل الدولي لكم بمقدار {offer:.0f} درهم، ويمكن لأحد "
               "مختصينا مساعدتكم في اختيار أسرع طريقة للتحويل. "
               "- خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, hum samajhte hain ki ghar paise bhejna aapke "
               "liye kitna zaroori hai. Humne aapki international transfer "
               "fees AED {offer:.0f} kam kar di hai, aur hamara specialist "
               "aapko sabse tez option samjha sakta hai. "
               "- Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, nauunawaan namin kung gaano kahalaga "
               "ang padala ninyo sa inyong pamilya. Binawasan na namin ang "
               "inyong international transfer fee ng AED {offer:.0f}, at may "
               "specialist na tutulong sa inyo para sa pinakamabilis na paraan "
               "ng pagpapadala. - Falcon Bank Customer Care"),
    },
    "loan_restructure_info": {
        "en": ("Dear customer, thank you for telling us about your concerns "
               "with your loan. There are options - restructuring, revised "
               "instalments, or early-settlement guidance - and none of them "
               "require you to decide today. A relationship manager will call "
               "you within one working day to explain them clearly. "
               "- Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، شكرًا لمشاركتنا مخاوفكم بشأن قرضكم. هناك "
               "خيارات متاحة — إعادة الجدولة، تعديل الأقساط، أو السداد المبكر "
               "— ولا شيء يتطلب قرارًا اليوم. سيتصل بكم مدير علاقاتكم خلال "
               "يوم عمل واحد لشرح الخيارات بوضوح. - خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, apne loan ki chinta share karne ke liye "
               "dhanyavaad. Aapke paas options hain - restructuring, EMI mein "
               "badlaav, ya early settlement - aur aaj hi koi faisla lene ki "
               "zaroorat nahi hai. Aapke relationship manager ek working day "
               "ke andar call karke sab saaf saaf samjhayenge. "
               "- Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, salamat po sa pagbahagi ng inyong "
               "alalahanin tungkol sa inyong loan. May mga opsyon po kayo - "
               "restructuring, pagbabago ng buwanang hulog, o maagang "
               "pagbabayad - at hindi ninyo kailangang magpasya ngayon. "
               "Tatawagan po kayo ng inyong relationship manager sa loob ng "
               "isang araw ng trabaho. - Falcon Bank Customer Care"),
    },
    "rm_call": {
        "en": ("Dear customer, thank you for your message. Your relationship "
               "manager will call you within one working day to discuss this "
               "personally and make sure you have what you need. "
               "- Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، شكرًا لرسالتكم. سيتصل بكم مدير علاقاتكم خلال "
               "يوم عمل واحد لمناقشة الأمر شخصيًا والتأكد من حصولكم على كل ما "
               "تحتاجونه. - خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, aapke message ke liye dhanyavaad. Aapke "
               "relationship manager ek working day ke andar aapko call "
               "karenge taaki is baare mein personally baat ho sake. "
               "- Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, salamat po sa inyong mensahe. Tatawagan "
               "po kayo ng inyong relationship manager sa loob ng isang araw "
               "ng trabaho upang mapag-usapan ito nang personal. "
               "- Falcon Bank Customer Care"),
    },
    "service_fix": {
        "en": ("Dear customer, we are sorry for the trouble - this is on us. "
               "Our technical team has been alerted and will resolve the issue "
               "with priority; we will confirm with you as soon as it is "
               "fixed. - Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، نعتذر عن هذا الخلل — المسؤولية علينا. تم "
               "تنبيه فريقنا التقني وسيعالج المشكلة بأولوية، وسنؤكد لكم فور "
               "إصلاحها. - خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, pareshani ke liye maafi chahte hain - yeh "
               "hamari galti hai. Hamari technical team ko alert kar diya gaya "
               "hai aur issue priority par theek hoga. Fix hote hi hum aapko "
               "confirm karenge. - Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, paumanhin po sa abala - kasalanan namin "
               "ito. Naabisuhan na ang aming technical team at aayusin ang "
               "problema nang may priyoridad. Kukumpirmahin namin sa inyo "
               "kapag naayos na. - Falcon Bank Customer Care"),
    },
    "proactive_checkin": {
        "en": ("Dear customer, thank you for reaching out. We have noted your "
               "message and want to make sure you are getting full value from "
               "your accounts - is there anything about your banking we can "
               "make easier? - Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، شكرًا لتواصلكم معنا. سجلنا ملاحظتكم ونريد "
               "التأكد من حصولكم على أفضل قيمة من حساباتكم — هل هناك ما "
               "يمكننا تسهيله في تعاملكم المصرفي؟ - خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, hum tak pahunchne ke liye dhanyavaad. Humne "
               "aapki baat note kar li hai aur yeh pakka karna chahte hain ki "
               "aapko apne accounts ka poora fayda mile - kya banking mein "
               "kuch hai jo hum aasaan bana sakte hain? "
               "- Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, salamat po sa pagsulat. Naitala na "
               "namin ang inyong mensahe at gusto naming matiyak na nakukuha "
               "ninyo ang buong halaga ng inyong mga account - may maitutulong "
               "po ba kami para mas mapadali ang inyong pagbabangko? "
               "- Falcon Bank Customer Care"),
    },
    "answer_query": {
        "en": ("Dear customer, thank you for your question. {query_answer} If "
               "anything is unclear, just reply here. "
               "- Falcon Bank Customer Care"),
        "ar": ("عميلنا العزيز، شكرًا لسؤالكم. سيوافيكم فريقنا بالتفاصيل "
               "الدقيقة قريبًا. إذا كان هناك أي شيء غير واضح، راسلونا هنا. "
               "- خدمة عملاء بنك فالكون"),
        "hi": ("Priya customer, aapke sawaal ke liye dhanyavaad. Hamari team "
               "jald hi aapko poori jaankari degi. Agar kuch samajh nahi aa "
               "raha ho to yahin reply kariye. - Falcon Bank Customer Care"),
        "tl": ("Mahal naming customer, salamat po sa inyong tanong. Ibibigay "
               "po ng aming team ang eksaktong detalye sa lalong madaling "
               "panahon. Kung may hindi malinaw, mag-reply lang po dito. "
               "- Falcon Bank Customer Care"),
    },
    "human_triage": {
        "en": ("[INTERNAL - no customer message drafted] This case is routed "
               "to human triage because the platform's confidence was low."),
    },
}


# ----------------------------------------------------------------- the LLM
SYSTEM_PROMPT = (
    "You are a customer-care writer for Falcon Bank UAE. Write a short, warm, "
    "honest outreach message (60-110 words) to the customer.\n"
    "Hard rules you must never break:\n"
    "1. Write the ENTIRE message in {language}. Not in any other language.\n"
    "2. NO false urgency, NO pressure, NO 'limited time' language.\n"
    "3. Mention ONLY the offer given below. Never invent offers, amounts, "
    "rates, or conditions.\n"
    "4. If the action is a dignified goodbye: thank them genuinely, help them "
    "leave smoothly, and do NOT attempt to retain them.\n"
    "5. Be respectful of a customer who may be under stress; plain language, "
    "no marketing cliches.\n"
    "6. NO placeholders and NO square brackets: you do not know the "
    "customer's name, so begin with 'Dear customer' (in {language}) and sign "
    "off with exactly 'Falcon Bank Customer Care' and nothing else after it."
)

_URGENCY_BLACKLIST = ["act now", "last chance", "limited time", "urgent",
                      "hurry", "immediately", "don't miss", "expires",
                      "only today", "final offer"]

_llm = None
_llm_failed = False


def _get_llm():
    """Lazy-load Qwen2.5-0.5B-Instruct. Set WAFA_USE_LLM=0 to skip."""
    global _llm, _llm_failed
    if os.environ.get("WAFA_USE_LLM", "1") == "0":
        return None
    if _llm is None and not _llm_failed:
        try:
            from transformers import pipeline
            _llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
        except Exception:
            _llm_failed = True
    return _llm


def _guardrails(draft: str, offer: float, action: str) -> list[str]:
    flags = []
    low = draft.lower()
    for phrase in _URGENCY_BLACKLIST:
        if phrase in low:
            flags.append(f"urgency language: '{phrase}'")
    placeholders = re.findall(r"\[[^\]]{0,40}\]|\{[^}]{0,40}\}", draft)
    if placeholders:
        flags.append(f"unfilled placeholders: {placeholders[:3]}")
    amounts = {float(a.replace(",", ""))
               for a in re.findall(r"aed\s*([\d,]+(?:\.\d+)?)", low)}
    allowed = {offer} if offer > 0 else set()
    invented = amounts - allowed
    if invented:
        flags.append(f"invented amounts: {sorted(invented)}")
    if action == "dignified_goodbye" and any(
            w in low for w in ["stay with us", "special offer", "discount",
                               "reconsider", "don't leave", "do not leave"]):
        flags.append("retention pressure in a goodbye message")
    if len(draft.split()) < 20:
        flags.append("draft too short")
    return flags


def _template_draft(action: str, offer: float, lang: str = "en") -> tuple[str, str]:
    """Returns (text, language actually used) - falls back to English if the
    action has no template in the customer's language."""
    variants = _TEMPLATES[action]
    used = lang if lang in variants else "en"
    text = variants[used].format(
        offer=offer,
        query_answer="Our team will get you the exact details shortly.")
    return text, used


def draft_outreach(signals: ListenSignals, risk: RiskAssessment,
                   action: str, rationale: list[str], offer: float,
                   ) -> tuple[str, str, str, list[str], str | None]:
    """Returns (draft, draft_language, source, guardrail_flags, prompt)."""
    lang = signals["language"]
    if action == "human_triage":
        text, used = _template_draft(action, offer, "en")
        return text, used, "template", [], None

    llm = _get_llm()
    prompt = None
    if llm is not None:
        offer_line = (f"Approved offer: AED {offer:.0f} ({ACTION_LABELS[action]})"
                      if offer > 0 else
                      f"Approved action: {ACTION_LABELS[action]} - no monetary offer")
        user_prompt = (
            f"Customer message: \"{signals['text']}\"\n"
            f"Issue: {signals['issue_type']}. Risk band: {risk['risk_band']}.\n"
            f"{offer_line}\n"
            f"Decision reasons: {'; '.join(rationale)}\n"
            f"Write the outreach message now.")
        prompt = SYSTEM_PROMPT.format(language=LANG_NAMES[lang]) + "\n\n" + user_prompt
        try:
            messages = [
                {"role": "system",
                 "content": SYSTEM_PROMPT.format(language=LANG_NAMES[lang])},
                {"role": "user", "content": user_prompt},
            ]
            out = llm(messages, max_new_tokens=220, do_sample=False)
            draft = out[0]["generated_text"][-1]["content"].strip()
            flags = _guardrails(draft, offer, action)
            # fairness guardrail: the draft must be in the customer's language
            from .m1_listen import detect_language
            detected, _ = detect_language(draft)
            if detected != lang:
                flags.append(f"language mismatch: wanted {LANG_NAMES[lang]}, "
                             f"draft came out in {LANG_NAMES[detected]}")
            if not flags:
                return draft, lang, "llm", flags, prompt
            # any violation -> the curated template IN THE CUSTOMER'S LANGUAGE
            text, used = _template_draft(action, offer, lang)
            return text, used, "template", flags, prompt
        except Exception:
            pass
    text, used = _template_draft(action, offer, lang)
    return text, used, "template", [], prompt


# -------------------------------------------------------------- entry point
def decide_and_act(signals: ListenSignals, risk: RiskAssessment) -> ActionPlan:
    action, rationale, offer = decide(signals, risk)
    cap = OFFER_CAP_PCT * risk["clv_estimate_aed"]
    draft, draft_lang, source, flags, prompt = draft_outreach(
        signals, risk, action, rationale, offer)
    return {
        "customer_id": signals["customer_id"],
        "action": action,
        "action_label": ACTION_LABELS[action],
        "rationale": rationale,
        "offer_value_aed": offer,
        "offer_within_budget": offer <= cap if offer > 0 else True,
        "draft_message": draft,
        "draft_language": draft_lang,
        "draft_source": source,
        "guardrail_flags": flags,
        "prompt_used": prompt,
    }
