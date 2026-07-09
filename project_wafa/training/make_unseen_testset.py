"""
Generator extension: an UNSEEN-template test set for the honest evaluation.

The provided messages.csv is template-generated, so any random train/test
split shares templates between sides and near-perfect scores are guaranteed.
This script extends the generator with entirely NEW phrasings (none appear in
generate_wafa_data.py) and emits data/messages_unseen.csv - same schema, same
issue x signal x language structure. Models never see these templates in
training, so accuracy here is the number that belongs in the report.

Run:  python training/make_unseen_testset.py
"""
from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 7            # different seed from the original generator on purpose
N_PER_CELL = 6      # 7 issues x 3 signals x 6 = 126 messages

ISSUES = ["Account_Closure", "Remittance_Transfer", "Loan_Mortgage",
          "Fees_Charges", "Card_Services", "App_Technical", "General_Query"]
SIGNALS = ["High", "Medium", "Low"]
LANGS = ["en", "ar", "hi", "tl"]

NEW_BODY = {
 "Account_Closure": {
  "en": ["please tell me the procedure to shut down all my accounts",
         "I wish to terminate my banking relationship with you"],
  "ar": ["ما هي إجراءات إنهاء جميع حساباتي لديكم",
         "أرغب في إنهاء تعاملي المصرفي معكم"],
  "hi": ["sare accounts khatam karne ka process kya hai",
         "mujhe apni poori banking yahan se khatam karni hai"],
  "tl": ["ano ang proseso para tanggalin ang lahat ng account ko",
         "gusto ko nang tapusin ang bank account ko dito"]},
 "Remittance_Transfer": {
  "en": ["I want to send my entire savings to my family overseas",
         "why is my money transfer stuck since yesterday"],
  "ar": ["أريد إرسال كل مدخراتي إلى عائلتي في الخارج",
         "لماذا تأخرت حوالتي المالية منذ الأمس"],
  "hi": ["poori savings ghar bhejni hai mujhe",
         "kal se mera paisa transfer mein atka hua hai"],
  "tl": ["gusto kong ipadala ang buong ipon ko sa pamilya ko",
         "bakit naantala ang padala ko kahapon pa"]},
 "Loan_Mortgage": {
  "en": ["is it possible to pay off my personal loan before leaving",
         "the monthly installment has become impossible for me"],
  "ar": ["هل يمكن سداد قرضي الشخصي قبل السفر",
         "القسط الشهري أصبح مستحيلا علي"],
  "hi": ["kya main jaane se pehle apna loan chuka sakta hoon",
         "har mahine ki kist ab bahut mushkil ho gayi hai"],
  "tl": ["pwede bang bayaran ko nang buo ang utang ko bago umalis",
         "hindi ko na kaya ang buwanang hulog ngayon"]},
 "Fees_Charges": {
  "en": ["there is a deduction on my statement I never agreed to",
         "your service charges doubled without any notice"],
  "ar": ["يوجد خصم في كشف حسابي لم أوافق عليه",
         "تضاعفت رسوم الخدمة دون أي إشعار"],
  "hi": ["statement mein aisa charge hai jo maine approve nahi kiya",
         "bina bataye service charge double ho gaya"],
  "tl": ["may kaltas sa statement ko na hindi ko inaprubahan",
         "nadoble ang singil ninyo nang walang abiso"]},
 "Card_Services": {
  "en": ["my credit card stopped working at the supermarket today",
         "someone may have stolen my card details please help"],
  "ar": ["توقفت بطاقتي الائتمانية عن العمل اليوم",
         "أعتقد أن بيانات بطاقتي قد سُرقت"],
  "hi": ["aaj supermarket mein mera card kaam nahi kar raha tha",
         "lagta hai kisi ne mere card ki details chura li"],
  "tl": ["hindi gumana ang credit card ko sa grocery kanina",
         "baka ninakaw ang detalye ng card ko tulungan niyo ako"]},
 "App_Technical": {
  "en": ["the mobile banking keeps logging me out every minute",
         "transaction history will not load on the app"],
  "ar": ["تطبيق الهاتف يخرجني من حسابي كل دقيقة",
         "سجل المعاملات لا يظهر في التطبيق"],
  "hi": ["mobile banking baar baar logout kar raha hai",
         "app mein transaction history nahi khul rahi"],
  "tl": ["lagi akong nalo-logout sa mobile banking",
         "ayaw magbukas ng transaction history sa app"]},
 "General_Query": {
  "en": ["which documents do I need to update my mobile number",
         "is the bank open during the public holiday"],
  "ar": ["ما المستندات المطلوبة لتحديث رقم هاتفي",
         "هل البنك مفتوح في العطلة الرسمية"],
  "hi": ["mobile number update karne ke liye kya documents chahiye",
         "kya bank chhutti ke din khula rehta hai"],
  "tl": ["anong dokumento ang kailangan para palitan ang numero ko",
         "bukas ba ang bangko sa holiday"]},
}

NEW_WRAP = {
 "High": {
  "en": ["My family has already flown home and I follow soon.",
         "My employer has ended my contract so we are packing up.",
         "I am winding down everything here before we go."],
  "ar": ["عائلتي سافرت بالفعل وسألحق بهم قريبا.",
         "انتهى عقد عملي وسنغادر البلاد."],
  "hi": ["meri family wapas ja chuki hai, main bhi jald jaunga.",
         "meri naukri khatam ho gayi, hum desh chhod rahe hain."],
  "tl": ["nakauwi na ang pamilya ko at susunod na ako.",
         "natapos na ang kontrata ko kaya uuwi na kami."]},
 "Medium": {
  "en": ["Honestly I am rethinking whether to keep banking here.",
         "Everything feels up in the air for us right now.",
         "I keep getting disappointed by this bank lately."],
  "ar": ["بصراحة أعيد التفكير في بقائي مع هذا البنك.",
         "كل شيء غير واضح بالنسبة لنا الآن."],
  "hi": ["sach kahoon to soch raha hoon yahan banking rakhoon ya nahi.",
         "abhi hamare liye sab kuch uljhan mein hai."],
  "tl": ["sa totoo lang iniisip ko kung itutuloy ko pa dito.",
         "magulo pa ang lahat para sa amin ngayon."]},
 "Low": {
  "en": ["Have a great day.", "Just wanted to ask.", "Many thanks.",
         "Whenever you get a chance."],
  "ar": ["أتمنى لكم يوما سعيدا.", "فقط أردت السؤال."],
  "hi": ["aapka din shubh ho.", "bas jaanna tha."],
  "tl": ["magandang araw po.", "gusto ko lang malaman."]},
}


def main() -> None:
    random.seed(SEED)
    customers = pd.read_csv(ROOT / "data" / "customers.csv")
    cust_ids = customers["customer_id"].tolist()

    rows, mid = [], 1
    for issue in ISSUES:
        for sig in SIGNALS:
            for _ in range(N_PER_CELL):
                lang = random.choices(LANGS, weights=[55, 20, 15, 10])[0]
                body = random.choice(NEW_BODY[issue].get(lang, NEW_BODY[issue]["en"]))
                wrap = random.choice(NEW_WRAP[sig].get(lang, NEW_WRAP[sig]["en"]))
                text = (f"{wrap} {body}." if random.random() < 0.5
                        else f"{body}. {wrap}")
                rows.append({"message_id": f"U{mid:04d}",
                             "customer_id": random.choice(cust_ids),
                             "text": text, "language": lang,
                             "issue_type": issue, "churn_signal": sig})
                mid += 1

    df = pd.DataFrame(rows).sample(frac=1, random_state=SEED).reset_index(drop=True)
    out = ROOT / "data" / "messages_unseen.csv"
    df.to_csv(out, index=False)

    # sanity: no template overlap with the original file
    orig = pd.read_csv(ROOT / "data" / "messages.csv")
    overlap = set(df["text"]) & set(orig["text"])
    print(f"wrote {out} ({len(df)} messages)")
    print("language mix:", df["language"].value_counts().to_dict())
    print(f"text overlap with training data: {len(overlap)} (must be 0)")


if __name__ == "__main__":
    main()
