"""
Project Wafa - Synthetic Dataset Generator (Falcon Bank UAE)
=============================================================
Produces two CSVs joined on customer_id:
  1. messages.csv  - multilingual customer messages (text -> NLP layer)
  2. customers.csv - banking profiles with churn ground truth (tabular -> ML layer)

Design goals:
  * Balanced across issue types and churn-signal levels
  * Churn signal NOT trivially tied to language or nationality (fairness)
  * Multilingual: English, Arabic, Hindi, Tagalog (romanized + native scripts)
"""
import random, numpy as np, pandas as pd
random.seed(42); np.random.seed(42)

ISSUES = ["Account_Closure","Remittance_Transfer","Loan_Mortgage","Fees_Charges",
          "Card_Services","App_Technical","General_Query"]
SIGNALS = ["High","Medium","Low"]          # churn signal expressed in the text
LANGS   = ["en","ar","hi","tl"]
N_PER_CELL = 12                             # 7 issues x 3 signals x 12 = 252 base (en)
N_CUSTOMERS = 240

# --- message templates: issue body x churn-signal wrapper, per language ---
BODY = {
 "Account_Closure":{
  "en":["I want to close my account","how do I close my salary account","what documents are needed to close accounts"],
  "ar":["أريد إغلاق حسابي","كيف أغلق حساب الراتب الخاص بي"],
  "hi":["mujhe apna account band karna hai","salary account kaise band hoga"],
  "tl":["gusto ko nang isara ang account ko","paano isara ang aking account"]},
 "Remittance_Transfer":{
  "en":["I need to transfer all my funds abroad","how long does an international transfer take","my remittance to home country failed"],
  "ar":["أحتاج تحويل أموالي إلى الخارج","فشلت حوالتي المالية"],
  "hi":["mujhe apne paise India bhejne hain","mera transfer fail ho gaya"],
  "tl":["kailangan kong ipadala ang pera ko sa Pilipinas","nabigo ang remittance ko"]},
 "Loan_Mortgage":{
  "en":["what happens to my loan if I leave the UAE","can I settle my mortgage early","my EMI is too high now"],
  "ar":["ماذا يحدث لقرضي إذا غادرت الإمارات","هل يمكنني سداد الرهن مبكرا"],
  "hi":["agar main UAE chhod doon to loan ka kya hoga","EMI bahut zyada hai"],
  "tl":["ano ang mangyayari sa loan ko kung aalis ako ng UAE","masyadong mataas ang hulog ko"]},
 "Fees_Charges":{
  "en":["why was I charged this maintenance fee","these charges are unfair","refund the hidden fee please"],
  "ar":["لماذا خصمتم هذه الرسوم","هذه الرسوم غير عادلة"],
  "hi":["yeh charges kyun lage hain","hidden fee wapas karo"],
  "tl":["bakit may bayad na ganito","hindi makatarungan ang singil"]},
 "Card_Services":{
  "en":["my card was declined abroad","I need to block my lost card","increase my card limit"],
  "ar":["تم رفض بطاقتي في الخارج","أريد إيقاف بطاقتي المفقودة"],
  "hi":["mera card foreign mein decline ho gaya","card block karna hai"],
  "tl":["na-decline ang card ko sa abroad","kailangan kong i-block ang card ko"]},
 "App_Technical":{
  "en":["the app keeps crashing when I log in","I cannot see my statements online","OTP never arrives"],
  "ar":["التطبيق يتعطل عند تسجيل الدخول","لا يصلني رمز التحقق"],
  "hi":["app baar baar crash ho raha hai","OTP nahi aa raha"],
  "tl":["laging nagko-crash ang app","hindi dumarating ang OTP"]},
 "General_Query":{
  "en":["what are the branch timings","how do I update my Emirates ID","do you have a branch in Sharjah"],
  "ar":["ما هي أوقات عمل الفرع","كيف أحدث الهوية الإماراتية"],
  "hi":["branch timing kya hai","Emirates ID kaise update karein"],
  "tl":["ano ang oras ng branch","paano i-update ang Emirates ID ko"]},
}
WRAP = {
 "High":{
  "en":["I am leaving the UAE next month.","We are relocating for good.","I have already resigned and am moving back home.","This is my final month here."],
  "ar":["سأغادر الإمارات الشهر القادم.","نحن ننتقل نهائيا."],
  "hi":["main agle mahine UAE chhod raha hoon.","hum hamesha ke liye ja rahe hain."],
  "tl":["aalis na ako ng UAE sa susunod na buwan.","lilipat na kami nang tuluyan."]},
 "Medium":{
  "en":["I am not sure about my plans here anymore.","Things are uncertain for my family.","I am very frustrated with the service lately.","I may have to make some big decisions soon."],
  "ar":["لست متأكدا من خططي هنا.","أنا محبط جدا من الخدمة."],
  "hi":["mujhe apne plans ka pata nahi.","main service se pareshan hoon."],
  "tl":["hindi ako sigurado sa mga plano ko.","dismayado ako sa serbisyo."]},
 "Low":{
  "en":["Thanks in advance.","Just checking.","Appreciate your help.","No rush at all."],
  "ar":["شكرا مقدما.","مجرد استفسار."],
  "hi":["dhanyavaad.","bas pooch raha tha."],
  "tl":["salamat po.","nagtatanong lang po."]},
}

rows=[]; mid=1
cust_ids=[f"FB{1000+i}" for i in range(N_CUSTOMERS)]
for issue in ISSUES:
    for sig in SIGNALS:
        for k in range(N_PER_CELL):
            # language mix: 55% en, 20% ar, 15% hi, 10% tl -- independent of signal
            lang = random.choices(LANGS, weights=[55,20,15,10])[0]
            body = random.choice(BODY[issue].get(lang, BODY[issue]["en"]))
            wrap = random.choice(WRAP[sig].get(lang, WRAP[sig]["en"]))
            text = f"{wrap} {body}." if random.random()<0.5 else f"{body}. {wrap}"
            rows.append({"message_id":f"M{mid:04d}","customer_id":random.choice(cust_ids),
                         "text":text,"language":lang,"issue_type":issue,"churn_signal":sig})
            mid+=1
messages=pd.DataFrame(rows).sample(frac=1,random_state=42).reset_index(drop=True)

# --- customer profiles with churn ground truth ---
REGIONS=["South_Asia","Southeast_Asia","MENA","Western","East_Asia"]
SEGMENTS=["Mass","Premium","Private"]
profs=[]
for cid in cust_ids:
    churned = random.random()<0.35                       # historical churn base rate
    tenure = random.randint(3,180)
    seg = random.choices(SEGMENTS,weights=[60,30,10])[0]
    bal = round(np.random.lognormal(9.5 if seg=="Mass" else 11 if seg=="Premium" else 12.3, 0.6),2)
    # churn drivers: balance decline, salary stop, remittance spike -- correlated with churned
    if churned:
        bal_trend = round(random.uniform(-0.65,-0.15),2)
        salary_active = random.random()<0.35
        remit_spike = random.random()<0.7
    else:
        bal_trend = round(random.uniform(-0.15,0.25),2)
        salary_active = random.random()<0.9
        remit_spike = random.random()<0.15
    profs.append({"customer_id":cid,
        "nationality_region":random.choice(REGIONS),      # independent of churn (fairness)
        "tenure_months":tenure,"segment":seg,
        "products_held":random.randint(1,6),
        "avg_balance_aed":bal,"balance_trend_3m":bal_trend,
        "salary_credit_active":salary_active,
        "remittance_count_3m":random.randint(0,12),
        "intl_transfer_spike":remit_spike,
        "complaints_6m":random.randint(0,6),
        "branch_visits_trend":round(random.uniform(-0.8,0.4),2),
        "clv_estimate_aed":int(bal*random.uniform(0.5,2.5)),
        "churned":churned})
customers=pd.DataFrame(profs)

messages.to_csv("messages.csv",index=False); customers.to_csv("customers.csv",index=False)
print("messages:",messages.shape,"customers:",customers.shape)
print("\nIssue x Signal balance:\n",pd.crosstab(messages.issue_type,messages.churn_signal))
print("\nLanguage x Signal (should be proportional, not skewed):\n",pd.crosstab(messages.language,messages.churn_signal))
print("\nChurn x Region (fairness -- should be ~even):\n",pd.crosstab(customers.nationality_region,customers.churned,normalize='index').round(2))
print("\nChurn drivers separate?\n",customers.groupby('churned')[['balance_trend_3m','salary_credit_active','intl_transfer_spike']].mean().round(2))
