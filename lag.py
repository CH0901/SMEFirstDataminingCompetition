"""며칠 전 위성까지 보면 '이번 주가 얼마나 더운지' 가 잡히는가.

기단은 3~5일 지속된다. 대상일 14시(05 UTC) 이전이면 전날·전전날 영상도
전부 합법인데, 지금은 전날 저녁(18/21 UTC)만 쓴다.

D-1, D-2, D-3 의 05 UTC 영상을 붙인다. 특히 그 날들의 **전국 평균**이
'이번 주 기단 수준' 을 직접 나타낸다. 기존 수집분에 연속된 날이 이미
들어있으므로 새로 받을 필요가 없다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eval14 as E, kma, train as T

CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
df["STN_CAT"]=df["STN"].astype("category")
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH
              or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
SP=[c for c in df.columns if c.startswith(("ANO_","NB_","DW_","DE_","DN_","DS_","dEW_","dNS_"))]

# 지연시킬 열: 14시 시점(_h05) 의 핵심 지표들
LAGK=[c for c in df.columns if c.endswith("_h05") and
      (c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_")))]
print(f"지연 대상 {len(LAGK)}개 열 x 3일")
key=df[["STN","date"]+LAGK].copy()
LAGF=[]
for k in (1,2,3):
    s=key.copy(); s["date"]=s["date"]+pd.Timedelta(days=k)
    s=s.rename(columns={c:f"L{k}_{c}" for c in LAGK})
    df=df.merge(s,on=["STN","date"],how="left")
    LAGF+= [f"L{k}_{c}" for c in LAGK]
# 지연값의 전국 평균 = 그 날 전국이 얼마나 더웠나
NATL=[]
for c in LAGF:
    n=f"N{c}"; df[n]=df.groupby("date")[c].transform("mean"); NATL.append(n)
# 최근 3일 추세 (오늘 - 3일전)
TR=[]
for c in LAGK:
    if f"NL3_{c}" in df:
        n=f"TREND_{c}"
        df[n]=df.groupby("date")[c].transform("mean")-df[f"NL3_{c}"]; TR.append(n)

miss=df[LAGF].isna().mean().mean()*100
print(f"지연 피처 {len(LAGF)}개 · 전국평균 {len(NATL)}개 · 추세 {len(TR)}개 · 결측 {miss:.1f}%\n")

W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
P=dict(T.PARAMS,num_leaves=63); R=800
def run(F,tgt="TA14"):
    s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    C=F+["STN_CAT"]
    m=lgb.train(P,lgb.Dataset(s[C],y-lin.predict(X),
                categorical_feature=["STN_CAT"]),R)
    p=m.predict(te[C])+lin.predict(te[F].astype(np.float64))
    return float(np.sqrt(np.nanmean((p-te[tgt].to_numpy())**2)))

b=run(feats+SP); print(f"  {'현재':30s} 기온 {b:.3f}")
for nm,ex in [("+ D-1~3 지점값",       LAGF),
              ("+ D-1~3 전국평균",     NATL),
              ("+ 3일 추세",           TR),
              ("+ 전국평균 · 추세",     NATL+TR),
              ("+ 전부",               LAGF+NATL+TR)]:
    r=run(feats+SP+ex)
    print(f"  {nm:30s} 기온 {r:.3f}  ({r-b:+.3f})", flush=True)
