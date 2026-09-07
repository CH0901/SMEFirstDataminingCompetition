"""타깃별로 연도 피처를 다르게 쓴다.

기온은 연도를 넣어야 편향이 잡히고(-0.54 -> +0.15), 습도는 빼는 게 낫다.
지표가 TA + 0.1*HM 이므로 각각 최적인 구성을 따로 쓰면 된다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
ALL=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
ALL=[f for f in dict.fromkeys(ALL) if f in df.columns]
NOYR=[f for f in ALL if f!="YEAR"]
df["STN_CAT"]=df["STN"].astype("category")
P=dict(T.PARAMS,num_leaves=63); R=800

def pred(F,tr,te,tgt):
    s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    m=lgb.train(P,lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),
                categorical_feature=["STN_CAT"]),R)
    return m.predict(te[F+["STN_CAT"]])+lin.predict(te[F].astype(np.float64))

def show(tr,te,label):
    ta_y=pred(ALL,tr,te,"TA14"); ta_n=pred(NOYR,tr,te,"TA14")
    hm_y=pred(ALL,tr,te,"HM14"); hm_n=pred(NOYR,tr,te,"HM14")
    def r(p,t): return float(np.sqrt(np.nanmean((p-te[t].to_numpy())**2)))
    combos=[("TA:연도O  HM:연도O", r(ta_y,"TA14"), r(hm_y,"HM14")),
            ("TA:연도X  HM:연도X", r(ta_n,"TA14"), r(hm_n,"HM14")),
            ("TA:연도O  HM:연도X", r(ta_y,"TA14"), r(hm_n,"HM14")),
            ("TA:연도X  HM:연도O", r(ta_n,"TA14"), r(hm_y,"HM14"))]
    print(f"\n[{label}]")
    for n,rt,rh in combos:
        print(f"  {n:22s} Score {rt+0.1*rh:6.3f}   TA {rt:.3f}  HM {rh:.3f}", flush=True)

W=pd.date_range("2026-08-16","2026-08-22")
show(df[df.date<pd.Timestamp("2026-08-16")], df[df.date.isin(W)], "2026-08-16~22 (실전 재현)")
T25=pd.date_range("2025-08-24","2025-08-30")
show(df[~df.date.isin(T25)], df[df.date.isin(T25)], "2025-08-24~30 (참고)")
