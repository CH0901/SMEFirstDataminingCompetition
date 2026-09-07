"""전국 피처를 타깃별로 최적화 + 광역(권역) 평균도 시험."""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
FB={"TA14":feats,"HM14":[f for f in feats if f!="YEAR"]}
KEY=[c for c in df.columns if any(c.startswith(p) for p in
     ("IR105_m9_","IR105_min9_","IR123_m9_","SW038_m9_","SWD_m9_","SWDmin_"))]
nat=df.groupby("date")[KEY].transform("mean")
add={}
for c in KEY:
    add[f"NAT_{c}"]=nat[c]; add[f"ANO_{c}"]=df[c]-nat[c]
# 권역 평균 (위도 3분할) — 전국보다 국지적인 기단
df["_z"]=pd.cut(df.LAT,3,labels=False)
reg=df.groupby(["date","_z"])[KEY].transform("mean")
for c in KEY: add[f"REG_{c}"]=df[c]-reg[c]
df=pd.concat([df,pd.DataFrame(add,index=df.index)],axis=1)
NAT=[f"NAT_{c}" for c in KEY]; ANO=[f"ANO_{c}" for c in KEY]; REG=[f"REG_{c}" for c in KEY]
df["STN_CAT"]=df["STN"].astype("category")
P=dict(T.PARAMS,num_leaves=63); R=800
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]

def pr(extra,tgt):
    F=FB[tgt]+extra; s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    m=lgb.train(P,lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),
                categorical_feature=["STN_CAT"]),R)
    p=m.predict(te[F+["STN_CAT"]])+lin.predict(te[F].astype(np.float64))
    return float(np.sqrt(np.nanmean((p-te[tgt].to_numpy())**2)))

opts=[("없음",[]),("전국평균",NAT),("편차",ANO),("편차+권역",ANO+REG),("전부",NAT+ANO+REG)]
print(f"{'전국 피처':16s} {'RMSE_TA':>9s} {'RMSE_HM':>9s}")
print(f"{'-'*16} {'-'*9} {'-'*9}")
res={}
for n,e in opts:
    t=pr(e,"TA14"); h=pr(e,"HM14"); res[n]=(t,h)
    print(f"  {n:14s} {t:9.3f} {h:9.3f}", flush=True)
bt=min(res,key=lambda k:res[k][0]); bh=min(res,key=lambda k:res[k][1])
print(f"\n  최적 조합: TA={bt} · HM={bh}")
print(f"  Score {res[bt][0]+0.1*res[bh][1]:.3f}   (현재 2.143)")
