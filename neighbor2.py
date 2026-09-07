"""이웃 8개 피처를 전 시각으로 확장하고 K 를 더 넓혀본다."""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
ano=[c for c in df.columns if c.startswith("ANO_")]
FB={"TA14":feats+ano,"HM14":[f for f in feats if f!="YEAR"]}
# 전 시각 핵심 통계
KEY=[c for c in df.columns if any(c.startswith(p) for p in
     ("IR105_m9_","IR105_min9_","SWD_m9_","SWDmin_","SW038_m9_"))]
print(f"이웃 피처 대상 {len(KEY)}개 (전 시각)")
stn=pd.read_csv("stations_scored.csv")
L,O,S=stn.LAT.values,stn.LON.values,stn.STN.values
D=np.sqrt(((L[:,None]-L[None,:])*111)**2+((O[:,None]-O[None,:])*111*np.cos(np.radians(L[:,None])))**2)
np.fill_diagonal(D,1e9)
groups={}
for K in (8,16):
    idx=np.argsort(D,axis=1)[:,:K]
    nb={S[i]: list(S[idx[i]]) for i in range(len(S))}
    piv=df.pivot_table(index="date",columns="STN",values=KEY)
    cols={}
    for c in KEY:
        sub=piv[c]
        m=pd.DataFrame({s: sub[[x for x in nb[s] if x in sub.columns]].mean(axis=1)
                        for s in S if s in sub.columns})
        st=m.stack().rename(f"N{K}_{c}").reset_index()
        st.columns=["date","STN",f"N{K}_{c}"]
        df=df.merge(st,on=["date","STN"],how="left")
    groups[K]=[c for c in df.columns if c.startswith(f"N{K}_")]
    print(f"  K={K}: {len(groups[K])}개 생성")
df["STN_CAT"]=df["STN"].astype("category")
P=dict(T.PARAMS,num_leaves=63); R=800
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
def pr(extra,tgt):
    F=FB[tgt]+extra; s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    m=lgb.train(P,lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),R)
    p=m.predict(te[F+["STN_CAT"]])+lin.predict(te[F].astype(np.float64))
    return float(np.sqrt(np.nanmean((p-te[tgt].to_numpy())**2)))
print(f"\n{'구성':24s} {'RMSE_TA':>9s} {'RMSE_HM':>9s}")
print(f"{'-'*24} {'-'*9} {'-'*9}")
for n,e in [("현재",[]),("+이웃8 전시각",groups[8]),("+이웃16 전시각",groups[16])]:
    print(f"  {n:22s} {pr(e,'TA14'):9.3f} {pr(e,'HM14'):9.3f}", flush=True)
