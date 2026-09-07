"""최근접 이웃 지점의 위성값.

전국 평균이 통했다(기온 1.435 -> 1.304). 그렇다면 그 중간 규모, 즉 가까운
몇 개 지점의 위성 신호도 정보가 될 수 있다. 이웃의 '위성' 값이므로 규칙상
허용된다 (ASOS 가 아니다).
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sme.core import evaluation as E
from sme.core import kma_client
from sme.model import train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma_client.DEFAULT_CHANNELS); df=E.assemble(CH)
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
ano=[c for c in df.columns if c.startswith("ANO_")]
FB={"TA14":feats+ano,"HM14":[f for f in feats if f!="YEAR"]}
KEY=[c for c in df.columns if any(c.startswith(p) for p in
     ("IR105_m9_h05","IR105_min9_h05","SWD_m9_h05","SWDmin_h05","SW038_m9_h05"))]
stn=pd.read_csv("data/reference/stations_scored.csv")
# 각 지점의 최근접 K개
L,O,S=stn.LAT.values,stn.LON.values,stn.STN.values
D=np.sqrt(((L[:,None]-L[None,:])*111)**2+((O[:,None]-O[None,:])*111*np.cos(np.radians(L[:,None])))**2)
np.fill_diagonal(D,1e9)
add={}
for K in (3,8):
    idx=np.argsort(D,axis=1)[:,:K]
    nb={S[i]: S[idx[i]] for i in range(len(S))}
    piv=df.pivot_table(index="date",columns="STN",values=KEY)
    vals={c: np.full(len(df),np.nan) for c in KEY}
    pos={(d,s):i for i,(d,s) in enumerate(zip(df.date,df.STN))}
    for c in KEY:
        sub=piv[c]
        m=pd.DataFrame({s: sub[list(nb[s])].mean(axis=1) for s in S if s in sub.columns})
        st=m.stack().rename(f"NB{K}_{c}").reset_index()
        st.columns=["date","STN",f"NB{K}_{c}"]
        df=df.merge(st,on=["date","STN"],how="left")
NB3=[c for c in df.columns if c.startswith("NB3_")]
NB8=[c for c in df.columns if c.startswith("NB8_")]
# 이웃 대비 편차
for base,lst in (("NB3",NB3),("NB8",NB8)):
    for c in lst:
        k=c[len(base)+1:]
        if k in df: df[f"D{base}_{k}"]=df[k]-df[c]
D3=[c for c in df.columns if c.startswith("DNB3_")]; D8=[c for c in df.columns if c.startswith("DNB8_")]
df["STN_CAT"]=df["STN"].astype("category")
print(f"이웃3 {len(NB3)}개 · 이웃8 {len(NB8)}개 · 편차 {len(D3)}/{len(D8)}개")
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
print(f"\n{'구성':22s} {'RMSE_TA':>9s}")
print(f"{'-'*22} {'-'*9}")
for n,e in [("현재",[]),("+이웃3",NB3),("+이웃8",NB8),("+이웃3 편차",D3),("+이웃8 편차",D8),("+이웃3+편차",NB3+D3)]:
    print(f"  {n:20s} {pr(e,'TA14'):9.3f}", flush=True)
