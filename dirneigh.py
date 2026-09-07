"""방향별 이웃 — 기단이 어느 쪽에서 오는지.

이웃 8개 평균이 통했다(기온 1.304 -> 1.252). 그런데 평균은 방향을 지운다.
서쪽 이웃이 뜨겁고 동쪽이 차가우면 서풍 기단 유입이고, 그 반대면 다른 상황이다.
동해안 푄(태백산맥 서쪽 구름 / 동쪽 맑음)도 정확히 방향 구조다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
ex=[c for c in df.columns if c.startswith(("ANO_","NB_"))]
FB={"TA14":feats+ex,"HM14":[f for f in feats if f!="YEAR"]}
KEY=[c for c in df.columns if c.endswith("_h05") and any(c.startswith(p) for p in
     ("IR105_m9","IR105_min9","SWD_m9","SW038_m9")) or c=="SWDmin_h05"]
stn=pd.read_csv("stations_scored.csv")
L,O,S=stn.LAT.values,stn.LON.values,stn.STN.values
dy=(L[:,None]-L[None,:])*111
dx=(O[:,None]-O[None,:])*111*np.cos(np.radians(L[:,None]))
D=np.sqrt(dx**2+dy**2); np.fill_diagonal(D,1e9)
# 방향별로 가장 가까운 3개 (동/서/남/북 각각 45도 섹터)
sect={"W":(dx<0)&(np.abs(dx)>np.abs(dy)), "E":(dx>0)&(np.abs(dx)>np.abs(dy)),
      "S":(dy<0)&(np.abs(dy)>=np.abs(dx)), "N":(dy>0)&(np.abs(dy)>=np.abs(dx))}
groups={}
for d,mask in sect.items():
    nb={}
    for i in range(len(S)):
        cand=np.where(mask[i])[0]
        if len(cand)==0: nb[S[i]]=[]; continue
        nb[S[i]]=list(S[cand[np.argsort(D[i,cand])[:3]]])
    for c in KEY:
        piv=df.pivot_table(index="date",columns="STN",values=c)
        m=pd.DataFrame({s_: (piv[[x for x in nb[s_] if x in piv.columns]].mean(axis=1)
                             if nb.get(s_) else np.nan) for s_ in S if s_ in piv.columns})
        st=m.stack().rename(f"{d}_{c}").reset_index(); st.columns=["date","STN",f"{d}_{c}"]
        df=df.merge(st,on=["date","STN"],how="left")
    groups[d]=[x for x in df.columns if x.startswith(f"{d}_") and x.endswith("_h05")]
DIR=sum(groups.values(),[])
# 동서 / 남북 대비
CON=[]
for c in KEY:
    if f"W_{c}" in df and f"E_{c}" in df: df[f"dEW_{c}"]=df[f"E_{c}"]-df[f"W_{c}"]; CON.append(f"dEW_{c}")
    if f"N_{c}" in df and f"S_{c}" in df: df[f"dNS_{c}"]=df[f"S_{c}"]-df[f"N_{c}"]; CON.append(f"dNS_{c}")
df["STN_CAT"]=df["STN"].astype("category")
print(f"방향 피처 {len(DIR)}개 · 대비 {len(CON)}개")
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
for n,e in [("현재",[]),("+방향별 이웃",DIR),("+동서·남북 대비",CON),("+둘 다",DIR+CON)]:
    print(f"  {n:20s} {pr(e,'TA14'):9.3f}", flush=True)
