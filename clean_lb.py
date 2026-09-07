"""리더보드 구간을 학습에서 빼고 정직하게 재본다."""
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
df["STN_CAT"]=df["STN"].astype("category")
P=dict(T.PARAMS,num_leaves=63); R=800
W=pd.date_range("2026-06-24","2026-06-30"); te=df[df.date.isin(W)]
for name,tr in [("누수 있음 (제출본)", df),
                ("누수 제거 (해당 주 제외)", df[~df.date.isin(W)])]:
    r={}
    for tgt in ("TA14","HM14"):
        F=FB[tgt]; s=tr[tr[tgt].notna()]
        X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        m=lgb.train(P,lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),R)
        r[tgt]=m.predict(te[F+["STN_CAT"]])+lin.predict(te[F].astype(np.float64))
    rt=np.sqrt(np.nanmean((r["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((r["HM14"]-te.HM14.to_numpy())**2))
    print(f"  {name:26s} Score {rt+0.1*rh:6.3f}   TA {rt:.3f}   HM {rh:.3f}", flush=True)
