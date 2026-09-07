"""어느 연도까지 넣는 게 최적인가."""
import numpy as np, pandas as pd, lightgbm as lgb
from sme.core import evaluation as E
from sme.core import kma_client
from sme.model import train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma_client.DEFAULT_CHANNELS); df=E.assemble(CH)
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
df["STN_CAT"]=df["STN"].astype("category"); df["year"]=df.date.dt.year
TEST=pd.date_range("2025-08-24","2025-08-30")
te=df[df.date.isin(TEST)]
print(f"연도별 표본: {df.year.value_counts().sort_index().to_dict()}\n")

def run(tr,name):
    out={}
    for tgt in ("TA14","HM14"):
        s=tr[tr[tgt].notna()]
        X=s[FE].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        ms=[lgb.train(dict(T.PARAMS,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
            lgb.Dataset(s[FE+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),T.ROUNDS)
            for sd in (42,7,2024)]
        out[tgt]=np.mean([m.predict(te[FE+["STN_CAT"]]) for m in ms],axis=0)\
                 +lin.predict(te[FE].astype(np.float64))
    rt=np.sqrt(np.nanmean((out["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((out["HM14"]-te.HM14.to_numpy())**2))
    print(f"  {name:24s} Score {rt+0.1*rh:6.3f}  TA {rt:.3f}  (학습 {len(tr):,}행 / {tr.date.nunique()}일)", flush=True)

base=df[~df.date.isin(TEST)]
print(f"{'학습 연도':24s} {'결과'}")
print(f"{'-'*24} {'-'*56}")
for lo,name in [(2023,"2023~2026 (3년, 기존)"),(2022,"2022~2026 (4년)"),
                (2021,"2021~2026 (5년)"),(2019,"2019~2026 (7년)")]:
    run(base[base.year>=lo], name)
