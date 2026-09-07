"""연도 범위 · 연도 피처 비교 (상대 비교이므로 가볍게)."""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
df["YEAR"]=df.date.dt.year.astype(float)
df["STN_CAT"]=df["STN"].astype("category"); df["y"]=df.date.dt.year
TEST=pd.date_range("2025-08-24","2025-08-30")
tr,te=df[~df.date.isin(TEST)],df[df.date.isin(TEST)]
P=dict(T.PARAMS, num_leaves=63); R=600      # 비교용 경량 설정

def run(feats, sub, name):
    out={}
    for tgt in ("TA14","HM14"):
        s=sub[sub[tgt].notna()]
        X=s[feats].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        m=lgb.train(P, lgb.Dataset(s[feats+["STN_CAT"]],y-lin.predict(X),
                    categorical_feature=["STN_CAT"]), R)
        out[tgt]=m.predict(te[feats+["STN_CAT"]])+lin.predict(te[feats].astype(np.float64))
    rt=np.sqrt(np.nanmean((out["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((out["HM14"]-te.HM14.to_numpy())**2))
    b=np.nanmean(out["TA14"]-te.TA14.to_numpy())
    print(f"  {name:30s} Score {rt+0.1*rh:6.3f}  TA {rt:.3f}  편향 {b:+.2f}  ({sub.date.nunique()}일)", flush=True)

print(f"{'구성':30s} {'결과'}")
print(f"{'-'*30} {'-'*56}")
run(FE, tr[tr.y>=2023], "2023~ (3년)")
run(FE, tr[tr.y>=2022], "2022~ (4년)")
run(FE, tr[tr.y>=2021], "2021~ (5년)")
run(FE, tr, "2019~ (7년)")
run(FE+["YEAR"], tr, "2019~ + 연도 피처")
run(FE+["YEAR"], tr[tr.y>=2021], "2021~ + 연도 피처")
