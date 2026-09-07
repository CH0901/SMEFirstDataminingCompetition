"""실제 채점 조건에 가깝게 재본다.

  (1) 완충 구간: 채점 주 앞뒤 15일을 학습에서 빼면? (실제로 학습은 8/9 까지)
  (2) 2026년 예측: 2026-08 을 통째로 빼고 그 기간을 맞춰본다.
      연도 피처가 2026 으로 외삽되는지 확인하는 것이기도 하다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma, train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
df["STN_CAT"]=df["STN"].astype("category")
P=dict(T.PARAMS,num_leaves=63); R=800     # 비교용 (상대 비교 목적)

def run(tr,te,name):
    out={}
    for tgt in ("TA14","HM14"):
        s=tr[tr[tgt].notna()]
        X=s[FE].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        m=lgb.train(P,lgb.Dataset(s[FE+["STN_CAT"]],y-lin.predict(X),
                    categorical_feature=["STN_CAT"]),R)
        out[tgt]=m.predict(te[FE+["STN_CAT"]])+lin.predict(te[FE].astype(np.float64))
    rt=np.sqrt(np.nanmean((out["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((out["HM14"]-te.HM14.to_numpy())**2))
    b=np.nanmean(out["TA14"]-te.TA14.to_numpy())
    print(f"  {name:36s} Score {rt+0.1*rh:6.3f}  TA {rt:.3f}  편향 {b:+.2f}  ({len(te)}행)", flush=True)

T25=pd.date_range("2025-08-24","2025-08-30")
te25=df[df.date.isin(T25)]
print(f"{'조건':36s} {'결과'}")
print(f"{'-'*36} {'-'*52}")
run(df[~df.date.isin(T25)], te25, "2025 채점주 (인접일 학습 포함)")
buf=pd.date_range(T25[0]-pd.Timedelta(days=15),T25[-1]+pd.Timedelta(days=15))
run(df[~df.date.isin(buf)], te25, "2025 채점주 (±15일 완충)")
# 2026 예측 — 실제 채점과 가장 비슷한 조건
te26=df[df.date.dt.year==2026]
if len(te26):
    run(df[df.date.dt.year<2026], te26, f"2026-08 예측 (2026 학습 제외)")
