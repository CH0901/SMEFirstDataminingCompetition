"""연도를 피처로 넣으면 온난화 추세를 외삽할 수 있는가.

규칙 2.1③ 이 'day, month, year' 를 허용 입력으로 명시한다.
8월 말 기온이 2022년 24.8도 -> 2025년 31.4도로 급상승했는데, 지금까지
연중일(DOY)만 넣고 연도는 안 썼다. 선형 성분에 연도를 넣으면 그 추세를
2026년으로 이어 그릴 수 있다 (트리는 학습 범위 밖 외삽을 못 한다).
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
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
df["YEAR"]=df.date.dt.year.astype(float)
df["STN_CAT"]=df["STN"].astype("category")
TEST=pd.date_range("2025-08-24","2025-08-30")
tr,te=df[~df.date.isin(TEST)],df[df.date.isin(TEST)]

def run(feats, sub, name):
    out={}
    for tgt in ("TA14","HM14"):
        s=sub[sub[tgt].notna()]
        X=s[feats].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        ms=[lgb.train(dict(T.PARAMS,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
            lgb.Dataset(s[feats+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),T.ROUNDS)
            for sd in (42,7,2024)]
        out[tgt]=np.mean([m.predict(te[feats+["STN_CAT"]]) for m in ms],axis=0)\
                 +lin.predict(te[feats].astype(np.float64))
    rt=np.sqrt(np.nanmean((out["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((out["HM14"]-te.HM14.to_numpy())**2))
    b=np.nanmean(out["TA14"]-te.TA14.to_numpy())
    print(f"  {name:30s} Score {rt+0.1*rh:6.3f}  TA {rt:.3f}  편향 {b:+.2f}", flush=True)

print(f"{'구성':30s} {'결과 (2025-08-24~30 검증)'}")
print(f"{'-'*30} {'-'*46}")
run(FE, tr[tr.date.dt.year>=2023], "3년 (2023~) · 연도 없음")
run(FE, tr, "7년 (2019~) · 연도 없음")
run(FE+["YEAR"], tr, "7년 + 연도 피처")
run(FE+["YEAR"], tr[tr.date.dt.year>=2021], "5년 + 연도 피처")
