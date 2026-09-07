"""다중 시각 학습이 14시 예측을 개선하는가."""
import numpy as np, pandas as pd, lightgbm as lgb
from sme.experiments.temporal import multihour as M
from sme.core import evaluation as E
from sme.core import kma_client
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CH=list(kma_client.DEFAULT_CHANNELS)
df=M.build(CH); F=M.feature_cols(df,CH)
TEST=pd.date_range("2025-08-24","2025-08-30")
te=df[(df.date.isin(TEST))&(df.target_hour==14)]           # 평가는 14시만
P=dict(E.PARAMS,learning_rate=0.04,num_leaves=31,min_data_in_leaf=30,lambda_l2=1.0,num_threads=4)

def run(tr,name):
    out={}
    for tgt in ("TA","HM"):
        s=tr[tr[tgt].notna()]
        X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=50.0)).fit(X,y)
        ms=[lgb.train(dict(P,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
            lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),600)
            for sd in (42,7,2024)]
        out[tgt]=np.mean([m.predict(te[F+["STN_CAT"]]) for m in ms],axis=0)\
                 +lin.predict(te[F].astype(np.float64))
    eT=out["TA"]-te.TA.to_numpy(); eH=out["HM"]-te.HM.to_numpy()
    rt=np.sqrt(np.nanmean(eT**2)); rh=np.sqrt(np.nanmean(eH**2))
    print(f"  {name:30s} Score {rt+0.1*rh:6.3f}   TA {rt:.3f}  HM {rh:.3f}  (학습 {len(tr):,}행)", flush=True)

base=df[~df.date.isin(TEST)]
print(f"{'학습 구성':30s} {'결과'}")
print(f"{'-'*30} {'-'*50}")
run(base[base.target_hour==14], "14시만 (현재 방식)")
run(base[base.target_hour.isin([13,14])], "13+14시")
run(base, "09+11+13+14시 전부")
