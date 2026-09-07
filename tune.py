"""모델 하이퍼파라미터를 체계적으로 본다.

지금까지 피처만 건드렸고 학습 설정은 초반에 정한 것을 그대로 썼다.
홀드아웃 한 주(672행)로 고르면 그 주에 과적합되므로, 날짜 그룹 5-fold CV 로
판단한다 (표본 30배).
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eval14 as E, kma

CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith("SWD_")]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
df["STN_CAT"]=df["STN"].astype("category")
G=FE+["STN_CAT"]

def cv(params, rounds, alpha=50.0, n=5):
    res={}
    for tgt in ("TA14","HM14"):
        oof=np.full(len(df),np.nan)
        for tr,va in GroupKFold(n).split(df,groups=df.date):
            s=df.iloc[tr]; s=s[s[tgt].notna()]
            X=s[FE].astype(np.float64); y=s[tgt].to_numpy(np.float64)
            lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                              Ridge(alpha=alpha)).fit(X,y)
            m=lgb.train(dict(params,verbose=-1,seed=42,bagging_seed=42,
                             feature_fraction_seed=42,deterministic=True,
                             force_row_wise=True,num_threads=4),
                        lgb.Dataset(s[G], y-lin.predict(X), categorical_feature=["STN_CAT"]),
                        rounds)
            v=df.iloc[va]
            oof[va]=m.predict(v[G])+lin.predict(v[FE].astype(np.float64))
        y=df[tgt].to_numpy(); ok=~np.isnan(oof)&~np.isnan(y)
        res[tgt]=float(np.sqrt(((y[ok]-oof[ok])**2).mean()))
    return res["TA14"]+0.1*res["HM14"], res["TA14"], res["HM14"]

BASE=dict(objective="regression",metric="rmse",learning_rate=0.04,num_leaves=31,
          min_data_in_leaf=30,feature_fraction=0.8,bagging_fraction=0.8,
          bagging_freq=1,lambda_l2=1.0)
print(f"{'설정':40s} {'Score':>7s} {'TA':>7s} {'HM':>7s}")
print(f"{'-'*40} {'-'*7} {'-'*7} {'-'*7}")
# 개별로 좋았던 방향을 합친다: 용량 크게 + 천천히 오래 + 선형 강하게 규제
trials=[
    ("현재 기준선",                          BASE, 600, 50.0),
    ("leaves63 + a500",                    dict(BASE,num_leaves=63), 600, 500.0),
    ("leaves63 + lr.02/1500 + a500",       dict(BASE,num_leaves=63,learning_rate=0.02), 1500, 500.0),
    ("leaves127 + lr.02/1500 + a500",      dict(BASE,num_leaves=127,learning_rate=0.02), 1500, 500.0),
    ("leaves63 + lr.02/2500 + a500",       dict(BASE,num_leaves=63,learning_rate=0.02), 2500, 500.0),
    ("leaves63 lr.02/1500 a500 ff.6",      dict(BASE,num_leaves=63,learning_rate=0.02,feature_fraction=0.6), 1500, 500.0),
]
for name,p,r,a in trials:
    s,t,h=cv(p,r,a)
    print(f"  {name:38s} {s:7.3f} {t:7.3f} {h:7.3f}", flush=True)
