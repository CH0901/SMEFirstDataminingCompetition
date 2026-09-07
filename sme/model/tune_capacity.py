"""용량을 더 키우면 어디서 꺾이는가 + 청천 분리대기창 병합 효과."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sme.core import evaluation as E
from sme.core import kma_client

CH=list(kma_client.DEFAULT_CHANNELS); df=E.assemble(CH)
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith("SWD_")]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
TAGS=("h00","h02","h04","h05","p18","p21")
# 청천 기준 분리대기창 + 구름량 대리 (앞선 실험에서 습도 2% 개선)
extra={}
for t in TAGS:
    a,b=f"IR105_min9_{t}",f"IR123_min9_{t}"
    if a in df and b in df: extra[f"SWDmin_{t}"]=df[a]-df[b]
    a,b=f"IR105_max9_{t}",f"IR105_min9_{t}"
    if a in df and b in df: extra[f"CLD_{t}"]=df[a]-df[b]
df=pd.concat([df,pd.DataFrame(extra,index=df.index)],axis=1)
CS=list(extra)
df["STN_CAT"]=df["STN"].astype("category")

def cv(feats, params, rounds, alpha, n=5):
    G=feats+["STN_CAT"]; res={}
    for tgt in ("TA14","HM14"):
        oof=np.full(len(df),np.nan)
        for tr,va in GroupKFold(n).split(df,groups=df.date):
            s=df.iloc[tr]; s=s[s[tgt].notna()]
            X=s[feats].astype(np.float64); y=s[tgt].to_numpy(np.float64)
            lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                              Ridge(alpha=alpha)).fit(X,y)
            m=lgb.train(dict(params,verbose=-1,seed=42,bagging_seed=42,
                             feature_fraction_seed=42,deterministic=True,
                             force_row_wise=True,num_threads=4),
                        lgb.Dataset(s[G],y-lin.predict(X),categorical_feature=["STN_CAT"]),rounds)
            v=df.iloc[va]; oof[va]=m.predict(v[G])+lin.predict(v[feats].astype(np.float64))
        y=df[tgt].to_numpy(); ok=~np.isnan(oof)&~np.isnan(y)
        res[tgt]=float(np.sqrt(((y[ok]-oof[ok])**2).mean()))
    return res["TA14"]+0.1*res["HM14"], res["TA14"], res["HM14"]

B=dict(objective="regression",metric="rmse",learning_rate=0.02,num_leaves=127,
       min_data_in_leaf=30,feature_fraction=0.8,bagging_fraction=0.8,
       bagging_freq=1,lambda_l2=1.0)
print(f"{'설정':44s} {'Score':>7s} {'TA':>7s} {'HM':>7s}")
print(f"{'-'*44} {'-'*7} {'-'*7} {'-'*7}")
for name,f,p,r,a in [
    ("leaves127 (재확인)",              FE, B, 1500, 500.0),
    ("leaves255",                      FE, dict(B,num_leaves=255), 1500, 500.0),
    ("leaves127 + min_leaf 15",        FE, dict(B,min_data_in_leaf=15), 1500, 500.0),
    ("leaves127 + 청천SWD",             FE+CS, B, 1500, 500.0),
    ("leaves255 + 청천SWD",             FE+CS, dict(B,num_leaves=255), 1500, 500.0),
]:
    s,t,h=cv(f,p,r,a)
    print(f"  {name:42s} {s:7.3f} {t:7.3f} {h:7.3f}", flush=True)
