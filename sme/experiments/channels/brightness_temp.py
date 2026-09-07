"""DN 을 밝기온도에 가깝게 변환하면 선형 성분이 나아지는가.

분리대기창 식 LST = A0 + A1*T13 + A2*(T13-T15) 는 '밝기온도' 에 대해 선형이다.
그런데 우리는 원시 DN 을 Ridge 에 넣고 있다. DN -> 복사휘도는 선형이지만
복사휘도 -> 밝기온도는 플랑크 함수라 오목한 비선형이다.

공식 변환계수가 KO 파일에 없으므로 정확한 BT 는 못 구한다. 대신 비선형
변환(로그·역수)을 피처로 넣어 선형 성분이 그 곡률을 흡수하게 한다.
트리는 단조변환에 불변이므로 GBM 쪽에는 영향이 없다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sme.core import evaluation as E
from sme.core import kma_client
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CH=list(kma_client.DEFAULT_CHANNELS)
df=E.assemble(CH)
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith("SWD_")]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
RAW=[c for c in FE if c.split("_")[0] in CH and any(
     c.endswith(f"_{t}") for t in ("h00","h02","h04","h05","p18","p21"))]

# 플랑크 곡률을 흉내내는 변환. DN 이 클수록 밝기온도 증가폭이 줄어든다.
LOG=[]
for c in RAW:
    n=f"L_{c}"; df[n]=np.log(np.clip(df[c],1,None)); LOG.append(n)
# 로그 공간에서 다시 계산한 분리대기창 차분
SWL=[]
for t in ("h00","h02","h04","h05","p18","p21"):
    for s in ("c","m9"):
        a,b=f"L_IR105_{s}_{t}",f"L_IR123_{s}_{t}"
        if a in df and b in df: n=f"SWDL_{s}_{t}"; df[n]=df[a]-df[b]; SWL.append(n)

df["STN_CAT"]=df["STN"].astype("category")
TEST=pd.date_range("2025-08-24","2025-08-30")
tr,te=df[~df.date.isin(TEST)],df[df.date.isin(TEST)]
P=dict(E.PARAMS,learning_rate=0.04,num_leaves=31,min_data_in_leaf=30,lambda_l2=1.0,num_threads=4)

def run(lin_f, name):
    out={}
    for tgt in ("TA14","HM14"):
        s=tr[tr[tgt].notna()]
        X=s[lin_f].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=50.0)).fit(X,y)
        # GBM 입력은 항상 원본 (단조변환 불변이라 바꿔도 무의미)
        ms=[lgb.train(dict(P,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
            lgb.Dataset(s[FE+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),600)
            for sd in (42,7,2024)]
        out[tgt]=np.mean([m.predict(te[FE+["STN_CAT"]]) for m in ms],axis=0)\
                 +lin.predict(te[lin_f].astype(np.float64))
    rt=np.sqrt(np.nanmean((out["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((out["HM14"]-te.HM14.to_numpy())**2))
    print(f"  {name:34s} Score {rt+0.1*rh:6.3f}   TA {rt:.3f}   HM {rh:.3f}", flush=True)

print(f"{'선형 성분 입력':34s} {'결과'}")
print(f"{'-'*34} {'-'*44}")
run(FE, "원시 DN (현재)")
run(FE+LOG, "+ log(DN)")
run(FE+LOG+SWL, "+ log(DN) + 로그공간 분리대기창")
run([f for f in FE if f not in RAW]+LOG+SWL, "log 로 완전 대체")
