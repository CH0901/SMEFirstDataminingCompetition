"""청천 화소 기준 분리대기창.

발견: GK2A DN 은 온도와 반대 방향이다 (상관 -0.46~-0.56). 따라서
  min9 (DN 최솟값) = 물리적으로 가장 뜨거운 화소 = 청천 지표
  max9 (DN 최댓값) = 가장 찬 화소 = 구름 꼭대기
실제로 min9 의 상관(-0.557)이 max9(-0.465)보다 강하고, 모델도 min9 을 3배 더 쓴다.

그런데 우리는 분리대기창 차분(SWD)을 c 와 m9 에서만 계산했다. KMA 공식
알고리즘은 '청천 육지 화소' 에서만 지표면온도를 산출한다. 구름이 섞인 평균으로
계산한 차분은 물리적으로 틀린 값이다. 청천 대리값인 min9 으로 계산한 SWD 가
원래 맞는 형태다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E, kma
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CH=list(kma.DEFAULT_CHANNELS)
df=E.assemble(CH)
FE=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith("SWD_")]
FE=[f for f in dict.fromkeys(FE) if f in df.columns]
TAGS=("h00","h02","h04","h05","p18","p21")

new={}
for t in TAGS:
    # 청천 기준 분리대기창
    a,b=f"IR105_min9_{t}",f"IR123_min9_{t}"
    if a in df and b in df: new[f"SWDmin_{t}"]=df[a]-df[b]
    # 구름 기준 (대조군)
    a,b=f"IR105_max9_{t}",f"IR123_max9_{t}"
    if a in df and b in df: new[f"SWDmax_{t}"]=df[a]-df[b]
    # 창 안의 청천-구름 대비 = 구름량 대리값
    a,b=f"IR105_max9_{t}",f"IR105_min9_{t}"
    if a in df and b in df: new[f"CLD_{t}"]=df[a]-df[b]
    # 하층 대리: m9 에서 표준편차만큼 청천 쪽으로 이동
    a,b=f"IR105_m9_{t}",f"IR105_sd9_{t}"
    if a in df and b in df: new[f"CLR_{t}"]=df[a]-df[b]
df=pd.concat([df,pd.DataFrame(new,index=df.index)],axis=1)
SWDMIN=[f"SWDmin_{t}" for t in TAGS if f"SWDmin_{t}" in df]
SWDMAX=[f"SWDmax_{t}" for t in TAGS if f"SWDmax_{t}" in df]
CLD=[f"CLD_{t}" for t in TAGS if f"CLD_{t}" in df]
CLR=[f"CLR_{t}" for t in TAGS if f"CLR_{t}" in df]

df["STN_CAT"]=df["STN"].astype("category")
TEST=pd.date_range("2025-08-24","2025-08-30")
tr,te=df[~df.date.isin(TEST)],df[df.date.isin(TEST)]
P=dict(E.PARAMS,learning_rate=0.04,num_leaves=31,min_data_in_leaf=30,lambda_l2=1.0,num_threads=4)

def run(F,name):
    out={}
    for tgt in ("TA14","HM14"):
        s=tr[tr[tgt].notna()]
        X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=50.0)).fit(X,y)
        ms=[lgb.train(dict(P,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
            lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),600)
            for sd in (42,7,2024)]
        out[tgt]=np.mean([m.predict(te[F+["STN_CAT"]]) for m in ms],axis=0)\
                 +lin.predict(te[F].astype(np.float64))
    rt=np.sqrt(np.nanmean((out["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((out["HM14"]-te.HM14.to_numpy())**2))
    print(f"  {name:32s} Score {rt+0.1*rh:6.3f}   TA {rt:.3f}   HM {rh:.3f}", flush=True)

print(f"{'구성':32s} {'결과'}")
print(f"{'-'*32} {'-'*44}")
run(FE, "현재 (SWD = c, m9)")
run(FE+SWDMIN, "+ SWD_min9 (청천 기준)")
run(FE+SWDMIN+CLD, "+ 구름량 대리(max-min)")
run(FE+SWDMIN+CLD+CLR, "+ 청천 보정 m9-sd9")
run(FE+SWDMIN+SWDMAX+CLD+CLR, "+ 전부")
