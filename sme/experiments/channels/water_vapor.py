"""WV073(하층 수증기) · IR133(CO2) 이 도움이 되는가.

습도가 지표의 34% 인데 진짜 수증기 채널을 한 번도 안 썼다. 분리대기창
차분이 수증기를 간접 보정할 뿐이다. IR133 은 CO2 흡수대라 대기 온도
프로파일 정보를 담는다.
"""
import glob, numpy as np, pandas as pd, lightgbm as lgb
from sme.core import evaluation as E
from sme.core import kma_client
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CH=list(kma_client.DEFAULT_CHANNELS); NEW=["WV073","IR133"]

def fold(folder, chans):
    sat=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{folder}/*.parquet"))],
                  ignore_index=True)
    sat["h"]=sat.time_utc.dt.hour; sat["date"]=sat.time_utc.dt.normalize()
    cols=[f"{c}_{s}" for c in chans for s in E.WIN_STATS if f"{c}_{s}" in sat.columns]
    cur=sat[sat.h<=5].pivot_table(index=["STN","date"],columns="h",values=cols)
    cur.columns=[f"{a}_h{b:02d}" for a,b in cur.columns]
    pv=sat[sat.h>5].copy(); pv["date"]=pv["date"]+pd.Timedelta(days=1)
    prv=pv.pivot_table(index=["STN","date"],columns="h",values=cols)
    prv.columns=[f"{a}_p{b:02d}" for a,b in prv.columns]
    return cur.join(prv,how="outer").reset_index()

base=E.assemble(CH)
FE=E.BASE+[c for c in base.columns if c.split("_")[0] in CH or c.startswith("SWD_")]
FE=[f for f in dict.fromkeys(FE) if f in base.columns]
NEWD=fold("data/sat_ir133_wv073", NEW)
df=base.merge(NEWD,on=["STN","date"],how="inner")     # 새 채널이 있는 날만
NF=[c for c in NEWD.columns if c not in ("STN","date")]
df["STN_CAT"]=df["STN"].astype("category")
print(f"표본 {len(df):,}행 · {df.date.nunique()}일 (새 채널 있는 날만) · 신규 피처 {len(NF)}개")

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
    print(f"  {name:28s} Score {rt+0.1*rh:6.3f}   TA {rt:.3f}   HM {rh:.3f}", flush=True)

print(f"\n{'구성':28s} {'결과'}")
print(f"{'-'*28} {'-'*44}")
run(FE, "현재 3채널")
run(FE+[c for c in NF if c.startswith("WV073")], "+ WV073 (수증기)")
run(FE+[c for c in NF if c.startswith("IR133")], "+ IR133 (CO2)")
run(FE+NF, "+ 둘 다")
