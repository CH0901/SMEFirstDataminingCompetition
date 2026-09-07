"""H1 검증 — 방향성 피처와 넓은 창이 동해안 오차를 줄이는가.

가설: 푄·해풍은 '산맥 서쪽 구름 / 동쪽 맑음' 같은 공간 배치에서 온다.
평균 하나로 뭉개면 그 정보가 사라지므로, 사분면 평균과 동서/남북 기울기를
넣고 창을 18km -> 50km 로 넓히면 최악 지점들이 개선되어야 한다.
"""
import glob, pandas as pd, numpy as np, lightgbm as lgb
from sme.core import evaluation as E
from sme.core import kma_client
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CH=list(kma_client.DEFAULT_CHANNELS)

def fold(folder, suffix, stats):
    sat=pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{folder}/*.parquet"))],
                  ignore_index=True)
    sat["utc_hour"]=sat.time_utc.dt.hour; sat["date"]=sat.time_utc.dt.normalize()
    cols=[f"{c}_{s}" for c in CH for s in stats if f"{c}_{s}" in sat.columns]
    cur=sat[sat.utc_hour<=5].pivot_table(index=["STN","date"],columns="utc_hour",values=cols)
    cur.columns=[f"{a}_h{b:02d}{suffix}" for a,b in cur.columns]
    pv=sat[sat.utc_hour>5].copy(); pv["date"]=pv["date"]+pd.Timedelta(days=1)
    prv=pv.pivot_table(index=["STN","date"],columns="utc_hour",values=cols)
    prv.columns=[f"{a}_p{b:02d}{suffix}" for a,b in prv.columns]
    return cur.join(prv,how="outer").reset_index()

base=E.assemble(CH)                                   # 기존 9x9
FE=E.BASE+[c for c in base.columns if c.split("_")[0] in CH or c.startswith("SWD_")]
FE=[f for f in dict.fromkeys(FE) if f in base.columns]

WIDE=fold("data/sat_ir105_ir123_sw038_h12","_W",("m9","sd9","gEW","gNS","gW","gE"))
df=base.merge(WIDE,on=["STN","date"],how="left")
W=[c for c in WIDE.columns if c not in ("STN","date")]
# 국소 대 광역 대비 — 이 지점이 주변보다 뜨거운가
CON=[]
for c in CH:
    for t in ["h00","h02","h04","h05","p18","p21"]:
        a,b=f"{c}_m9_{t}",f"{c}_m9_{t}_W"
        if a in df and b in df: n=f"{c}_con_{t}"; df[n]=df[a]-df[b]; CON.append(n)
df["STN_CAT"]=df["STN"].astype("category")
print(f"표본 {len(df):,}행 · 광역 피처 {len(W)}개 · 대비 {len(CON)}개")

TEST=pd.date_range("2025-08-24","2025-08-30")
buf=pd.date_range(TEST[0]-pd.Timedelta(days=15),TEST[-1]+pd.Timedelta(days=15))
tr,te=df[~df.date.isin(buf)],df[df.date.isin(TEST)].copy()
P=dict(E.PARAMS,learning_rate=0.04,num_leaves=31,min_data_in_leaf=30,lambda_l2=1.0,num_threads=4)

def run(feats,name):
    out={}
    for tgt in ("TA14","HM14"):
        s=tr[tr[tgt].notna()]; X=s[feats].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),Ridge(alpha=50.0)).fit(X,y)
        ms=[lgb.train(dict(P,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
            lgb.Dataset(s[feats+["STN_CAT"]],y-lin.predict(X),categorical_feature=["STN_CAT"]),600)
            for sd in (42,7,2024)]
        out[tgt]=np.mean([m.predict(te[feats+["STN_CAT"]]) for m in ms],axis=0)\
                 +lin.predict(te[feats].astype(np.float64))
    eTA=out["TA14"]-te.TA14.to_numpy(); eHM=out["HM14"]-te.HM14.to_numpy()
    ok=~np.isnan(eTA); rt=np.sqrt(np.nanmean(eTA**2)); rh=np.sqrt(np.nanmean(eHM**2))
    bad=te.assign(e=eTA).groupby("STN").e.apply(lambda x: np.sqrt(np.nanmean(x**2))).sort_values(ascending=False)
    print(f"  {name:26s} Score {rt+0.1*rh:6.3f}  TA {rt:.3f}  최악5지점평균 {bad.head(5).mean():.2f}", flush=True)

print(f"\n{'구성':30s} {'결과'}")
print(f"{'-'*30} {'-'*44}")
run(FE, "기존 (9x9)")
# 최소 구성 — 대상 시각(h05) 의 방향성만
g05=[c for c in W if c.endswith("_h05_W") and ("_gEW_" in c or "_gNS_" in c)]
run(FE+g05, f"+ h05 방향성만 ({len(g05)}개)")
# 대상 시각의 국소-광역 대비만
c05=[c for c in CON if c.endswith("_h05")]
run(FE+c05, f"+ h05 국소-광역대비만 ({len(c05)}개)")
run(FE+g05+c05, f"+ 둘 다 ({len(g05+c05)}개)")
# 기울기는 이미 9x9 에서도 뽑힌다 — 그것만
g9=[c for c in base.columns if ("_gEW_" in c or "_gNS_" in c) and c.endswith("_h05")]
run(FE+[c for c in g9 if c not in FE], f"+ 9x9 방향성 ({len(g9)}개)")
