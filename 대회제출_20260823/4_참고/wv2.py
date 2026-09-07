"""수증기 채널(WV073)·CO2 채널(IR133)이 지금 구조에서 도움이 되는가.

앞선 시험은 공간 피처가 없던 시절 구조였다. 지금은 전국 편차·이웃이
성능의 대부분을 차지하므로, 그 위에서 다시 재야 한다.

WV073 은 하층 수증기 흡수대라 습도의 직접 대리값이고, IR133 은 CO2
흡수대라 대기 온도 프로파일을 담는다. 습도가 점수의 37% 인데 지금까지
진짜 수증기 채널을 한 번도 안 썼다.
"""
import glob, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eval14 as E, kma, train as T

CH=list(kma.DEFAULT_CHANNELS); NEW=["WV073","IR133"]
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
feats=E.BASE+[c for c in base.columns if c.split("_")[0] in CH
              or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in base.columns]
SP=[c for c in base.columns if c.startswith(("ANO_","NB_","DW_","DE_","DN_","DS_","dEW_","dNS_"))]
NEWD=fold("data/sat_ir133_wv073", NEW)
df=base.merge(NEWD,on=["STN","date"],how="inner")
NF=[c for c in NEWD.columns if c not in ("STN","date")]
# 새 채널로도 전국 편차를 만든다 (지금 구조에서 가장 잘 듣는 형태)
for c in [x for x in NF if x.endswith("_h05")]:
    m=df.groupby("date")[c].transform("mean")
    df[f"ANO_{c}"]=df[c]-m
ANOW=[f"ANO_{c}" for c in NF if c.endswith("_h05")]
df["STN_CAT"]=df["STN"].astype("category")
TEST=pd.date_range("2025-08-24","2025-08-30")
tr,te=df[df.date<TEST[0]],df[df.date.isin(TEST)]
print(f"표본 {len(df):,}행 · {df.date.nunique()}일 · 학습 {len(tr):,} / 검증 {len(te):,}")
print(f"신규 피처 {len(NF)}개 (+ 편차 {len(ANOW)}개)\n")
P=dict(T.PARAMS,num_leaves=63); R=800

def run(extra,name):
    o={}
    for tgt,f0 in (("TA14",feats+SP),("HM14",[f for f in feats if f!="YEAR"])):
        F=f0+extra
        s=tr[tr[tgt].notna()]
        X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        C=F+["STN_CAT"]
        m=lgb.train(P,lgb.Dataset(s[C],y-lin.predict(X),
                    categorical_feature=["STN_CAT"]),R)
        o[tgt]=m.predict(te[C])+lin.predict(te[F].astype(np.float64))
    rt=float(np.sqrt(np.nanmean((o["TA14"]-te.TA14.to_numpy())**2)))
    rh=float(np.sqrt(np.nanmean((o["HM14"]-te.HM14.to_numpy())**2)))
    print(f"  {name:26s} 점수 {rt+0.1*rh:6.3f}   기온 {rt:.3f}   습도 {rh:.3f}",
          flush=True)

run([], "현재 3채널")
run([c for c in NF if c.startswith("WV073")], "+ WV073 (수증기)")
run([c for c in NF if c.startswith("IR133")], "+ IR133 (CO2)")
run(NF, "+ 둘 다")
run(NF+ANOW, "+ 둘 다 · 전국 편차까지")
