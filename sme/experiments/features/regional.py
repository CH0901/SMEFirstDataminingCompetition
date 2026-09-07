"""전국 규모 위성 신호를 피처로 넣는다.

지금 각 행은 자기 지점 주변 9x9 만 본다. 그런데 기온은 그날 한반도를 덮은
기단이 크게 좌우한다. 다른 지점들의 '위성' 값(ASOS 가 아니라 위성이므로
규칙상 허용)을 모으면 그 기단 상태를 알 수 있다.

  전국 평균     그날 한반도 전체가 얼마나 더운가
  지점 - 전국    이 지점이 전국 대비 얼마나 특이한가 (일별 변동을 상쇄)
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
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
FB={"TA14":feats,"HM14":[f for f in feats if f!="YEAR"]}

# 전국 평균과 지점 편차 — 핵심 통계만 (m9, min9, SWD)
KEY=[c for c in df.columns if any(c.startswith(p) for p in
     ("IR105_m9_","IR105_min9_","IR123_m9_","SW038_m9_","SWD_m9_","SWDmin_"))]
nat=df.groupby("date")[KEY].transform("mean")
add={}
for c in KEY:
    add[f"NAT_{c}"]=nat[c]          # 그날 전국 평균
    add[f"ANO_{c}"]=df[c]-nat[c]    # 전국 대비 이 지점의 편차
df=pd.concat([df,pd.DataFrame(add,index=df.index)],axis=1)
NAT=[f"NAT_{c}" for c in KEY]; ANO=[f"ANO_{c}" for c in KEY]
df["STN_CAT"]=df["STN"].astype("category")
print(f"전국 피처 {len(NAT)}개 · 편차 피처 {len(ANO)}개")

P=dict(T.PARAMS,num_leaves=63); R=800
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]

def run(extra,name):
    r={}
    for tgt in ("TA14","HM14"):
        F=FB[tgt]+extra; s=tr[tr[tgt].notna()]
        X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        m=lgb.train(P,lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),
                    categorical_feature=["STN_CAT"]),R)
        r[tgt]=m.predict(te[F+["STN_CAT"]])+lin.predict(te[F].astype(np.float64))
    rt=np.sqrt(np.nanmean((r["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((r["HM14"]-te.HM14.to_numpy())**2))
    print(f"  {name:28s} Score {rt+0.1*rh:6.3f}  TA {rt:.3f}  HM {rh:.3f}", flush=True)

print(f"\n{'구성':28s} {'2026-08-16~22 (실전 재현)'}")
print(f"{'-'*28} {'-'*44}")
run([], "현재")
run(NAT, "+ 전국 평균")
run(ANO, "+ 전국 대비 편차")
run(NAT+ANO, "+ 둘 다")
