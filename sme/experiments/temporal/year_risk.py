"""연도 피처가 2026 예측에서 해로운가."""
import numpy as np, pandas as pd, lightgbm as lgb
from sme.core import evaluation as E
from sme.core import kma_client
from sme.model import train as T
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
CH=list(kma_client.DEFAULT_CHANNELS); df=E.assemble(CH)
ALL=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
ALL=[f for f in dict.fromkeys(ALL) if f in df.columns]
NOYR=[f for f in ALL if f!="YEAR"]
df["STN_CAT"]=df["STN"].astype("category")
P=dict(T.PARAMS,num_leaves=63); R=800

def run(F,tr,te,name):
    out={}
    for tgt in ("TA14","HM14"):
        s=tr[tr[tgt].notna()]
        X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
        lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                          Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
        m=lgb.train(P,lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),
                    categorical_feature=["STN_CAT"]),R)
        out[tgt]=m.predict(te[F+["STN_CAT"]])+lin.predict(te[F].astype(np.float64))
    rt=np.sqrt(np.nanmean((out["TA14"]-te.TA14.to_numpy())**2))
    rh=np.sqrt(np.nanmean((out["HM14"]-te.HM14.to_numpy())**2))
    b=np.nanmean(out["TA14"]-te.TA14.to_numpy())
    print(f"  {name:38s} Score {rt+0.1*rh:6.3f}  TA {rt:.3f}  편향 {b:+.2f}", flush=True)

W=pd.date_range("2026-08-16","2026-08-22"); te=df[df.date.isin(W)]
tr=df[df.date < pd.Timestamp("2026-08-16")]
print(f"실전 재현: 2026-08-16~22 예측 ({len(te)}행)\n")
print(f"{'구성':38s} {'결과'}")
print(f"{'-'*38} {'-'*44}")
run(ALL, tr, te, "연도 피처 있음 (현재)")
run(NOYR, tr, te, "연도 피처 제거")
run(NOYR, tr[tr.date.dt.month>=8], te, "연도 제거 + 8월 이후만 학습")
run(ALL, tr[tr.date.dt.month>=8], te, "연도 있음 + 8월 이후만 학습")
