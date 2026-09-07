"""습도를 제대로 손봐본다.

점수의 37%(0.72/1.96)가 습도인데 지금까지 기온만 붙들었다. 습도 RMSE 를
7.23 -> 6.2 로 줄이면 점수 0.1 이 붙는데, 기온에서 같은 0.1 을 짜내는 것보다
훨씬 쉬울 수 있다.

지금 습도 설정은 '기온용 피처에서 연도와 공간 피처를 뺀 것' 뿐이다.
습도 자체를 위해 고른 것은 하나도 없었다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import eval14 as E, kma, train as T
CH=list(kma.DEFAULT_CHANNELS); df=E.assemble(CH)
df["STN_CAT"]=df["STN"].astype("category")
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH
              or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
CUR=[f for f in feats if f!="YEAR"]
ANO=[c for c in df.columns if c.startswith("ANO_")]
NB=[c for c in df.columns if c.startswith("NB_")]
DIR=[c for c in df.columns if c.startswith(("DW_","DE_","DN_","DS_","dEW_","dNS_"))]
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
P=dict(T.PARAMS,num_leaves=63); R=800

def run(F,alpha=T.RIDGE_ALPHA,par=P,rounds=R,logit=False):
    s=tr[tr["HM14"].notna()]
    X=s[F].astype(np.float64); y=s["HM14"].to_numpy(np.float64)
    yy=np.log(np.clip(y,1,99.5)/(100-np.clip(y,1,99.5))) if logit else y
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=alpha)).fit(X,yy)
    C=F+["STN_CAT"]
    m=lgb.train(par,lgb.Dataset(s[C],yy-lin.predict(X),
                categorical_feature=["STN_CAT"]),rounds)
    p=m.predict(te[C])+lin.predict(te[F].astype(np.float64))
    if logit: p=100/(1+np.exp(-p))
    return float(np.sqrt(np.nanmean((p-te["HM14"].to_numpy())**2)))

b=run(CUR); print(f"  {'현재 설정':34s} {b:6.3f}")
for name,F,kw in [
  ("+ 연도",                    CUR+["YEAR"], {}),
  ("+ 전국 편차 ANO",            CUR+ANO, {}),
  ("+ 이웃 NB",                 CUR+NB, {}),
  ("+ 방향별 이웃",              CUR+DIR, {}),
  ("+ 공간 피처 전부",            CUR+ANO+NB+DIR, {}),
  ("로짓 변환 (0~100 경계 반영)",  CUR, dict(logit=True)),
  ("리지 약하게 (alpha 50)",      CUR, dict(alpha=50.0)),
  ("리지 강하게 (alpha 2000)",    CUR, dict(alpha=2000.0)),
  ("잎 127 · 1500라운드",        CUR, dict(par=T.PARAMS,rounds=T.ROUNDS)),
]:
    r=run(F,**kw)
    print(f"  {name:34s} {r:6.3f}  ({r-b:+.3f})  점수기여 {0.1*(r-b):+.3f}",
          flush=True)
