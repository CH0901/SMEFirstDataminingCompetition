"""예측 습도를 기온 예측의 입력으로 (2단계 적층).

타깃 분해: 14시 기온 분산의 25% 가 '날짜x지점 상호작용' 이고, 그 성분과
가장 강하게 연결된 것이 습도다(상관 -0.476). 물리적으로도 지표 에너지가
현열(기온)과 잠열(증발)로 나뉘므로, 습한 곳은 덜 뜨거워진다.

지금은 TA 와 HM 을 독립적으로 예측한다. 예측된 HM 을 TA 의 입력으로 주면
그 연결을 모델이 쓸 수 있다. 예측값은 우리 모델 출력이므로 규칙상 무관하다.
누수를 막기 위해 HM 예측은 폴드 밖(out-of-fold)으로 만든다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sme.core import evaluation as E
from sme.core import kma_client
from sme.model import train as T
CH=list(kma_client.DEFAULT_CHANNELS); df=E.assemble(CH)
feats=E.BASE+[c for c in df.columns if c.split("_")[0] in CH or c.startswith(("SWD_","SWDmin_","CLD_"))]
feats=[f for f in dict.fromkeys(feats) if f in df.columns]
ex=[c for c in df.columns if c.startswith(("ANO_","NB_"))]
FB={"TA14":feats+ex,"HM14":[f for f in feats if f!="YEAR"]}
df["STN_CAT"]=df["STN"].astype("category")
P=dict(T.PARAMS,num_leaves=63); R=800
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]

def fit_pred(F,trs,tes,tgt):
    s=trs[trs[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    m=lgb.train(P,lgb.Dataset(s[F+["STN_CAT"]],y-lin.predict(X),
                categorical_feature=["STN_CAT"]),R)
    return m.predict(tes[F+["STN_CAT"]])+lin.predict(tes[F].astype(np.float64))

def rmse(p,t,tgt): return float(np.sqrt(np.nanmean((p-t[tgt].to_numpy())**2)))

# 기준
base=rmse(fit_pred(FB["TA14"],tr,te,"TA14"),te,"TA14")
print(f"  {'기준 (독립 예측)':30s} RMSE_TA {base:.3f}")

# 1단계: HM 예측을 out-of-fold 로 만든다
FH=FB["HM14"]
tr=tr.copy(); tr["pHM"]=np.nan
for a,b in GroupKFold(5).split(tr,groups=tr.date):
    sub=tr.iloc[a]
    tr.iloc[b, tr.columns.get_loc("pHM")]=fit_pred(FH,sub,tr.iloc[b],"HM14")
te=te.copy(); te["pHM"]=fit_pred(FH,tr,te,"HM14")
print(f"  (HM 예측 생성 — 학습 OOF, 검증은 전체 학습)")

for extra,name in [(["pHM"],"+ 예측 습도"),
                   (["pHM"],"+ 예측 습도 (기준 재확인)")]:
    r=rmse(fit_pred(FB["TA14"]+extra,tr,te,"TA14"),te,"TA14")
    print(f"  {name:30s} RMSE_TA {r:.3f}   ({r-base:+.3f})", flush=True)
    break
