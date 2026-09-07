"""최종 설정 그대로, 실전과 같은 조건에서 방향별 이웃의 효과를 확인한다.

학습은 2026-08-15 까지만, 예측은 2026-08-16~22. 실제 채점(8/24~30)과 같은
'미래를 전혀 보지 않은' 구조다.
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
ANO=[c for c in df.columns if c.startswith(("ANO_","NB_"))]
DIR=[c for c in df.columns if c.startswith(("DW_","DE_","DN_","DS_","dEW_","dNS_"))]
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
print(f"학습 {len(tr):,}행 / {tr.date.nunique()}일 · 검증 {len(te):,}행")
print(f"피처: 기본 {len(feats)} · 편차·이웃 {len(ANO)} · 방향 {len(DIR)}\n")

def run(F,tgt):
    s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    r=y-lin.predict(X); C=F+["STN_CAT"]
    ps=[lgb.train(dict(T.PARAMS,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
                  lgb.Dataset(s[C],r,categorical_feature=["STN_CAT"]),
                  T.ROUNDS).predict(te[C]) for sd in T.SEEDS]
    p=np.mean(ps,0)+lin.predict(te[F].astype(np.float64))
    return float(np.sqrt(np.nanmean((p-te[tgt].to_numpy())**2)))

for name,ta_f in [("현재 (편차·이웃)", feats+ANO),
                  ("+ 방향별 이웃",     feats+ANO+DIR)]:
    ta=run(ta_f,"TA14")
    hm=run([f for f in ta_f if f!="YEAR"],"HM14")
    print(f"  {name:20s} 점수 {ta+0.1*hm:.3f}   기온 {ta:.3f}  습도 {hm:.3f}",
          flush=True)
