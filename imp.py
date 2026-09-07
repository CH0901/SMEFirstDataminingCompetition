"""무엇이 실제로 성능을 만들었나 — 그룹별 기여도.

개별 피처 중요도는 상관된 피처끼리 나눠 가져서 오해를 부른다.
그래서 '그룹을 통째로 빼면 얼마나 나빠지나' 로 잰다.
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
SP=[c for c in df.columns if c.startswith(("ANO_","NB_","DW_","DE_","DN_","DS_","dEW_","dNS_"))]
FULL=feats+SP
W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
P=dict(T.PARAMS,num_leaves=63); R=800

def run(F,tgt):
    s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    C=F+["STN_CAT"]
    m=lgb.train(P,lgb.Dataset(s[C],y-lin.predict(X),
                categorical_feature=["STN_CAT"]),R)
    p=m.predict(te[C])+lin.predict(te[F].astype(np.float64))
    return float(np.sqrt(np.nanmean((p-te[tgt].to_numpy())**2))), m

GRP={
 "위성 채널 전체":[c for c in FULL if c.split("_")[0] in CH],
 "  └ IR105 (10.4um 열적외)":[c for c in FULL if c.startswith("IR105")],
 "  └ IR123 (12.4um 열적외)":[c for c in FULL if c.startswith("IR123")],
 "  └ SW038 (3.8um 단파적외)":[c for c in FULL if c.startswith("SW038")],
 "분리대기창 보정 SWD/CLD":[c for c in FULL if c.startswith(("SWD_","SWDmin_","CLD_"))],
 "공간 피처 전체":SP,
 "  └ 전국 편차 ANO":[c for c in SP if c.startswith("ANO_")],
 "  └ 이웃 NB":[c for c in SP if c.startswith("NB_")],
 "  └ 방향별 이웃":[c for c in SP if c.startswith(("DW_","DE_","DN_","DS_","dEW_","dNS_"))],
 "위경도·고도":["LAT","LON","HT"],
 "연중일 sin/cos":["DOY_SIN","DOY_COS"],
 "연도":["YEAR"],
}
base,mdl=run(FULL,"TA14")
print(f"전체 기온 RMSE {base:.3f}\n")
print(f"{'빼면 나빠지는 정도 (기온)':34s}{'RMSE':>7s}{'악화':>8s}")
print("-"*49)
for n,g in GRP.items():
    F=[c for c in FULL if c not in set(g)]
    if len(F)==len(FULL): continue
    r,_=run(F,"TA14")
    print(f"{n:34s}{r:7.3f}{r-base:+8.3f}", flush=True)

print("\n개별 피처 상위 12개 (GBM 분할 이득)")
imp=pd.Series(mdl.feature_importance("gain"),
              index=mdl.feature_name()).sort_values(ascending=False)
for k,v in imp.head(12).items():
    print(f"   {k:32s} {v/imp.sum()*100:5.1f}%")
