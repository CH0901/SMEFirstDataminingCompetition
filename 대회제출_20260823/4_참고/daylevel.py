"""오차가 '그날 수준' 에서 오는가, '지점별 차이' 에서 오는가.

타깃 분산의 66% 가 날짜 효과였다. 그런데 지금 피처를 보면 전국 편차
ANO_ = 값 - 전국평균 만 넣고 **전국평균 자체는 안 넣었다**. 날짜 신호를
빼기만 하고 되돌려주지 않은 셈이다.

먼저 우리 오차를 두 성분으로 가른다:
  그날 전체 편향  = 그날 96지점 오차의 평균      (기단 수준을 못 맞힌 몫)
  지점별 산포     = 그 편향을 뺀 나머지          (국지 차이를 못 맞힌 몫)
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

# 전국 일평균 자체를 피처로 (= 오늘 전국이 얼마나 더운가)
nat_keys=[c[4:] for c in df.columns if c.startswith("ANO_")]
NAT=[]
for c in nat_keys:
    n=f"NAT_{c}"; df[n]=df.groupby("date")[c].transform("mean"); NAT.append(n)
# 전국 수준의 하루 변화 (어제 저녁 -> 오늘 낮): 데워지는 중인가
for a,b in [("_h05","_p21"),("_h05","_h00")]:
    for c in nat_keys:
        if c.endswith(a) and f"NAT_{c[:-len(a)]}{b}" in df:
            n=f"NATd_{c}{b}"; df[n]=df[f"NAT_{c}"]-df[f"NAT_{c[:-len(a)]}{b}"]; NAT.append(n)
NAT=[c for c in dict.fromkeys(NAT) if c in df.columns]

W=pd.date_range("2026-08-16","2026-08-22")
tr,te=df[df.date<pd.Timestamp("2026-08-16")],df[df.date.isin(W)]
P=dict(T.PARAMS,num_leaves=63); R=800
print(f"전국평균 피처 {len(NAT)}개 추가\n")

def run(F,tgt):
    s=tr[tr[tgt].notna()]
    X=s[F].astype(np.float64); y=s[tgt].to_numpy(np.float64)
    lin=make_pipeline(SimpleImputer(strategy="median"),StandardScaler(),
                      Ridge(alpha=T.RIDGE_ALPHA)).fit(X,y)
    C=F+["STN_CAT"]
    m=lgb.train(P,lgb.Dataset(s[C],y-lin.predict(X),
                categorical_feature=["STN_CAT"]),R)
    return m.predict(te[C])+lin.predict(te[F].astype(np.float64))

def report(p,name):
    t=te["TA14"].to_numpy(); e=p-t
    d=pd.DataFrame({"date":te.date.to_numpy(),"e":e}).dropna()
    bias=d.groupby("date")["e"].transform("mean")
    r_all=float(np.sqrt(np.nanmean(e**2)))
    r_day=float(np.sqrt(np.mean(d.groupby("date")["e"].mean()**2)))
    r_stn=float(np.sqrt(np.mean((d["e"]-bias)**2)))
    print(f"  {name:24s} 기온 {r_all:.3f}  = 그날수준 {r_day:.3f} + 지점별 {r_stn:.3f}")
    return r_all

base=run(feats+SP,"TA14"); report(base,"현재")
print()
for nm,ex in [("+ 전국평균 수준", NAT),
              ("+ 전국평균 (편차 없이)", [c for c in NAT])]:
    F=(feats+SP+ex) if nm=="+ 전국평균 수준" else (feats+ex)
    report(run(F,"TA14"), nm)

print("\n참고 — 그날 전국 평균기온을 완벽히 안다면?")
d=pd.DataFrame({"date":te.date.to_numpy(),"e":base-te["TA14"].to_numpy()}).dropna()
fix=d["e"]-d.groupby("date")["e"].transform("mean")
print(f"  그날 편향을 0 으로 보정      기온 {float(np.sqrt(np.mean(fix**2))):.3f}")
