"""홀드아웃 한 주가 아니라 193일 전체 CV 로 판단한다 (표본 25배)."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sme.core import evaluation as E

CH=["IR087","NR016","SW038"]
df = E.assemble(CH)
BASE=E.BASE; SATF=[c for c in df.columns if c.split("_")[0] in CH]
TAGS=[f"h{h:02d}" for h in (0,2,4,5)]+[f"p{h:02d}" for h in (18,21)]
DIFF=[]
for a,b in [("IR087","SW038"),("IR087","NR016"),("SW038","NR016")]:
    for t in TAGS:
        for s in ("c","m9"):
            ca,cb=f"{a}_{s}_{t}",f"{b}_{s}_{t}"
            if ca in df and cb in df:
                n=f"D_{a}_{b}_{s}_{t}"; df[n]=df[ca]-df[cb]; DIFF.append(n)

P=dict(E.PARAMS, learning_rate=0.04, num_leaves=31, min_data_in_leaf=30,
       lambda_l2=1.0, num_threads=4)
def cv(feats, tgt):
    oof=np.full(len(df),np.nan)
    for tr,va in GroupKFold(5).split(df, groups=df.date):
        s=df.iloc[tr]; s=s[s[tgt].notna()]
        m=lgb.train(P, lgb.Dataset(s[feats], s[tgt]), 600)
        oof[va]=m.predict(df[feats].iloc[va])
    y=df[tgt].to_numpy(); ok=~np.isnan(oof)&~np.isnan(y)
    return float(np.sqrt(((y[ok]-oof[ok])**2).mean()))

print(f"표본 {len(df):,}행 · {df.date.nunique()}일 · 5-fold 날짜 그룹 CV\n")
print(f"{'구성':24s} {'RMSE_TA':>8s} {'RMSE_HM':>8s} {'Score':>7s}")
for name,f in [("현재",BASE+SATF), ("+ 채널간 차분",BASE+SATF+DIFF)]:
    ta,hm=cv(f,"TA14"),cv(f,"HM14")
    print(f"  {name:22s} {ta:8.3f} {hm:8.3f} {ta+0.1*hm:7.3f}", flush=True)
