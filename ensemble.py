"""시드 앙상블이 도움이 되는지 확인.

데이터가 포화됐으니 남은 개선 여지는 모델 쪽뿐이다. 시드만 바꿔 여러 개를
평균내는 것은 비용이 거의 없고 결정론적이라 규칙에도 어긋나지 않는다.
"""
import numpy as np, pandas as pd, lightgbm as lgb
import eval14 as E

CH=["IR087","NR016","SW038"]
HOLD = pd.date_range("2025-08-24","2025-08-30")
df = E.assemble(CH)
FE = E.BASE + [c for ch in CH for c in df.columns if c.startswith(ch+"_")]
te, tr = df[df.date.isin(HOLD)], df[~df.date.isin(HOLD)]

def run(seeds, leaves, rounds, lr):
    out={}
    for t in ("TA14","HM14"):
        s = tr[tr[t].notna()]
        ps=[]
        for sd in seeds:
            P = dict(E.PARAMS, learning_rate=lr, num_leaves=leaves,
                     min_data_in_leaf=30, lambda_l2=1.0, seed=sd,
                     bagging_seed=sd, feature_fraction_seed=sd, num_threads=4)
            ps.append(lgb.train(P, lgb.Dataset(s[FE], s[t]), rounds).predict(te[FE]))
        p=np.mean(ps,axis=0); y=te[t].to_numpy(); ok=~np.isnan(y)
        out[t]=float(np.sqrt(((y[ok]-p[ok])**2).mean()))
    return out["TA14"], out["HM14"], out["TA14"]+0.1*out["HM14"]

print(f"{'구성':34s} {'RMSE_TA':>8s} {'RMSE_HM':>8s} {'Score':>7s}")
for name, sd, lv, rd, lr in [
    ("현재 (시드1, leaves31, 600)", [42], 31, 600, 0.04),
    ("시드 3개 평균",               [42,7,2024], 31, 600, 0.04),
    ("시드 5개 평균",               [42,7,2024,13,99], 31, 600, 0.04),
    ("시드3 + leaves63",           [42,7,2024], 63, 600, 0.04),
    ("시드3 + lr0.02, 1200라운드",   [42,7,2024], 31, 1200, 0.02),
]:
    ta,hm,sc = run(sd,lv,rd,lr)
    print(f"  {name:32s} {ta:8.3f} {hm:8.3f} {sc:7.3f}", flush=True)
