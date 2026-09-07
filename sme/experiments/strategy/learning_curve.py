"""학습량-성능 곡선. 더 모을 가치가 있는지 판단한다."""
import numpy as np, pandas as pd, lightgbm as lgb
from sme.core import evaluation as E

CH=["IR087","NR016","SW038"]
HOLD = pd.date_range("2025-08-24","2025-08-30")
df = E.assemble(CH)
FE = E.BASE + [c for ch in CH for c in df.columns if c.startswith(ch+"_")]
te = df[df.date.isin(HOLD)]
pool = df[~df.date.isin(HOLD)]
days = np.array(sorted(pool.date.unique()))
P = dict(E.PARAMS, learning_rate=0.04, num_leaves=31, min_data_in_leaf=30,
         lambda_l2=1.0, bagging_seed=E.SEED, feature_fraction_seed=E.SEED, num_threads=4)

print(f"홀드아웃 {len(te)}행 (2025-08-24~30) · 학습 풀 {len(days)}일\n")
print(f"{'학습일수':>8s} {'학습행수':>9s} {'RMSE_TA':>8s} {'RMSE_HM':>8s} {'Score':>7s}")
rng = np.random.RandomState(0)
for frac in (0.25, 0.5, 0.75, 1.0):
    n = max(1, int(len(days)*frac))
    sel = set(rng.choice(days, n, replace=False)) if frac < 1 else set(days)
    tr = pool[pool.date.isin(sel)]
    out = {}
    for t in ("TA14","HM14"):
        s = tr[tr[t].notna()]
        m = lgb.train(P, lgb.Dataset(s[FE], s[t]), 600)
        p = m.predict(te[FE]); y = te[t].to_numpy()
        ok = ~np.isnan(y)
        out[t] = float(np.sqrt(((y[ok]-p[ok])**2).mean()))
    sc = out["TA14"]+0.1*out["HM14"]
    print(f"{n:8d} {len(tr):9,d} {out['TA14']:8.3f} {out['HM14']:8.3f} {sc:7.3f}", flush=True)
