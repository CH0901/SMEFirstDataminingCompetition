"""문헌 기반 추가 피처 시험 (추가 다운로드 없이 가능한 것만).

  A. 채널 간 차분 (split-window 계열)
     IR087-SW038 은 하층운·안개 판별에 쓰이는 고전 조합이다.
  B. 태양 기하 (날짜·좌표에서 계산 -> 규칙 2.1 ③ 허용)
"""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
import eval14 as E
from geo import solar_position, satellite_zenith

CH=["IR087","NR016","SW038"]
HOLD = pd.date_range("2025-08-24","2025-08-30")
df = E.assemble(CH)
BASE = E.BASE
SATF = [c for c in df.columns if c.split("_")[0] in CH]

# A. 채널 간 차분: 같은 시각·같은 통계끼리
TAGS = [f"h{h:02d}" for h in (0,2,4,5)] + [f"p{h:02d}" for h in (18,21)]
DIFF = []
for a, b in [("IR087","SW038"), ("IR087","NR016"), ("SW038","NR016")]:
    for t in TAGS:
        for s in ("c","m9"):
            ca, cb = f"{a}_{s}_{t}", f"{b}_{s}_{t}"
            if ca in df and cb in df:
                n = f"D_{a}_{b}_{s}_{t}"; df[n] = df[ca]-df[cb]; DIFF.append(n)

# B. 태양·위성 기하 (대상 시각 14 KST = 05 UTC 기준)
lonlat = df[["LON","LAT"]].to_numpy()
utc = pd.to_datetime(df["date"]) + pd.Timedelta(hours=5)
z, a = solar_position(utc, lonlat[:,0], lonlat[:,1])
df["SOL_ZEN"] = z; df["SOL_AZ"] = a; df["SOL_COS"] = np.cos(np.radians(z))
df["SAT_ZEN"] = satellite_zenith(lonlat[:,0], lonlat[:,1])
GEOM = ["SOL_ZEN","SOL_AZ","SOL_COS","SAT_ZEN"]

te, tr = df[df.date.isin(HOLD)], df[~df.date.isin(HOLD)]
P = dict(E.PARAMS, learning_rate=0.04, num_leaves=31, min_data_in_leaf=30,
         lambda_l2=1.0, num_threads=4)
SEEDS=[42,7,2024]
def run(feats):
    out={}
    for t in ("TA14","HM14"):
        s = tr[tr[t].notna()]
        ps=[lgb.train(dict(P,seed=sd,bagging_seed=sd,feature_fraction_seed=sd),
                      lgb.Dataset(s[feats], s[t]), 600).predict(te[feats]) for sd in SEEDS]
        p=np.mean(ps,axis=0); y=te[t].to_numpy(); ok=~np.isnan(y)
        out[t]=float(np.sqrt(((y[ok]-p[ok])**2).mean()))
    return out["TA14"], out["HM14"], out["TA14"]+0.1*out["HM14"]

print(f"{'구성':30s} {'RMSE_TA':>8s} {'RMSE_HM':>8s} {'Score':>7s}  피처수")
for name, f in [("현재 (위성+지점+연중일)", BASE+SATF),
                ("+ 채널간 차분",          BASE+SATF+DIFF),
                ("+ 태양·위성 기하",       BASE+SATF+GEOM),
                ("+ 둘 다",               BASE+SATF+DIFF+GEOM)]:
    ta,hm,sc = run(f)
    print(f"  {name:28s} {ta:8.3f} {hm:8.3f} {sc:7.3f}  {len(f):4d}", flush=True)
