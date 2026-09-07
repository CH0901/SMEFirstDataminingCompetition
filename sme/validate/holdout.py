"""정직한 종단 검증.

채점 기간(8/24~30)을 학습에서 완전히 제외하고 모델을 만든 뒤,
추론 노트북을 그대로 돌려 그 기간을 예측한다. 실제 채점과 같은 조건이다.
"""
import pickle, numpy as np, pandas as pd, lightgbm as lgb
from sme.core import evaluation as E
from sme.core.grid import lonlat_to_rowcol, satellite_zenith

from sme.core import kma_client
CH = list(kma_client.DEFAULT_CHANNELS)
HOLD = pd.date_range("2025-08-24","2025-08-30")

df = E.assemble(CH)
mask = ~df["date"].isin(HOLD)
tr = df[mask]
print(f"전체 {len(df):,}행 -> 학습 {len(tr):,}행 (홀드아웃 {len(df)-len(tr):,}행 제외)")
print(f"학습 날짜 {tr.date.nunique()}일")

FE = E.BASE + [c for c in df.columns if c.split("_")[0] in CH or c.startswith("SWD_")]
FE = [f for f in dict.fromkeys(FE) if f in df.columns]
from sme.model import train as T
P = T.PARAMS          # 최종 학습과 같은 설정을 쓴다
ROUNDS = T.ROUNDS
ALPHA = T.RIDGE_ALPHA
SEEDS=[42,7,2024]
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
models, linears = {}, {}
for t in ("TA14","HM14"):
    s_ = tr[tr[t].notna()]
    X = s_[FE].astype(np.float64); y = s_[t].to_numpy(np.float64)
    lin = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=ALPHA))
    lin.fit(X, y); linears[t] = lin
    models[t] = [lgb.train(dict(P, seed=sd, bagging_seed=sd, feature_fraction_seed=sd),
                           lgb.Dataset(X, y - lin.predict(X)), ROUNDS) for sd in SEEDS]

stn = pd.read_csv("data/reference/stations_scored.csv")
grid = dict(origin_latitude=38.0, central_meridian=126.0, standard_parallel1=30.0,
            standard_parallel2=60.0, pixel_size=2000.0,
            upper_left_easting=-899000.0, upper_left_northing=899000.0)
rows, cols = lonlat_to_rowcol(stn.LON.to_numpy(), stn.LAT.to_numpy(), grid)

sat_zen = satellite_zenith(stn.LON.to_numpy(), stn.LAT.to_numpy())
pickle.dump(dict(models=models, linears=linears, sat_zenith=sat_zen, features=FE, channels=CH,
                 win_stats=list(E.WIN_STATS), stations=stn,
                 pixel_rows=rows, pixel_cols=cols, grid=grid,
                 hours_utc=[0,2,4,5], hours_utc_prev=[18,21],
                 target_hour_utc=5, seed=E.SEED,
                 lightgbm_version=lgb.__version__),
            open("model_holdout.pkl","wb"), protocol=4)
print("model_holdout.pkl 저장")
