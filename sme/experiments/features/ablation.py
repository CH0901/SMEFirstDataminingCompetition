"""피처 그룹별 기여도 + 평년값 금지 시나리오 대비."""
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sme.core import evaluation as E

CH = ["IR087","NR016","SW038"]
lab = E.load_labels(); stn = pd.read_csv("data/reference/stations_scored.csv")
geo = pd.read_csv("data/reference/geofeat.csv")
sat = E.causal_satellite(CH)
df = sat.merge(lab, on=["STN","date"]).merge(stn[["STN","LAT","LON","HT"]], on="STN")
df = df.merge(geo, on="STN", how="left")
df["DOY"] = df.date.dt.dayofyear
df["DOY_SIN"] = np.sin(2*np.pi*df.DOY/365.25); df["DOY_COS"] = np.cos(2*np.pi*df.DOY/365.25)

SATF = [c for c in df.columns if c.split("_")[0] in CH]
META = ["LAT","LON","HT","DOY_SIN","DOY_COS"]
GEOF = ["COAST_KM","IS_COASTAL","SEA_FRAC_18","SEA_FRAC_50","LOG_COAST"]
CLIM = ["TA_CLIMO","HM_CLIMO"]

def cv(feats, tgt, use_climo):
    oof = np.full(len(df), np.nan)
    for tr, va in GroupKFold(4).split(df, groups=df.date):
        dd = df
        if use_climo:                       # 누수 없이 학습 폴드만으로 평년값
            climo = E.build_climatology(lab[lab.date.isin(df.date.iloc[tr].unique())])
            dd = df.merge(climo, on=["STN","DOY"], how="left")
            for c in CLIM: dd[c] = dd[c].fillna(dd[c].median())
        m = lgb.train(E.PARAMS, lgb.Dataset(dd[feats].iloc[tr], dd[tgt].iloc[tr]), 500)
        oof[va] = m.predict(dd[feats].iloc[va])
    ok = ~np.isnan(oof) & df[tgt].notna().to_numpy()
    return float(np.sqrt(((df[tgt].to_numpy()[ok]-oof[ok])**2).mean()))

print(f"표본 {len(df):,}행 · {df.date.nunique()}일\n")
print(f"{'구성':38s} {'RMSE_TA':>8s} {'RMSE_HM':>8s} {'Score':>7s}")
cases = [
    ("위성+메타 (현재)",              META+SATF,            False),
    ("위성+메타+지리",                META+GEOF+SATF,       False),
    ("위성+메타+평년값",              META+SATF+CLIM,       True),
    ("위성+메타+지리+평년값",          META+GEOF+SATF+CLIM,  True),
]
for name, f, uc in cases:
    ta, hm = cv(f,"TA14",uc), cv(f,"HM14",uc)
    print(f"  {name:36s} {ta:8.3f} {hm:8.3f} {ta+0.1*hm:7.3f}", flush=True)
