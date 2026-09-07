"""분리대기창(split-window) 채널쌍이 도움이 되는지 검증.

근거: KMA 공식 GK2A 지표면온도 알고리즘(ATBD)
    LST = A0 + A1*T13 + A2*(T13 - T15)          T13=IR105(10.35um), T15=IR123(12.37um)
  두 채널은 파장이 가까워 지표 신호는 거의 같지만 수증기 흡수량이 다르다.
  그래서 '차이' 가 대기(수증기) 효과를 지워주는 보정항이 된다.
  보조변수로 위성 천정각이 들어가는데, 경로 길이 효과라 sec(theta) 형태다.

앞서 채널 선택에서 이 둘을 떨어뜨렸던 것은 GBM 단독 구조에서 쟀기 때문이다.
트리는 T13 - T15 같은 선형 조합을 직접 표현하지 못한다. 선형 성분이 앞에
붙은 지금 구조에서는 결과가 달라질 수 있다.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

from sme.core import evaluation as E
from sme.experiments.strategy.robustness import evaluate, fit_predict

SW_CH = ["IR105", "IR123"]
BASE_CH = ["IR087", "NR016", "SW038"]


def load_folder(pattern: str, channels: list[str]) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    sat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sat["utc_hour"] = sat["time_utc"].dt.hour
    sat["date"] = sat["time_utc"].dt.normalize()
    cols = [f"{c}_{s}" for c in channels for s in E.WIN_STATS
            if f"{c}_{s}" in sat.columns]

    cur = sat[sat.utc_hour <= E.TARGET_HOUR_UTC].pivot_table(
        index=["STN", "date"], columns="utc_hour", values=cols)
    cur.columns = [f"{a}_h{b:02d}" for a, b in cur.columns]

    pv = sat[sat.utc_hour > E.TARGET_HOUR_UTC].copy()
    pv["date"] = pv["date"] + pd.Timedelta(days=1)
    prv = pv.pivot_table(index=["STN", "date"], columns="utc_hour", values=cols)
    prv.columns = [f"{a}_p{b:02d}" for a, b in prv.columns]

    out = cur.join(prv, how="outer")
    for c in cols:
        if f"{c}_h05" in out and f"{c}_h04" in out:
            out[f"{c}_d1h"] = out[f"{c}_h05"] - out[f"{c}_h04"]
        if f"{c}_h05" in out and f"{c}_p21" in out:
            out[f"{c}_dnt"] = out[f"{c}_h05"] - out[f"{c}_p21"]
    return out.reset_index()


def build() -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    base = load_folder("data/sat_days/*.parquet", BASE_CH)
    sw = load_folder("data/sat_ir105_ir123/*.parquet", SW_CH)
    if sw.empty:
        raise SystemExit("split-window 데이터가 아직 없습니다.")

    lab = E.load_labels()
    stn = pd.read_csv("data/reference/stations_scored.csv")
    df = base.merge(sw, on=["STN", "date"], how="inner")
    df = df.merge(lab, on=["STN", "date"], how="inner")
    df = df.merge(stn[["STN", "LAT", "LON", "HT"]], on="STN", how="inner")
    df["DOY"] = df.date.dt.dayofyear
    df["DOY_SIN"] = np.sin(2 * np.pi * df.DOY / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * df.DOY / 365.25)

    # 분리대기창 보정항: 같은 시각의 두 채널 차이
    SWD = []
    tags = [f"h{h:02d}" for h in (0, 2, 4, 5)] + [f"p{h:02d}" for h in (18, 21)]
    for t in tags:
        for s in ("c", "m9"):
            a, b = f"IR105_{s}_{t}", f"IR123_{s}_{t}"
            if a in df and b in df:
                n = f"SWD_{s}_{t}"
                df[n] = df[a] - df[b]
                SWD.append(n)

    # 위성 천정각의 경로 길이 효과. 물리식에 sec(theta)-1 로 들어간다.
    from sme.core.grid import satellite_zenith
    z = satellite_zenith(df.LON.to_numpy(), df.LAT.to_numpy())
    df["SEC_SATZEN"] = 1.0 / np.cos(np.radians(z)) - 1.0
    GEOM = ["SEC_SATZEN"]

    F_BASE = [c for c in df.columns if c.split("_")[0] in BASE_CH]
    F_SW = [c for c in df.columns if c.split("_")[0] in SW_CH]
    return df, F_BASE, F_SW, SWD + GEOM


if __name__ == "__main__":
    df, F_BASE, F_SW, F_EXTRA = build()
    IN = pd.date_range("2025-08-24", "2025-08-30")
    OUT = df.loc[df.date.dt.month <= 7, "date"].unique()
    tr = df[~df.date.isin(IN) & ~df.date.isin(OUT)]
    te_in, te_out = df[df.date.isin(IN)], df[df.date.isin(OUT)]
    print(f"학습 {len(tr):,}행 ({tr.date.nunique()}일) · "
          f"계절안 {len(te_in):,}행 · 계절밖 {len(te_out):,}행 "
          f"({len(pd.unique(OUT))}일)")
    print(f"피처: 기존 {len(F_BASE)} · split-window {len(F_SW)} · "
          f"보정항 {len(F_EXTRA)}\n")

    print(f"{'구성':30s} | {'계절안':>7s} | {'계절밖':>7s} {'TA편향':>8s}")
    print(f"{'-'*30} | {'-'*7} | {'-'*7} {'-'*8}")
    for name, feats in [
        ("기존 3채널",                 E.BASE + F_BASE),
        ("+ IR105·IR123",            E.BASE + F_BASE + F_SW),
        ("+ 차분·천정각",              E.BASE + F_BASE + F_SW + F_EXTRA),
        ("split-window 만",           E.BASE + F_SW + F_EXTRA),
    ]:
        ta = fit_predict(tr, feats, "TA14", True)
        hm = fit_predict(tr, feats, "HM14", True)
        a, b = evaluate(ta, hm, te_in), evaluate(ta, hm, te_out)
        print(f"  {name:28s} | {a[0]:7.3f} | {b[0]:7.3f} {b[3]:+8.2f}",
              flush=True)
