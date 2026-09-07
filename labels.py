"""ASOS 시간자료 -> 일 단위 레이블.

제출 ID 가 'YYYYMMDD_STN' 이라 지점당 하루 1개 값인데, 그게 일평균인지
일최고인지 특정 시각 값인지 대회 페이지에 확인이 필요하다. 확정 전까지
후보를 전부 만들어두고 나중에 골라 쓴다.

시간대: ASOS TM 은 KST 이므로 KST 날짜로 묶는다.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("data")

# 하루 몇 시각 이상 관측돼야 그 날을 유효로 볼지. 관측이 몇 개뿐인 날의
# 평균은 실제 일평균과 크게 어긋나므로 걸러낸다.
MIN_HOURS = 20


def build_daily(pattern: str = "data/asos_20??0601_*.parquet") -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"ASOS 파일이 없습니다: {pattern}")
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["date"] = pd.to_datetime(d["TM"]).dt.normalize()

    g = d.groupby(["STN", "date"])
    out = g.agg(
        n_hours=("TA", "size"),
        TA_MEAN=("TA", "mean"), TA_MAX=("TA", "max"), TA_MIN=("TA", "min"),
        HM_MEAN=("HM", "mean"), HM_MAX=("HM", "max"), HM_MIN=("HM", "min"),
    ).reset_index()

    # 특정 시각 값도 후보로 남긴다 (09시는 종관 기준시각)
    for hh in (9, 15):
        sub = d[pd.to_datetime(d["TM"]).dt.hour == hh]
        sub = sub.groupby(["STN", "date"])[["TA", "HM"]].mean().reset_index()
        sub = sub.rename(columns={"TA": f"TA_H{hh:02d}", "HM": f"HM_H{hh:02d}"})
        out = out.merge(sub, on=["STN", "date"], how="left")

    before = len(out)
    out = out[out["n_hours"] >= MIN_HOURS].reset_index(drop=True)
    print(f"[labels] {before:,} -> {len(out):,}일 "
          f"(관측 {MIN_HOURS}시각 미만 {before-len(out):,}일 제외)")
    return out


def build_climatology(daily: pd.DataFrame, window: int = 15,
                      cols: tuple[str, ...] = ("TA_MEAN", "HM_MEAN")
                      ) -> pd.DataFrame:
    """지점 x 연중일 평년값. 앞뒤 window 일을 함께 평균해 매끄럽게 만든다."""
    d = daily.copy()
    d["DOY"] = pd.to_datetime(d["date"]).dt.dayofyear
    doys = sorted(d["DOY"].unique())
    recs = []
    for doy in doys:
        gap = np.abs(d["DOY"] - doy)
        sel = d[np.minimum(gap, 365 - gap) <= window]
        g = sel.groupby("STN")[list(cols)].mean()
        g.columns = [f"{c.split('_')[0]}_CLIMO" for c in cols]
        recs.append(g.reset_index().assign(DOY=doy))
    return pd.concat(recs, ignore_index=True)


if __name__ == "__main__":
    daily = build_daily()
    p = OUT / "labels_daily.parquet"
    daily.to_parquet(p, index=False)
    print(f"[labels] {len(daily):,}행 · 지점 {daily.STN.nunique()}개 -> {p}")

    climo = build_climatology(daily)
    pc = OUT / "climatology.parquet"
    climo.to_parquet(pc, index=False)
    print(f"[labels] 평년값 {len(climo):,}행 -> {pc}")

    print("\n타깃 후보 분포:")
    cands = [c for c in daily.columns if c.startswith(("TA_", "HM_"))]
    print(daily[cands].describe().loc[["mean", "std", "min", "max"]].round(1)
          .to_string())
