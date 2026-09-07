"""피처 조립. 그룹 단위로 켜고 끌 수 있게 만든다.

규정("천리안2A호 기본관측 자료만")의 해석이 확정되기 전이라, 나중에 특정
그룹을 통째로 빼야 할 수 있다. 그래서 피처를 만들 때부터 그룹으로 나눠
이름표를 붙여둔다. 빼는 건 GROUPS 에서 한 줄 지우는 것으로 끝난다.

그룹 등급:
  safe      위성 기본관측 + 순수 계산. 규정 논란 없음
  meta      지점 정적 정보(고도 등). 가이드 p.13 이 명시적으로 권장
  derived   과거 ASOS 로 만든 기후값. 회색지대 — 운영진 확인 필요
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 그룹명 -> 등급. 빼려면 여기서 지우면 그 그룹 컬럼이 통째로 빠진다.
GROUPS: dict[str, str] = {
    "sat_raw":   "safe",     # 채널별 창 통계 (IR105_c, IR105_m9, ...)
    "sat_diff":  "safe",     # 채널 조합 (split-window 등)
    "geom":      "safe",     # 태양/위성 기하
    "calendar":  "safe",     # 연중일 등 달력
    "station":   "meta",     # 고도, 위경도
    "climo":     "derived",  # 지점x연중일 기후 평년값
}


def _tag(cols: list[str], group: str) -> dict[str, str]:
    return {c: group for c in cols}


# ---------------------------------------------------------------- 개별 그룹

def add_sat_diff(df: pd.DataFrame) -> list[str]:
    """채널 조합. 단일 채널보다 물리적 의미가 뚜렷하다."""
    made = []
    pairs = [
        # 하층 수증기량. 두 창채널의 흡수 차이에서 나온다
        ("IR105", "IR112", "SPLIT_WIN"),
        # 상층 대 중층 습도 -> 대기 안정도
        ("WV063", "WV073", "WV_UPMID"),
        # 구름 상(얼음/물) 구분
        ("NR016", "IR105", "CLOUD_PHASE"),
        # 주간 하층운 탐지에 쓰이는 고전 조합
        ("SW038", "IR105", "SW_IR"),
    ]
    for a, b, name in pairs:
        for stat in ("c", "m9"):
            ca, cb = f"{a}_{stat}", f"{b}_{stat}"
            if ca in df.columns and cb in df.columns:
                col = f"{name}_{stat}"
                df[col] = df[ca] - df[cb]
                made.append(col)
    return made


def add_calendar(df: pd.DataFrame, time_col: str = "time_utc") -> list[str]:
    """달력 피처. 연중일은 원형이라 sin/cos 로 넣어야 12/31 과 1/1 이 붙는다."""
    t = pd.to_datetime(df[time_col])
    doy = t.dt.dayofyear
    df["DOY_SIN"] = np.sin(2 * np.pi * doy / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * doy / 365.25)
    df["HOUR_UTC"] = t.dt.hour
    return ["DOY_SIN", "DOY_COS", "HOUR_UTC"]


def add_station(df: pd.DataFrame, stn: pd.DataFrame) -> list[str]:
    """지점 정적 정보. 고도는 기온의 최대 설명변수다."""
    cols = [c for c in ("HT", "LAT", "LON") if c in stn.columns]
    df_out = df.merge(stn[["STN"] + cols], on="STN", how="left")
    df[cols] = df_out[cols].to_numpy()
    return cols


def build_climatology(asos_daily: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    """지점 x 연중일 기후 평년값. 예측의 출발점으로 쓴다.

    창(window)으로 앞뒤 며칠을 함께 평균해 표본을 늘리고 매끄럽게 만든다.
    반환: STN, DOY, TA_CLIMO, HM_CLIMO
    """
    d = asos_daily.copy()
    d["DOY"] = pd.to_datetime(d["date"]).dt.dayofyear
    recs = []
    for doy in range(1, 367):
        # 연말/연초를 감싸도록 원형 거리로 고른다
        dist = np.minimum(np.abs(d["DOY"] - doy), 365 - np.abs(d["DOY"] - doy))
        sel = d[dist <= window]
        if sel.empty:
            continue
        g = sel.groupby("STN")[["TA", "HM"]].mean()
        g["DOY"] = doy
        recs.append(g.reset_index())
    out = pd.concat(recs, ignore_index=True)
    return out.rename(columns={"TA": "TA_CLIMO", "HM": "HM_CLIMO"})


def add_climo(df: pd.DataFrame, climo: pd.DataFrame,
              time_col: str = "time_utc") -> list[str]:
    doy = pd.to_datetime(df[time_col]).dt.dayofyear
    key = pd.DataFrame({"STN": df["STN"].to_numpy(), "DOY": doy.to_numpy()})
    m = key.merge(climo, on=["STN", "DOY"], how="left")
    df["TA_CLIMO"] = m["TA_CLIMO"].to_numpy()
    df["HM_CLIMO"] = m["HM_CLIMO"].to_numpy()
    return ["TA_CLIMO", "HM_CLIMO"]


# ---------------------------------------------------------------- 조립

def build(sat: pd.DataFrame, stn: pd.DataFrame, *,
          climo: pd.DataFrame | None = None,
          groups: dict[str, str] | None = None) -> tuple[pd.DataFrame, dict]:
    """피처 테이블과 {컬럼: 그룹} 지도를 만든다.

    groups 를 주면 그 그룹만 포함한다 (기본은 GROUPS 전체).
    """
    groups = GROUPS if groups is None else groups
    df = sat.copy()
    tags: dict[str, str] = {}

    if "sat_raw" in groups:
        raw = [c for c in df.columns
               if c.split("_")[0] in
               ("VI004", "VI005", "VI006", "VI008", "NR013", "NR016", "SW038",
                "WV063", "WV069", "WV073", "IR087", "IR096", "IR105", "IR112",
                "IR123", "IR133")]
        tags |= _tag(raw, "sat_raw")
    if "sat_diff" in groups:
        tags |= _tag(add_sat_diff(df), "sat_diff")
    if "geom" in groups:
        g = [c for c in ("SAT_ZEN", "SOL_ZEN", "SOL_AZ", "SOL_COS")
             if c in df.columns]
        tags |= _tag(g, "geom")
    if "calendar" in groups:
        tags |= _tag(add_calendar(df), "calendar")
    if "station" in groups:
        tags |= _tag(add_station(df, stn), "station")
    if "climo" in groups and climo is not None:
        tags |= _tag(add_climo(df, climo), "climo")

    keep = ["time_utc", "STN"] + [c for c in tags]
    return df[keep], tags


def drop_group(tags: dict[str, str], group: str) -> list[str]:
    """특정 그룹을 뺀 컬럼 목록. '나중에 이상한 거 빼기' 용."""
    return [c for c, g in tags.items() if g != group]
