"""정적 지리 피처.

규칙이 명시적으로 허용하는 범위:
  "정적인 지리 정보(위경도, 고도, 해안선까지의 거리, 육지/해양 여부, 지형 등)는
   추가 입력으로 허용됩니다."

시간에 따라 변하지 않으므로 추론 때 계산할 필요도 없다. 학습 때 한 번
구해 모델 번들에 실어두면 된다.

습도에 특히 중요하다. 해안은 바다에서 수증기가 계속 공급되고 기온
일교차도 작지만, 내륙 분지는 정반대다. 위경도만으로는 이 구분이 안 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from global_land_mask import globe

# 한반도를 덮는 격자 해상도(도). 0.02도 ~ 2km 로 위성 화소와 비슷하다.
GRID_STEP = 0.02
MAX_SEARCH_DEG = 3.0   # 해안 탐색 반경 (약 330km)


def _coast_distance(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """각 지점에서 가장 가까운 바다까지의 거리(km).

    지점 주변 격자를 훑어 바다 칸을 찾고 최소 거리를 잰다. 우리나라는
    삼면이 바다라 대부분 100km 안에서 찾아진다.
    """
    out = np.full(len(lats), np.nan)
    for i, (la, lo) in enumerate(zip(lats, lons)):
        step = GRID_STEP
        for radius in (0.5, 1.0, 2.0, MAX_SEARCH_DEG):
            gl = np.arange(la - radius, la + radius + step, step)
            go = np.arange(lo - radius, lo + radius + step, step)
            LA, LO = np.meshgrid(gl, go, indexing="ij")
            sea = ~globe.is_land(LA, LO)
            if not sea.any():
                continue
            # 위도 1도 = 111km, 경도 1도 = 111km * cos(위도)
            dy = (LA[sea] - la) * 111.0
            dx = (LO[sea] - lo) * 111.0 * np.cos(np.radians(la))
            out[i] = float(np.sqrt(dx * dx + dy * dy).min())
            break
    return out


def _sea_fraction(lats: np.ndarray, lons: np.ndarray,
                  radius_km: float) -> np.ndarray:
    """지점 주변 반경 안에서 바다가 차지하는 비율.

    위성 9x9 창이 실제로 바다를 얼마나 보고 있는지에 대응한다. 해안
    관측소는 창의 절반이 바다라 관측값 해석이 달라진다.
    """
    out = np.zeros(len(lats))
    for i, (la, lo) in enumerate(zip(lats, lons)):
        rd = radius_km / 111.0
        gl = np.arange(la - rd, la + rd + GRID_STEP, GRID_STEP)
        go = np.arange(lo - rd, lo + rd + GRID_STEP, GRID_STEP)
        LA, LO = np.meshgrid(gl, go, indexing="ij")
        dy = (LA - la) * 111.0
        dx = (LO - lo) * 111.0 * np.cos(np.radians(la))
        inside = (dx * dx + dy * dy) <= radius_km ** 2
        if inside.any():
            out[i] = float((~globe.is_land(LA[inside], LO[inside])).mean())
    return out


def build(stations: pd.DataFrame) -> pd.DataFrame:
    """지점표(STN, LAT, LON, HT) -> 정적 지리 피처."""
    la = stations["LAT"].to_numpy()
    lo = stations["LON"].to_numpy()
    out = pd.DataFrame({"STN": stations["STN"].to_numpy()})
    out["COAST_KM"] = _coast_distance(la, lo)
    out["IS_COASTAL"] = (out["COAST_KM"] < 10).astype(int)
    # 위성 창(18km)과 그보다 넓은 규모, 두 축척으로 본다
    out["SEA_FRAC_18"] = _sea_fraction(la, lo, 18.0)
    out["SEA_FRAC_50"] = _sea_fraction(la, lo, 50.0)
    # 동/서/남해 어느 쪽에 붙어 있는지가 기단 유입 방향을 가른다
    out["LOG_COAST"] = np.log1p(out["COAST_KM"])
    return out


if __name__ == "__main__":
    stn = pd.read_csv("stations_scored.csv")
    g = build(stn)
    g = g.merge(stn[["STN", "NAME", "HT", "LAT", "LON"]], on="STN")
    print(f"지점 {len(g)}개\n")
    print("해안에서 가장 먼 5곳:")
    print(g.nlargest(5, "COAST_KM")[
        ["STN", "NAME", "HT", "COAST_KM", "SEA_FRAC_18"]].round(1)
        .to_string(index=False))
    print("\n가장 가까운 5곳:")
    print(g.nsmallest(5, "COAST_KM")[
        ["STN", "NAME", "HT", "COAST_KM", "SEA_FRAC_18"]].round(1)
        .to_string(index=False))
    g.drop(columns=["NAME", "HT", "LAT", "LON"]).to_csv("geofeat.csv", index=False)
    print(f"\n-> geofeat.csv ({len(g.columns)-4}개 피처)")
