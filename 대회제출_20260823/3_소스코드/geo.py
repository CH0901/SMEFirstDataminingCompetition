"""GK2A 격자 <-> 위경도 변환, 지점 주변 화소 추출.

GK2A KO 영역은 Lambert Conformal Conic 투영이고, 투영 파라미터가 NetCDF
전역속성에 전부 들어있다. 그래서 파일에서 직접 읽어 변환기를 만들면
채널 해상도(0.5/1/2km)가 달라도 같은 코드로 처리된다.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from pyproj import CRS, Transformer


@lru_cache(maxsize=8)
def _transformer(lat0: float, lon0: float, sp1: float, sp2: float) -> Transformer:
    crs = CRS.from_proj4(
        f"+proj=lcc +lat_1={sp1} +lat_2={sp2} +lat_0={lat0} +lon_0={lon0} "
        "+x_0=0 +y_0=0 +a=6378137 +rf=298.257222101 +units=m +no_defs"
    )
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True)


def lonlat_to_rowcol(lon, lat, meta: dict) -> tuple[np.ndarray, np.ndarray]:
    """위경도 -> (row, col). meta 는 read_gk2a 가 돌려준 전역속성 dict."""
    tf = _transformer(
        float(meta["origin_latitude"]),
        float(meta["central_meridian"]),
        float(meta["standard_parallel1"]),
        float(meta["standard_parallel2"]),
    )
    x, y = tf.transform(np.asarray(lon, float), np.asarray(lat, float))
    ps = float(meta["pixel_size"])
    col = (x - float(meta["upper_left_easting"])) / ps
    row = (float(meta["upper_left_northing"]) - y) / ps
    return row, col


def extract_windows(img: np.ndarray, rows, cols, half: int = 4) -> np.ndarray:
    """각 지점 주변 (2*half+1)^2 창을 잘라 (n_points, w, w) 로 돌려준다.

    격자 밖이거나 창이 잘리는 지점은 NaN 으로 채운다.
    half=4 면 9x9 (2km 채널 기준 18km 반경).
    """
    w = 2 * half + 1
    n = len(rows)
    out = np.full((n, w, w), np.nan, dtype=np.float32)
    H, W = img.shape
    r0 = np.rint(np.asarray(rows)).astype(int)
    c0 = np.rint(np.asarray(cols)).astype(int)

    for i in range(n):
        rs, re = r0[i] - half, r0[i] + half + 1
        cs, ce = c0[i] - half, c0[i] + half + 1
        if rs < 0 or cs < 0 or re > H or ce > W:
            continue  # 가장자리 지점은 NaN 유지
        out[i] = img[rs:re, cs:ce].astype(np.float32)
    return out


def window_features(win: np.ndarray, prefix: str) -> dict[str, np.ndarray]:
    """창 배열 (n, w, w) -> 지점별 요약 통계.

    중심 화소만 쓰면 구름 가장자리에서 값이 널뛴다. 주변 통계를 같이 넣으면
    그 노이즈가 줄고, 표준편차 자체가 '구름이 깨져 있는지'를 알려준다.

    방향성도 함께 뽑는다. 평균 하나로 뭉개면 '서쪽은 구름, 동쪽은 맑음' 같은
    공간 배치가 사라지는데, 동해안의 큰 오차(푄·해풍)가 바로 그런 배치에서
    온다. 사분면 평균과 동서/남북 기울기가 그 정보를 담는다.
    """
    n, w, _ = win.shape
    c = w // 2
    flat = win.reshape(n, -1)
    inner = win[:, c - 1:c + 2, c - 1:c + 2].reshape(n, -1)
    with np.errstate(all="ignore"):
        # 배열은 [row, col] = [북->남, 서->동] 순서다
        north = np.nanmean(win[:, :c, :].reshape(n, -1), axis=1)
        south = np.nanmean(win[:, c + 1:, :].reshape(n, -1), axis=1)
        west = np.nanmean(win[:, :, :c].reshape(n, -1), axis=1)
        east = np.nanmean(win[:, :, c + 1:].reshape(n, -1), axis=1)
        return {
            f"{prefix}_c":    win[:, c, c],
            f"{prefix}_m3":   np.nanmean(inner, axis=1),
            f"{prefix}_m9":   np.nanmean(flat, axis=1),
            f"{prefix}_sd9":  np.nanstd(flat, axis=1),
            f"{prefix}_min9": np.nanmin(flat, axis=1),
            f"{prefix}_max9": np.nanmax(flat, axis=1),
            # 공간 배치: 기울기가 이류 방향과 지형 대비를 담는다
            f"{prefix}_gEW":  east - west,
            f"{prefix}_gNS":  south - north,
            f"{prefix}_gW":   west,
            f"{prefix}_gE":   east,
        }


# ---------------------------------------------------------------- 기하

# GK2A 는 정지궤도라 '통과 시각' 이 없다. 동경 128.2도 상공에 고정되어
# 같은 반구를 계속 관측한다. 그래서 위성 기하는 지점마다 상수다.
GK2A_SUB_LON = 128.2
GEO_ALT_KM = 35786.0
EARTH_R_KM = 6378.137


def satellite_zenith(lon, lat, sub_lon: float = GK2A_SUB_LON) -> np.ndarray:
    """지점에서 본 위성의 천정각(도).

    비스듬히 볼수록 복사가 통과하는 대기 경로가 길어져 관측값이 달라진다.
    정지위성이라 지점당 상수 -> 한 번 계산해 피처로 붙이면 된다.
    """
    lon = np.radians(np.asarray(lon, float) - sub_lon)
    lat = np.radians(np.asarray(lat, float))
    # 지심각
    psi = np.arccos(np.clip(np.cos(lat) * np.cos(lon), -1, 1))
    r = EARTH_R_KM + GEO_ALT_KM
    # 삼각형(지구중심-지점-위성)에서 천정각
    return np.degrees(np.arctan2(r * np.sin(psi),
                                 r * np.cos(psi) - EARTH_R_KM))


def solar_position(when, lon, lat) -> tuple[np.ndarray, np.ndarray]:
    """태양 천정각·방위각(도). when 은 UTC 시각(스칼라 또는 배열).

    NOAA 근사식. 기온 일변화의 원인이자 가시광 채널 밝기의 기준이라
    시각을 그냥 숫자로 넣는 것보다 훨씬 물리적인 피처가 된다.
    """
    t = pd.to_datetime(when, utc=True)
    t = pd.DatetimeIndex(np.atleast_1d(t))
    lon = np.asarray(lon, float)
    lat = np.asarray(lat, float)

    # 율리우스 세기
    jd = t.to_julian_date().to_numpy()
    jc = (jd - 2451545.0) / 36525.0

    geom_mean_lon = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360
    geom_mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    m = np.radians(geom_mean_anom)
    sun_eq = (np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + np.sin(2 * m) * (0.019993 - 0.000101 * jc)
              + np.sin(3 * m) * 0.000289)
    true_lon = geom_mean_lon + sun_eq
    omega = 125.04 - 1934.136 * jc
    app_lon = np.radians(true_lon - 0.00569 - 0.00478 * np.sin(np.radians(omega)))

    e0 = (23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813)))
                / 60) / 60)
    oblique = np.radians(e0 + 0.00256 * np.cos(np.radians(omega)))
    decl = np.arcsin(np.sin(oblique) * np.sin(app_lon))

    # 균시차 (분)
    y = np.tan(oblique / 2) ** 2
    gml = np.radians(geom_mean_lon)
    eot = 4 * np.degrees(
        y * np.sin(2 * gml) - 2 * ecc * np.sin(m)
        + 4 * ecc * y * np.sin(m) * np.cos(2 * gml)
        - 0.5 * y * y * np.sin(4 * gml) - 1.25 * ecc * ecc * np.sin(2 * m)
    )

    minutes = (t.hour * 60 + t.minute + t.second / 60).to_numpy()
    true_solar = (minutes + eot + 4 * lon) % 1440
    hour_angle = np.radians(np.where(true_solar / 4 < 0,
                                     true_solar / 4 + 180,
                                     true_solar / 4 - 180))

    latr = np.radians(lat)
    cos_z = (np.sin(latr) * np.sin(decl)
             + np.cos(latr) * np.cos(decl) * np.cos(hour_angle))
    zenith = np.degrees(np.arccos(np.clip(cos_z, -1, 1)))

    az = np.degrees(np.arctan2(
        np.sin(hour_angle),
        np.cos(hour_angle) * np.sin(latr) - np.tan(decl) * np.cos(latr),
    )) + 180
    return zenith, az % 360
