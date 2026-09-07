"""다중 시각 학습 — 14시 외 다른 시각도 학습에 쓴다.

규칙 3: "시각 제한은 없습니다. 14시 외 다른 시각의 데이터로 학습해도 됩니다."

지금까지 14시 타깃만 써서 21,307행으로 학습했다. 우리가 가진 위성 시각을
KST 로 옮기면 03, 06, 09, 11, 13, 14시인데, 이 중 09/11/13시도 타깃이 될 수
있다. 각 타깃에 대해 '그 시각까지 관측된' 위성만 쓰면 인과성도 지켜진다.

  타깃 14시 -> 사용 가능: 14, 13, 11, 09, 06, 03
  타깃 13시 -> 사용 가능:     13, 11, 09, 06, 03
  타깃 11시 -> 사용 가능:         11, 09, 06, 03
  타깃 09시 -> 사용 가능:             09, 06, 03

시각마다 개수가 다르므로 절대 시각으로 피처를 만들 수 없다. 대신 '가장 최근
관측(r0), 그 다음(r1), ...' 처럼 **타깃 기준 상대 순위**로 정렬한다. 이러면
모든 타깃이 같은 피처 구조를 갖고, 14시 예측에도 그대로 쓸 수 있다.

타깃 시각 자체와 태양 천정각을 피처로 넣어 시각 차이를 모델이 알게 한다.
"""
from __future__ import annotations

import glob

import numpy as np
import pandas as pd

import eval14 as E
import kma
from geo import satellite_zenith, solar_position

# KST 시각 -> 위성 UTC 시각. (전날 것은 KST 로 옮기면 당일 새벽이 된다)
KST_TO_UTC = {3: (-1, 18), 6: (-1, 21), 9: (0, 0), 11: (0, 2), 13: (0, 4), 14: (0, 5)}
TARGET_HOURS = (9, 11, 13, 14)          # 타깃으로 삼을 KST 시각
N_RANK = 6                              # 상대 순위 슬롯 개수


def load_sat() -> pd.DataFrame:
    files = sorted(glob.glob("data/sat_final/*.parquet"))
    sat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sat["utc_hour"] = sat["time_utc"].dt.hour
    # 위성 시각을 'KST 로 본 날짜와 시각' 으로 환산한다
    kst = sat["time_utc"] + pd.Timedelta(hours=9)
    sat["kst_date"] = kst.dt.normalize()
    sat["kst_hour"] = kst.dt.hour
    return sat


def build(channels: list[str]) -> pd.DataFrame:
    sat = load_sat()
    stats = [f"{c}_{s}" for c in channels for s in E.WIN_STATS
             if f"{c}_{s}" in sat.columns]

    # 라벨: 전 시각 ASOS 에서 타깃 시각들만 뽑는다
    files = sorted(glob.glob("data/asos_20??0601_*.parquet"))
    a = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    a["TM"] = pd.to_datetime(a["TM"])
    a = a[a["TM"].dt.hour.isin(TARGET_HOURS)]
    lab = pd.DataFrame({
        "STN": a["STN"], "date": a["TM"].dt.normalize(),
        "target_hour": a["TM"].dt.hour, "TA": a["TA"], "HM": a["HM"],
    })

    # 각 타깃 시각마다, 그 시각 이하의 관측을 최근 순으로 정렬해 붙인다
    frames = []
    for th in TARGET_HOURS:
        avail = sorted([h for h in KST_TO_UTC if h <= th], reverse=True)
        piece = lab[lab.target_hour == th].copy()
        for rank, h in enumerate(avail[:N_RANK]):
            src = sat[sat.kst_hour == h][["STN", "kst_date"] + stats]
            src = src.rename(columns={"kst_date": "date",
                                      **{c: f"{c}_r{rank}" for c in stats}})
            piece = piece.merge(src, on=["STN", "date"], how="left")
        # 부족한 순위는 NaN 으로 채워 구조를 맞춘다
        for rank in range(len(avail), N_RANK):
            for c in stats:
                piece[f"{c}_r{rank}"] = np.nan
        frames.append(piece)

    df = pd.concat(frames, ignore_index=True)

    # 분리대기창 보정항 (순위별)
    for rank in range(N_RANK):
        for st in ("c", "m9"):
            x, y = f"IR105_{st}_r{rank}", f"IR123_{st}_r{rank}"
            if x in df and y in df:
                df[f"SWD_{st}_r{rank}"] = df[x] - df[y]
    # 최근 관측과 그 이전의 변화 (구름이 끼는/걷히는 중인지)
    for c in stats:
        if f"{c}_r0" in df and f"{c}_r1" in df:
            df[f"{c}_d01"] = df[f"{c}_r0"] - df[f"{c}_r1"]

    stn = pd.read_csv("stations_scored.csv")
    df = df.merge(stn[["STN", "LAT", "LON", "HT"]], on="STN", how="inner")
    df["DOY"] = df["date"].dt.dayofyear
    df["DOY_SIN"] = np.sin(2 * np.pi * df.DOY / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * df.DOY / 365.25)
    z = satellite_zenith(df.LON.to_numpy(), df.LAT.to_numpy())
    df["SEC_SATZEN"] = 1.0 / np.cos(np.radians(z)) - 1.0
    # 타깃 시각의 태양 위치 — 시각 차이를 물리적으로 표현한다
    utc = df["date"] + pd.to_timedelta(df.target_hour - 9, unit="h")
    sz, _ = solar_position(utc, df.LON.to_numpy(), df.LAT.to_numpy())
    df["SOL_ZEN_T"] = sz
    df["SOL_COS_T"] = np.cos(np.radians(sz))
    df["TARGET_HOUR"] = df["target_hour"]
    df["STN_CAT"] = df["STN"].astype("category")
    return df


BASE = ["LAT", "LON", "HT", "DOY_SIN", "DOY_COS", "SEC_SATZEN",
        "SOL_ZEN_T", "SOL_COS_T", "TARGET_HOUR"]


def feature_cols(df: pd.DataFrame, channels: list[str]) -> list[str]:
    sat = [c for c in df.columns
           if (c.split("_")[0] in channels or c.startswith("SWD_"))
           and ("_r" in c or c.endswith("_d01"))]
    return BASE + sat


if __name__ == "__main__":
    CH = list(kma.DEFAULT_CHANNELS)
    df = build(CH)
    F = feature_cols(df, CH)
    print(f"표본 {len(df):,}행 (14시만 쓸 때 21,307행)")
    print(f"  타깃 시각별: {df.target_hour.value_counts().sort_index().to_dict()}")
    print(f"  피처 {len(F)}개 · 결측률 {df[F].isna().mean().mean()*100:.1f}%")
