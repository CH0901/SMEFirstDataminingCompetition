"""채널별 기여도 측정.

학습에 쓴 채널은 채점 때도 전부 다운로드해야 한다. 채널 하나당 추론
다운로드가 늘고, 그만큼 API 장애로 통째로 실패할 확률도 커진다.
그래서 '많이 넣고 나중에 빼기'가 아니라 '처음부터 적게 고르기'가 맞다.

방법: 기후값+메타만 쓴 기준 모델에 채널을 하나씩만 더해 개선폭을 잰다.
채널마다 독립적으로 재므로 표본이 적어도 순위가 비교적 안정적이다.
"""
from __future__ import annotations

import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from sme.core import labels

SEED = 42
# 창 통계 중 이것만 쓴다. min9/max9 는 m9 와 상관이 높아 표본이 적을 때
# 과적합만 늘린다.
WIN_STATS = ("c", "m9", "sd9")


def load_satellite() -> pd.DataFrame:
    files = sorted(glob.glob("data/sat_days/*.parquet"))
    if not files:
        raise SystemExit("위성 데이터가 없습니다.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def to_daily(sat: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    """시각별 위성 -> 일 단위 요약.

    타깃이 하루 1개 값이므로 하루치를 접어야 한다. 주간/야간을 나누는 게
    핵심이다 — 지표는 낮에 데워지고 밤에 식으며, 그 진폭 자체가
    구름·습도의 지표가 된다.
    """
    d = sat.copy()
    d["date"] = (d["time_utc"] + pd.Timedelta(hours=9)).dt.normalize()  # KST
    d["is_day"] = d["SOL_ZEN"] < 85

    cols = [f"{ch}_{s}" for ch in channels for s in WIN_STATS
            if f"{ch}_{s}" in d.columns]
    g = d.groupby(["STN", "date"])

    out = g[cols].mean().add_suffix("_dmean")
    out = out.join(g[cols].std().add_suffix("_dstd"))
    for flag, tag in ((True, "day"), (False, "night")):
        sub = d[d["is_day"] == flag].groupby(["STN", "date"])[cols].mean()
        out = out.join(sub.add_suffix(f"_{tag}"))
    # 주야 진폭: 지표 열관성과 구름 지속성을 함께 담는다
    for c in cols:
        if f"{c}_day" in out.columns and f"{c}_night" in out.columns:
            out[f"{c}_amp"] = out[f"{c}_day"] - out[f"{c}_night"]
    return out.reset_index()


def assemble(channels: list[str]) -> pd.DataFrame:
    sat = load_satellite()
    daily_sat = to_daily(sat, channels)

    lab = pd.read_parquet("data/labels_daily.parquet")
    lab["date"] = pd.to_datetime(lab["date"])
    stn = pd.read_csv("data/reference/stations_scored.csv")

    # 기하는 하루 대표값(정오 부근)만 남긴다
    geo = sat.copy()
    geo["date"] = (geo["time_utc"] + pd.Timedelta(hours=9)).dt.normalize()
    geo = geo.groupby(["STN", "date"]).agg(
        SAT_ZEN=("SAT_ZEN", "first"), SOL_ZEN_MIN=("SOL_ZEN", "min")
    ).reset_index()

    df = daily_sat.merge(geo, on=["STN", "date"])
    df = df.merge(lab[["STN", "date", "TA_MEAN", "HM_MEAN"]],
                  on=["STN", "date"], how="inner")
    df = df.merge(stn[["STN", "LAT", "LON", "HT"]], on="STN", how="inner")

    df["DOY"] = df["date"].dt.dayofyear
    df["DOY_SIN"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * df["DOY"] / 365.25)
    return df


def add_climo(df: pd.DataFrame) -> pd.DataFrame:
    """기후 평년값을 붙인다. 위성이 못 오면 이것만으로도 예측이 나간다."""
    lab = pd.read_parquet("data/labels_daily.parquet")
    lab["date"] = pd.to_datetime(lab["date"])
    # 위성 표본에 들어간 날짜는 평년값 계산에서 빼 누수를 막는다
    used = set(zip(df["STN"], df["date"]))
    mask = ~pd.Series(list(zip(lab["STN"], lab["date"]))).isin(used)
    climo = labels.build_climatology(lab[mask.to_numpy()])
    return df.merge(climo, on=["STN", "DOY"], how="left")


BASE_COLS = ["LAT", "LON", "HT", "DOY_SIN", "DOY_COS",
             "SAT_ZEN", "SOL_ZEN_MIN", "TA_CLIMO", "HM_CLIMO"]

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1,
              seed=SEED, deterministic=True, force_row_wise=True)


def cv_rmse(df: pd.DataFrame, feats: list[str], target: str,
            n_splits: int = 4) -> float:
    """날짜 단위 그룹 CV. 같은 날이 학습/검증에 동시에 들어가면
    이웃 지점끼리 정보가 새어 성능이 부풀려진다."""
    X, y = df[feats], df[target]
    groups = df["date"]
    oof = np.full(len(df), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y.iloc[tr]),
                      num_boost_round=300)
        oof[va] = m.predict(X.iloc[va])
    ok = ~np.isnan(oof) & y.notna().to_numpy()
    return float(np.sqrt(((y.to_numpy()[ok] - oof[ok]) ** 2).mean()))


if __name__ == "__main__":
    CHANNELS = ["IR087", "IR096", "IR105", "IR112", "IR123", "IR133",
                "NR013", "NR016", "SW038", "WV063", "WV069", "WV073"]
    df = add_climo(assemble(CHANNELS)).dropna(subset=["TA_CLIMO", "HM_CLIMO"])
    print(f"표본 {len(df):,}행 · 날짜 {df['date'].nunique()}일 · "
          f"지점 {df['STN'].nunique()}개\n")

    base = {t: cv_rmse(df, BASE_COLS, t) for t in ("TA_MEAN", "HM_MEAN")}
    print(f"{'기준(기후값+메타, 위성 없음)':32s} "
          f"TA {base['TA_MEAN']:.3f}   HM {base['HM_MEAN']:.3f}   "
          f"평균 {(base['TA_MEAN']+base['HM_MEAN'])/2:.3f}\n")

    rows = []
    for ch in CHANNELS:
        f = BASE_COLS + [c for c in df.columns if c.startswith(ch + "_")]
        ta, hm = cv_rmse(df, f, "TA_MEAN"), cv_rmse(df, f, "HM_MEAN")
        rows.append((ch, ta, hm, base["TA_MEAN"] - ta, base["HM_MEAN"] - hm))
        print(f"  +{ch:6s}  TA {ta:.3f} ({base['TA_MEAN']-ta:+.3f})   "
              f"HM {hm:.3f} ({base['HM_MEAN']-hm:+.3f})", flush=True)

    r = pd.DataFrame(rows, columns=["채널", "TA", "HM", "TA개선", "HM개선"])
    r["종합개선"] = (r["TA개선"] + r["HM개선"]) / 2
    print("\n=== 기여도 순위 (지표가 (RMSE_TA+RMSE_HM)/2 이므로 종합 기준) ===")
    print(r.sort_values("종합개선", ascending=False).round(3).to_string(index=False))
    r.to_csv("data/reference/channel_value.csv", index=False)
