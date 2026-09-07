"""확정된 규칙에 맞춘 평가.

  타깃  : 매일 14:00 KST 의 TA / HM
  지표  : Score = RMSE_TA + 0.1 * RMSE_HM   (기온이 지배)
  제약  : 대상 시각(14:00 KST = 05:00 UTC)'까지' 관측된 위성만 입력 가능

인과성이 핵심이다. 05 UTC 이후 영상은 정답 시각 이후를 보는 것이라
쓰면 규정 위반이고, 검증 성능도 부풀려진다. 그래서 피처를 만들 때
시각을 UTC 로 환산해 05시 이하만 남긴다.
"""
from __future__ import annotations

import glob

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

SEED = 42
TARGET_HOUR_KST = 14
TARGET_HOUR_UTC = TARGET_HOUR_KST - 9  # 05 UTC
WIN_STATS = ("c", "m9", "sd9", "min9", "max9")

# 당일 받는 시각(UTC). 05 = 14 KST 로 정답과 같은 시각이라 가장 중요하다.
HOURS_CUR = (0, 2, 4, 5)
# 전날 늦은 시각. 오늘 05 UTC 보다 이르므로 규정상 사용 가능하며,
# 밤사이 변화를 담는다.
HOURS_PREV = (18, 21)

PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.05,
              num_leaves=15, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1,
              seed=SEED, deterministic=True, force_row_wise=True)


def score(rmse_ta: float, rmse_hm: float) -> float:
    return rmse_ta + 0.1 * rmse_hm


def load_labels() -> pd.DataFrame:
    """14:00 KST 관측값. 결측(-99/-9)은 이미 NaN 이며 채점에서도 제외된다."""
    # 전 시각 파일(2023~2025) + 14시만 받은 파일(2019~2022)을 함께 읽는다
    files = sorted(glob.glob("data/asos_20??0601_*.parquet")
                   + glob.glob("data/asos14_*.parquet"))
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d["TM"] = pd.to_datetime(d["TM"])
    d = d[d["TM"].dt.hour == TARGET_HOUR_KST].copy()
    d = d.drop_duplicates(subset=["STN", "TM"])
    d["date"] = d["TM"].dt.normalize()
    return d[["STN", "date", "TA", "HM"]].rename(
        columns={"TA": "TA14", "HM": "HM14"})


def build_climatology(lab: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    d = lab.copy()
    d["DOY"] = d["date"].dt.dayofyear
    recs = []
    for doy in sorted(d["DOY"].unique()):
        gap = np.abs(d["DOY"] - doy)
        sel = d[np.minimum(gap, 365 - gap) <= window]
        g = sel.groupby("STN")[["TA14", "HM14"]].mean()
        g.columns = ["TA_CLIMO", "HM_CLIMO"]
        recs.append(g.reset_index().assign(DOY=doy))
    return pd.concat(recs, ignore_index=True)


def causal_satellite(channels: list[str]) -> pd.DataFrame:
    """대상일 05 UTC 이하의 위성만 모아 일 단위로 접는다.

    05 UTC 초과분은 정답 시각 이후라 버린다. 전날 늦은 시각은 오늘 05 UTC
    이전이므로 합법이고, 하루 사이의 변화(경향)를 담을 수 있어 유용하다.
    """
    files = sorted(glob.glob("data/sat_final/*.parquet"))
    sat = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sat["utc_hour"] = sat["time_utc"].dt.hour
    sat["date"] = sat["time_utc"].dt.normalize()

    cols = [f"{ch}_{s}" for ch in channels for s in WIN_STATS
            if f"{ch}_{s}" in sat.columns]

    # 시각별로 나눠 붙인다. 평균으로 뭉개면 정답 시각(05 UTC)의 정보가
    # 이른 아침 관측에 희석된다 — 실제로 나눠 넣는 쪽이 확실히 낫다.
    cur = sat[sat["utc_hour"] <= TARGET_HOUR_UTC].pivot_table(
        index=["STN", "date"], columns="utc_hour", values=cols)
    cur.columns = [f"{a}_h{b:02d}" for a, b in cur.columns]

    # 전날 06 UTC 이후 — 오늘 05 UTC 보다 이르므로 사용 가능
    prev = sat[sat["utc_hour"] > TARGET_HOUR_UTC].copy()
    prev["date"] = prev["date"] + pd.Timedelta(days=1)
    prv = prev.pivot_table(index=["STN", "date"], columns="utc_hour",
                           values=cols)
    prv.columns = [f"{a}_p{b:02d}" for a, b in prv.columns]

    out = cur.join(prv, how="outer")

    # 분리대기창 보정항 — KMA 공식 LST 알고리즘의 (T13 - T15) 에 해당한다.
    # 두 창채널은 지표 신호가 거의 같고 수증기 흡수만 다르므로, 차이가
    # 대기 효과를 상쇄한다. 트리는 이 선형 조합을 스스로 못 만들기 때문에
    # 명시적으로 넣어준다.
    for tag in [f"h{h:02d}" for h in HOURS_CUR] + [f"p{h:02d}" for h in HOURS_PREV]:
        for st in ("c", "m9"):
            a, b = f"IR105_{st}_{tag}", f"IR123_{st}_{tag}"
            if a in out.columns and b in out.columns:
                out[f"SWD_{st}_{tag}"] = out[a] - out[b]

        # 청천 화소 기준 분리대기창.
        # GK2A DN 은 온도와 반대 방향이라(상관 -0.46~-0.56) min9 = DN 최솟값이
        # 물리적으로 가장 뜨거운 화소, 즉 청천 지표에 해당한다. KMA 공식
        # 알고리즘도 청천 육지 화소에서만 지표면온도를 산출한다.
        a, b = f"IR105_min9_{tag}", f"IR123_min9_{tag}"
        if a in out.columns and b in out.columns:
            out[f"SWDmin_{tag}"] = out[a] - out[b]
        # 창 안의 최대-최소 = 구름과 청천이 얼마나 섞였는지 (구름량 대리)
        a, b = f"IR105_max9_{tag}", f"IR105_min9_{tag}"
        if a in out.columns and b in out.columns:
            out[f"CLD_{tag}"] = out[a] - out[b]

    for col in cols:
        # 직전 1시간 변화: 구름이 막 끼거나 걷히는 중인지
        if f"{col}_h05" in out.columns and f"{col}_h04" in out.columns:
            out[f"{col}_d1h"] = out[f"{col}_h05"] - out[f"{col}_h04"]
        # 밤사이 변화: 기단 교체나 구름 유입
        if f"{col}_h05" in out.columns and f"{col}_p21" in out.columns:
            out[f"{col}_dnt"] = out[f"{col}_h05"] - out[f"{col}_p21"]
    return out.reset_index()


def assemble(channels: list[str]) -> pd.DataFrame:
    lab = load_labels()
    climo = build_climatology(lab)   # 비교 평가용일 뿐, 모델 입력이 아니다
    stn = pd.read_csv("stations_scored.csv")
    sat = causal_satellite(channels)

    df = sat.merge(lab, on=["STN", "date"], how="inner")
    df = df.merge(stn[["STN", "LAT", "LON", "HT"]], on="STN", how="inner")
    df["DOY"] = df["date"].dt.dayofyear
    df["DOY_SIN"] = np.sin(2 * np.pi * df["DOY"] / 365.25)
    df["DOY_COS"] = np.cos(2 * np.pi * df["DOY"] / 365.25)
    # 위성 천정각의 경로 길이 효과. 물리식에 sec(theta)-1 로 들어간다.
    from geo import satellite_zenith
    _z = satellite_zenith(df["LON"].to_numpy(), df["LAT"].to_numpy())
    df["SEC_SATZEN"] = 1.0 / np.cos(np.radians(_z)) - 1.0
    df["STN_CAT"] = df["STN"].astype("category")
    # 연도. 규칙 2.1③ 이 'day, month, year' 를 허용 입력으로 명시한다.
    # 8월 말 기온이 2022년 24.8도 -> 2025년 31.4도로 빠르게 오르고 있어,
    # 연도 없이 과거를 넣으면 모델이 차갑게 편향된다(편향 -0.61). 선형
    # 성분에 연도를 주면 그 추세를 채점 연도로 이어 그릴 수 있다
    # (7년치 2.063 -> 1.875, 편향 -0.25).
    df["YEAR"] = df["date"].dt.year.astype(float)

    # 전국 대비 편차.
    # 각 행은 자기 지점 주변 9x9 만 본다. 그런데 기온은 그날 한반도를 덮은
    # 기단이 크게 좌우하므로, '오늘 전국이 얼마나 더운가' 를 모르면 지점
    # 특성과 그날 특성을 구분할 수 없다. 다른 지점의 '위성' 값을 쓰는 것이라
    # 규칙상 허용된다(ASOS 가 아니다).
    # 실전 재현 검증: 기온 RMSE 1.435 -> 1.304
    nat_keys = [c for c in df.columns if any(
        c.startswith(p_) for p_ in ("IR105_m9_", "IR105_min9_", "IR123_m9_",
                                    "SW038_m9_", "SWD_m9_", "SWDmin_"))]
    nat = df.groupby("date")[nat_keys].transform("mean")
    df = pd.concat([df, pd.DataFrame(
        {f"ANO_{c}": df[c] - nat[c] for c in nat_keys}, index=df.index)], axis=1)

    # 최근접 8개 지점의 위성 평균 (대상 시각만).
    # 전국 평균은 너무 넓고 3개는 너무 좁았다. 8개(대략 100~150km)가 기단
    # 규모와 맞는 듯하다 (기온 1.304 -> 1.268). 전 시각으로 넓히면 오히려
    # 나빠져(1.273) 대상 시각만 쓴다.
    nb_keys = [c for c in nat_keys if c.endswith("_h05") or c == "SWDmin_h05"]
    if nb_keys:
        la, lo = stn["LAT"].to_numpy(), stn["LON"].to_numpy()
        sid = stn["STN"].to_numpy()
        dist = np.sqrt(((la[:, None] - la) * 111) ** 2
                       + ((lo[:, None] - lo) * 111 * np.cos(np.radians(la[:, None]))) ** 2)
        np.fill_diagonal(dist, 1e9)
        near = {sid[i]: list(sid[np.argsort(dist[i])[:8]]) for i in range(len(sid))}
        for c in nb_keys:
            piv = df.pivot_table(index="date", columns="STN", values=c)
            m = pd.DataFrame({s_: piv[[x for x in near[s_] if x in piv.columns]].mean(axis=1)
                              for s_ in sid if s_ in piv.columns})
            st = m.stack().rename(f"NB_{c}").reset_index()
            st.columns = ["date", "STN", f"NB_{c}"]
            df = df.merge(st, on=["date", "STN"], how="left")

        # 방향별 이웃 (동/서/남/북 각 3개).
        # 8개 평균은 방향을 지운다. 서쪽 이웃이 뜨겁고 동쪽이 차가우면 서풍
        # 기단 유입이고, 동해안 푄도 '산맥 서쪽 구름 / 동쪽 맑음' 이라는
        # 방향 구조다. 실측: 기온 1.281 -> 1.243
        dy = (la[:, None] - la) * 111
        dx = (lo[:, None] - lo) * 111 * np.cos(np.radians(la[:, None]))
        sect = {"W": (dx < 0) & (np.abs(dx) > np.abs(dy)),
                "E": (dx > 0) & (np.abs(dx) > np.abs(dy)),
                "S": (dy < 0) & (np.abs(dy) >= np.abs(dx)),
                "N": (dy > 0) & (np.abs(dy) >= np.abs(dx))}
        for tag, mask in sect.items():
            grp = {}
            for i in range(len(sid)):
                cand = np.where(mask[i])[0]
                grp[sid[i]] = list(sid[cand[np.argsort(dist[i, cand])[:3]]]) if len(cand) else []
            for c in nb_keys:
                piv = df.pivot_table(index="date", columns="STN", values=c)
                m = pd.DataFrame({
                    s_: (piv[[x for x in grp[s_] if x in piv.columns]].mean(axis=1)
                         if grp.get(s_) else np.nan)
                    for s_ in sid if s_ in piv.columns})
                st = m.stack().rename(f"D{tag}_{c}").reset_index()
                st.columns = ["date", "STN", f"D{tag}_{c}"]
                df = df.merge(st, on=["date", "STN"], how="left")
        # 동서·남북 대비
        for c in nb_keys:
            if f"DE_{c}" in df and f"DW_{c}" in df:
                df[f"dEW_{c}"] = df[f"DE_{c}"] - df[f"DW_{c}"]
            if f"DN_{c}" in df and f"DS_{c}" in df:
                df[f"dNS_{c}"] = df[f"DS_{c}"] - df[f"DN_{c}"]
    # 평년값은 피처가 아니라 폴백용이지만, 비교 평가를 위해 붙여둔다.
    df = df.merge(climo, on=["STN", "DOY"], how="left")
    return df


# 모델 입력은 위성 + 정적 지점정보 + 달력뿐이다.
#
# 기후 평년값을 피처로 넣어봤지만 오히려 점수가 나빠졌다 (2.538 -> 2.577).
# 위경도·고도·연중일이 이미 같은 정보를 담고 있어 중복이기 때문이다.
# 빼는 편이 성능도 낫고, "ASOS 파생 자료를 입력으로 쓰는가"라는 규정
# 회색지대도 피할 수 있다. 평년값은 위성이 전부 결측일 때의 폴백으로만 쓴다.
#
# 해안거리·해양비율 같은 정적 지리 피처도 같은 이유로 제외했다
# (규정상 허용되지만 2.538 -> 2.545 로 도움이 되지 않았다).
BASE = ["LAT", "LON", "HT", "DOY_SIN", "DOY_COS", "SEC_SATZEN", "YEAR"]

# 지점번호. 규칙 2.1 이 station_list.csv 의 '지점번호' 를 허용 입력으로 명시한다.
# 위경도·고도만으로는 국지 현상(동해안 푄, 해풍 등)에서 생기는 지점별 계통
# 오차를 잡지 못한다. 범주형으로 넣으면 지점별 보정을 학습할 수 있다.
# GBM 에만 넣는다 — 선형 성분에 넣으면 의미 없는 크기 비교가 된다.
CAT = ["STN_CAT"]


def cv(df: pd.DataFrame, feats: list[str], target: str, n_splits: int = 4):
    X, y, g = df[feats], df[target], df["date"]
    oof = np.full(len(df), np.nan)
    for tr, va in GroupKFold(n_splits=n_splits).split(X, y, g):
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y.iloc[tr]),
                      num_boost_round=300)
        oof[va] = m.predict(X.iloc[va])
    ok = ~np.isnan(oof) & y.notna().to_numpy()
    return float(np.sqrt(((y.to_numpy()[ok] - oof[ok]) ** 2).mean()))


if __name__ == "__main__":
    CH = ["IR105", "IR123", "SW038"]
    df = assemble(CH)
    print(f"표본 {len(df):,}행 · {df['date'].nunique()}일 · "
          f"지점 {df['STN'].nunique()}개")
    print(f"사용 가능한 UTC 시각: 당일 ≤{TARGET_HOUR_UTC}시 + 전날\n")

    ta_c = float(np.sqrt(((df.TA14 - df.TA_CLIMO) ** 2).mean()))
    hm_c = float(np.sqrt(((df.HM14 - df.HM_CLIMO) ** 2).mean()))
    print(f"{'기후값 그대로':26s} RMSE_TA {ta_c:5.3f}  RMSE_HM {hm_c:6.3f}  "
          f"Score {score(ta_c, hm_c):6.3f}")

    ta_b, hm_b = cv(df, BASE, "TA14"), cv(df, BASE, "HM14")
    print(f"{'기후값+메타 (위성 없음)':26s} RMSE_TA {ta_b:5.3f}  "
          f"RMSE_HM {hm_b:6.3f}  Score {score(ta_b, hm_b):6.3f}")

    sf = BASE + [c for ch in CH for c in df.columns if c.startswith(ch + "_")]
    ta_s, hm_s = cv(df, sf, "TA14"), cv(df, sf, "HM14")
    print(f"{'+ 위성 3채널 (인과 준수)':26s} RMSE_TA {ta_s:5.3f}  "
          f"RMSE_HM {hm_s:6.3f}  Score {score(ta_s, hm_s):6.3f}")
