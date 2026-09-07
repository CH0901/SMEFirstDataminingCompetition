"""모델 학습 -> model.pkl.

산출물은 캐글 Dataset 으로 올려 추론 노트북에서 불러 쓴다.
추론 때 피처를 똑같이 만들어야 하므로, 피처 이름 순서와 기후값 표를
모델과 함께 한 묶음으로 저장한다.

재현성: 규정이 "무작위성 제거"를 요구한다. 시드를 고정하고 LightGBM 의
deterministic 옵션을 켠다.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sme.core import evaluation
from sme.core import paths

SEED = 42
# 시드만 바꿔 학습한 모델을 평균낸다. 부스팅은 배깅·피처샘플링 때문에
# 시드마다 결과가 흔들리는데, 평균이 그 분산을 걷어낸다.
#   실측(홀드아웃 1주): 1개 2.282 -> 3개 2.254 -> 5개 2.250
# 시드가 고정되어 있으므로 재실행해도 같은 결과가 나온다 (규칙: 무작위성 제거).
SEEDS = [42, 7, 2024]
OUT = paths.MODEL_FILE

# 날짜 그룹 5-fold CV 로 고른 설정. 초기값(leaves31/lr.04/600r)은 표본이
# 59일이던 시절에 정한 것이라 데이터가 4배 늘어난 지금은 용량이 부족했다.
#   기준선 2.484 -> leaves63 2.467 -> +lr.02/1500r 2.458 -> leaves127 2.454
# 라운드는 1500 에서 포화했고(2500 도 동일), 선형 규제는 강할수록 나았다.
PARAMS = dict(objective="regression", metric="rmse", learning_rate=0.02,
              num_leaves=127, min_data_in_leaf=30, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=SEED, bagging_seed=SEED,
              feature_fraction_seed=SEED, deterministic=True,
              force_row_wise=True, num_threads=4)
ROUNDS = 1500
RIDGE_ALPHA = 500.0


def main() -> None:
    from sme.core import kma_client
    channels = list(kma_client.DEFAULT_CHANNELS)
    df = evaluation.assemble(channels)
    # 채널 파생 + 분리대기창 보정항(SWD_*) 을 모두 포함시킨다.
    feats = evaluation.BASE + [c for c in df.columns
                           if c.split("_")[0] in channels
                           or c.startswith(("SWD_", "SWDmin_", "CLD_"))]
    ano = [c for c in df.columns if c.startswith(("ANO_", "NB_", "DW_", "DE_", "DN_", "DS_", "dEW_", "dNS_"))]
    feats = [f for f in dict.fromkeys(feats) if f in df.columns]
    # 지점번호는 GBM 에만 넣는다 (선형에 넣으면 번호 크기를 비교하게 된다)
    gbm_feats = feats + evaluation.CAT

    # 연도는 타깃마다 다르게 쓴다.
    # 8월 말 기온은 연도 간 변동이 커서 연도를 넣어야 편향이 잡히지만
    # (2026 검증: TA 1.506 -> 1.435, 편향 -0.54 -> +0.15), 습도는 그 연도
    # 효과가 계절과 뒤섞여 오히려 크게 나빠졌다 (HM 7.078 -> 9.654).
    #   TA:연도O HM:연도X = 2026 검증 2.143 (최고), 2025 검증 1.862
    # 전국 편차와 이웃 평균은 기온에만 넣는다. 습도에서는 오히려 나빠졌다
    # (HM 7.078 -> 7.340). 지표가 TA + 0.1*HM 이라 각각 최적을 따로 쓴다.
    feats_by_target = {
        "TA14": feats + ano,
        "HM14": [f for f in feats if f != "YEAR"],
    }
    print(f"학습 표본 {len(df):,}행 · 피처 {len(feats)}개 · "
          f"날짜 {df['date'].nunique()}일")

    # 선형(Ridge) 을 앞에 두고 GBM 은 그 잔차만 학습한다.
    # 트리는 학습 범위 밖으로 외삽하지 못해 '8월이니까 30도' 를 6월에도
    # 그대로 내놓았다(편향 +4.3도). 선형항은 기울기를 따라 값이 이어지므로
    # 그 편향을 걷어낸다. 계절 안 성능도 함께 좋아졌다.
    models, linears = {}, {}
    for target in ("TA14", "HM14"):
        sub = df[df[target].notna()]
        tf = feats_by_target[target]
        tg = tf + evaluation.CAT
        X = sub[tf].astype(np.float64)
        y = sub[target].to_numpy(dtype=np.float64)

        lin = make_pipeline(SimpleImputer(strategy="median"),
                            StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
        lin.fit(X, y)
        linears[target] = lin
        resid = y - lin.predict(X)

        ds = lgb.Dataset(sub[tg], resid, categorical_feature=evaluation.CAT)
        models[target] = [
            lgb.train(dict(PARAMS, seed=sd, bagging_seed=sd,
                           feature_fraction_seed=sd), ds, num_boost_round=ROUNDS)
            for sd in SEEDS
        ]
        print(f"  {target}: {len(sub):,}행 · 선형+GBM(시드 {len(SEEDS)}개) 학습 완료")

    # 기후 평년값은 싣지 않는다.
    # 운영진 답변: "ASOS는 모델 학습시 label값으로만 활용 가능하고
    # 그 외 입력값으로 활용하실 수 없습니다."
    # 지점별 과거 평균기온·기후 평년값을 추론에 쓰는 것은 금지 대상이다.
    # 위성이 결측이면 모델이 위경도·고도·연중일만으로 예측한다
    # (LightGBM 이 결측 피처를 스스로 처리한다).
    # 지점 좌표·고도는 대회가 제공한 station_list.csv 를 그대로 쓴다.
    # (운영진: "station_list 파일에 있는 정적인 정보를 사용하시면 됩니다")
    stn = pd.read_csv("data/reference/stations_scored.csv")

    # GK2A 는 정지위성이고 KO 격자는 고정이므로 지점별 화소 좌표가 상수다.
    # 미리 계산해 실어두면 추론 노트북에서 pyproj 가 필요 없어진다.
    from sme.core.grid import lonlat_to_rowcol, satellite_zenith
    grid = dict(origin_latitude=38.0, central_meridian=126.0,
                standard_parallel1=30.0, standard_parallel2=60.0,
                pixel_size=2000.0, upper_left_easting=-899000.0,
                upper_left_northing=899000.0)
    rows, cols = lonlat_to_rowcol(stn["LON"].to_numpy(),
                                  stn["LAT"].to_numpy(), grid)
    sat_zen = satellite_zenith(stn["LON"].to_numpy(), stn["LAT"].to_numpy())

    bundle = {
        "models": models,
        "linears": linears,   # 외삽 담당 선형 성분
        "features": feats,          # 전체 피처 (참고용)
        "gbm_features": gbm_feats,
        "features_by_target": feats_by_target,          # 선형 입력 (타깃별)
        "gbm_features_by_target": {k: v + evaluation.CAT
                                   for k, v in feats_by_target.items()},
        "categorical": evaluation.CAT,
        "stn_categories": list(df["STN_CAT"].cat.categories),
        "channels": channels,
        "win_stats": list(evaluation.WIN_STATS),
        "stations": stn,            # STN, LAT, LON, HT
        "sat_zenith": sat_zen,   # 지점별 위성 천정각(도) — 정지위성이라 상수
        "pixel_rows": rows,         # 지점별 화소 좌표 (2km 격자 고정)
        "pixel_cols": cols,
        "grid": grid,               # 검증용 격자 정의
        "target_hour_utc": evaluation.TARGET_HOUR_UTC,
        "hours_utc": [0, 2, 4, 5],          # 당일 (≤05 UTC = 14 KST)
        "hours_utc_prev": [18, 21],         # 전날 늦은 시각
        "seeds": SEEDS,
        "lightgbm_version": lgb.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }
    with open(OUT, "wb") as f:
        pickle.dump(bundle, f, protocol=4)  # 구버전 호환 위해 protocol 4
    print(f"\n저장: {OUT}  ({OUT.stat().st_size/1e6:.2f} MB)")
    print("  기후 평년값 미포함 (ASOS 는 학습 라벨로만 사용)")
    print(f"  lightgbm {lgb.__version__} / pandas {pd.__version__} / "
          f"numpy {np.__version__}")
    print("  → 캐글 Dataset 으로 Public 업로드 후 노트북 Input 에 연결")


if __name__ == "__main__":
    main()
