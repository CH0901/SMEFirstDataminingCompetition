"""강건성 실험 — 계절 안/밖을 동시에 본다.

발견한 문제: 8~9월로만 학습한 모델이 6월에 +4.28도 편향을 냈다.
편향만 빼면 RMSE 1.97 이므로 상대 변동은 잘 잡는다. 즉 '기온의 절대 수준'을
위성이 아니라 계절에서 가져오고 있었다. 트리는 학습 범위 밖으로 외삽하지
못하므로, 8월 구간에서 배운 '30도쯤' 을 6월에도 그대로 내놓는다.

해법: 선형 성분을 앞에 두고 GBM 은 그 잔차만 학습한다.
선형은 기울기를 따라 범위 밖으로도 값이 이어지므로 외삽 담당이 되고,
GBM 은 범위 안의 비선형 구조를 담당한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sme.core import evaluation as E

CH = ["IR087", "NR016", "SW038"]
SEEDS = (42, 7, 2024)
P = dict(E.PARAMS, learning_rate=0.04, num_leaves=31, min_data_in_leaf=30,
         lambda_l2=1.0, num_threads=4)


def make_linear():
    """외삽 담당. 위성 피처가 float32 라 그대로 스케일링하면 정밀도 문제가
    생긴다. float64 로 올려서 계산한다."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=50.0),
    )


def fit_predict(tr, feats, target, hybrid):
    s = tr[tr[target].notna()]
    X = s[feats].astype(np.float64)
    y = s[target].to_numpy(dtype=np.float64)

    lin = None
    if hybrid:
        lin = make_linear()
        lin.fit(X, y)
        y = y - lin.predict(X)

    ms = [lgb.train(dict(P, seed=sd, bagging_seed=sd, feature_fraction_seed=sd),
                    lgb.Dataset(X, y), 600) for sd in SEEDS]

    def predict(df):
        Z = df[feats].astype(np.float64)
        p = np.mean([m.predict(Z) for m in ms], axis=0)
        return p + lin.predict(Z) if hybrid else p

    return predict


def evaluate(pred_ta, pred_hm, te):
    out = {}
    for tgt, f in (("TA14", pred_ta), ("HM14", pred_hm)):
        p = f(te)
        y = te[tgt].to_numpy(dtype=np.float64)
        ok = ~np.isnan(y)
        out[tgt] = (float(np.sqrt(((y[ok] - p[ok]) ** 2).mean())),
                    float((p[ok] - y[ok]).mean()))
    score = out["TA14"][0] + 0.1 * out["HM14"][0]
    return score, out["TA14"][0], out["HM14"][0], out["TA14"][1]


if __name__ == "__main__":
    df = E.assemble(CH)
    SATF = [c for c in df.columns if c.split("_")[0] in CH]
    IN = pd.date_range("2025-08-24", "2025-08-30")          # 계절 안
    OUT = df.loc[df.date.dt.month <= 7, "date"].unique()    # 계절 밖 (6~7월)
    tr = df[~df.date.isin(IN) & ~df.date.isin(OUT)]
    te_in, te_out = df[df.date.isin(IN)], df[df.date.isin(OUT)]
    print(f"학습 {len(tr):,}행 ({tr.date.nunique()}일) · "
          f"계절안 {len(te_in):,}행 · 계절밖 {len(te_out):,}행 "
          f"({len(pd.unique(OUT))}일)\n")

    print(f"{'구성':26s} | {'계절안':>7s} | {'계절밖':>7s} {'TA편향':>8s}")
    print(f"{'-'*26} | {'-'*7} | {'-'*7} {'-'*8}")
    for name, feats, hyb in [
        ("현재 (GBM only)",        E.BASE + SATF,             False),
        ("DOY 제거",               ["LAT", "LON", "HT"] + SATF, False),
        ("선형+GBM 잔차",           E.BASE + SATF,             True),
        ("DOY제거 + 선형+GBM",      ["LAT", "LON", "HT"] + SATF, True),
    ]:
        ta = fit_predict(tr, feats, "TA14", hyb)
        hm = fit_predict(tr, feats, "HM14", hyb)
        s_in = evaluate(ta, hm, te_in)
        s_out = evaluate(ta, hm, te_out)
        print(f"  {name:24s} | {s_in[0]:7.3f} | {s_out[0]:7.3f} {s_out[3]:+8.2f}",
              flush=True)
