"""선형+GBM 구조에서 채널 조합을 다시 고른다.

기존 선택(IR087/NR016/SW038)은 GBM 단독 기준이었다. 구조가 바뀌었으니
다시 재야 하고, 채점 때 다운로드 장수(=실패 위험)도 같이 본다.
"""
import pandas as pd, numpy as np
import splitwindow as SW, eval14 as E
from robust import fit_predict, evaluate

df, F_BASE, F_SW, F_EXTRA = SW.build()
IN = pd.date_range("2025-08-24","2025-08-30")
OUT = df.loc[df.date.dt.month<=7,"date"].unique()
tr = df[~df.date.isin(IN) & ~df.date.isin(OUT)]
te_in, te_out = df[df.date.isin(IN)], df[df.date.isin(OUT)]

def cols(chs):
    return [c for c in df.columns if c.split("_")[0] in chs]

COMBOS = [
    (["IR105","IR123"],                       True),
    (["IR105","IR123","NR016"],               True),
    (["IR105","IR123","SW038"],               True),
    (["IR105","IR123","IR087"],               True),
    (["IR105","IR123","NR016","SW038"],       True),
    (["IR087","NR016","SW038"],               False),
    (["IR105","IR123","IR087","NR016","SW038"], True),
]
print(f"학습 {len(tr):,}행 · 계절안 {len(te_in):,} · 계절밖 {len(te_out):,}\n")
print(f"{'채널':34s} {'장수':>4s} | {'계절안':>7s} | {'계절밖':>7s} {'편향':>7s}")
print(f"{'-'*34} {'-'*4} | {'-'*7} | {'-'*7} {'-'*7}")
for chs, use_sw in COMBOS:
    f = E.BASE + cols(chs) + (F_EXTRA if use_sw else [])
    ta = fit_predict(tr, f, "TA14", True); hm = fit_predict(tr, f, "HM14", True)
    a, b = evaluate(ta,hm,te_in), evaluate(ta,hm,te_out)
    n = len(chs)*6*7          # 채점 7일 x 6시각 x 채널수
    print(f"  {','.join(chs):32s} {n:4d} | {a[0]:7.3f} | {b[0]:7.3f} {b[3]:+7.2f}", flush=True)
