"""전진 선택으로 최소 채널 조합을 찾는다.

채널을 늘리면 추론 때 받을 파일이 늘고 실패 확률도 커진다.
개선폭이 꺾이는 지점에서 멈추는 게 목적.
"""
import pandas as pd, numpy as np
from channel_value import assemble, add_climo, cv_rmse, BASE_COLS

ALL = ["IR087","IR096","IR105","IR112","IR123","IR133",
       "NR013","NR016","SW038","WV063","WV069","WV073"]
df = add_climo(assemble(ALL)).dropna(subset=["TA_CLIMO","HM_CLIMO"])

def score(chs):
    f = BASE_COLS + [c for ch in chs for c in df.columns if c.startswith(ch+"_")]
    ta, hm = cv_rmse(df, f, "TA_MEAN"), cv_rmse(df, f, "HM_MEAN")
    return ta, hm, (ta+hm)/2

b = score([])
print(f"채널 0개  {'':44s} TA {b[0]:.3f}  HM {b[1]:.3f}  종합 {b[2]:.3f}", flush=True)
chosen, rest, prev = [], ALL[:], b[2]
for _ in range(6):
    s, best = min((score(chosen+[c])[2], c) for c in rest)
    chosen.append(best); rest.remove(best)
    ta, hm, avg = score(chosen)
    print(f"채널 {len(chosen)}개 +{best:6s} {','.join(chosen):38s} "
          f"TA {ta:.3f}  HM {hm:.3f}  종합 {avg:.3f}  (개선 {prev-avg:+.3f})", flush=True)
    prev = avg
ta, hm, avg = score(ALL)
print(f"\n전체 12채널 {'':41s} TA {ta:.3f}  HM {hm:.3f}  종합 {avg:.3f}", flush=True)
