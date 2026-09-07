"""정답 레이블(14시 KST)만 수집한다.

전 시각을 받을 필요가 없다. 타깃이 14시 한 시각이므로 하루 1회 호출이면
충분하고, 4년치도 몇 분이면 끝난다. (전 시각 수집은 하루 24회였다)
"""
import sys, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import kma

start, end, out = sys.argv[1], sys.argv[2], sys.argv[3]
days = pd.date_range(start, end, freq="D")
rows, fail = [], 0
with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(kma.fetch_asos_hour, d.strftime("%Y%m%d") + "1400"): d
            for d in days}
    for i, fut in enumerate(as_completed(futs), 1):
        try:
            df = fut.result()
            if len(df):
                rows.append(df)
        except Exception:
            fail += 1
        if i % 50 == 0:
            print(f"  {i}/{len(days)}일  실패 {fail}", flush=True)

if not rows:
    # API 가 죽어 한 건도 못 받은 경우. 빈 concat 은 예외를 내므로 먼저 걸러낸다.
    print(f"[labels] 수집 실패 — {len(days)}일 전부 응답 없음. 나중에 다시 실행하세요.")
    raise SystemExit(1)

df = pd.concat(rows, ignore_index=True).sort_values(["TM", "STN"])
df.to_parquet(out, index=False)
print(f"[labels] {len(df):,}행 · {df.TM.dt.normalize().nunique()}일 · "
      f"지점 {df.STN.nunique()}개 -> {out}  (실패 {fail})")
