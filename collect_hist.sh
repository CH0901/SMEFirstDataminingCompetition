#!/bin/zsh
# 2019~2022년 8~9월 수집. 위성은 2019년부터 있는데 우리는 2023년부터만 썼다.
# 용량을 키우니(leaves 31->127) 성능이 올랐다는 건 모델이 데이터를 더 소화할
# 여력이 있다는 뜻이다. '88일에서 포화' 라던 결론은 용량이 작던 시절 것이다.
cd "$(dirname "$0")"
PY=./.venv/bin/python
# 라벨 먼저 (빠름)
for Y in 2021 2022; do
  [ -f data/asos14_${Y}.parquet ] || $PY collect_labels.py ${Y}0601 ${Y}0930 data/asos14_${Y}.parquet 2>&1 | grep -E "^\[labels\]"
done
# 위성 — 채점 시기와 같은 8~9월만
for Y in 2022 2021 2020 2019; do
  echo "--- 위성 ${Y}년 8~9월 ---"
  $PY collect.py sat --start ${Y}0801 --end ${Y}0930 2>&1 | grep -E "^\[sat\]" | tail -1
  echo "  누적 $(ls data/sat_final 2>/dev/null | wc -l | tr -d ' ')일"
done
echo "=== 과거 데이터 수집 완료 $(date +%H:%M) ==="
