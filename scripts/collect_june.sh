#!/bin/zsh
# 계절 밖 검증용. 6~7월을 받아 "학습 범위 밖에서 얼마나 무너지는지" 재고,
# 강건성 개선을 실제로 측정할 수 있게 한다.
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
for R in "20250624 20250630" "20240624 20240630" "20250701 20250714" \
         "20250615 20250623" "20240701 20240714"; do
  set -- ${=R}
  $PY -m sme.collect.satellite sat --start $1 --end $2 2>&1 | grep -E "^\[sat\]" | tail -1
  echo "  누적 $(ls data/sat_days/|wc -l|tr -d ' ')일"
done
