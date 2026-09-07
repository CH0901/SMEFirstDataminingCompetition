#!/bin/zsh
# 2026년 데이터. 규칙 3 이 '2026-08-23 까지 관측된 데이터' 를 허용한다.
# 채점이 2026-08-24~30 인데 학습에 2026 년 사례가 하나도 없으면 연도 피처가
# 검증되지 않은 외삽을 하게 된다.
cd "$(dirname "$0")/.."; PY=./.venv/bin/python
echo "--- 2026 라벨 (14시만) ---"
$PY -m sme.collect.labels 20260601 20260822 data/asos14_2026.parquet 2>&1 | grep -E "^\[labels\]"
echo "--- 2026 위성 ---"
$PY -m sme.collect.satellite sat --start 20260610 --end 20260822 2>&1 | grep -E "^\[sat\]" | tail -1
echo "=== 2026 수집 완료 $(date +%H:%M) · sat_final $(ls data/sat_final|wc -l|tr -d ' ')일 ==="
