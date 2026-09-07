#!/bin/zsh
# split-window 채널쌍(IR105 10.5um, IR123 12.3um) 추가 수집.
# KMA 공식 LST 알고리즘이 쓰는 조합이며, 두 채널의 차이가 수증기에 의한
# 대기 흡수를 보정한다. 기존 3채널 파일과 별도 폴더에 받아 나중에 합친다.
cd "$(dirname "$0")"
PY=./.venv/bin/python
for R in "20250801 20250930" "20240801 20240930" "20230801 20230930" \
         "20250615 20250714" "20240624 20240630" "20260801 20260809"; do
  set -- ${=R}
  $PY collect.py sat --start $1 --end $2 --channels IR105,IR123 2>&1 | grep -E "^\[sat\]" | tail -1
done
echo "=== split-window 수집 완료 $(date +%H:%M) ==="
