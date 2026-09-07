#!/bin/zsh
# ASOS 가 끝나기를 기다렸다가 위성 수집을 이어서 돌린다.
# 두 작업이 겹치면 초당 10회 제한을 같이 나눠 쓰게 되므로 순서를 지킨다.
cd "$(dirname "$0")"
PY=./.venv/bin/python

while pgrep -f "collect.py asos" > /dev/null; do sleep 20; done
echo "=== ASOS 완료 확인. 위성 수집 시작 $(date +%H:%M) ==="

for Y in 2023 2024 2025; do
  echo "--- 위성 ${Y}년 6~9월 ---"
  $PY collect.py sat --start ${Y}0601 --end ${Y}0930 2>&1 | grep -vE "NotOpenSSL|warnings.warn"
done
echo "=== 전체 수집 완료 $(date +%H:%M) ==="
ls -la data/*.parquet
