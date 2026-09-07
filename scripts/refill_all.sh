#!/bin/zsh
# 광역 복구가 끝나면 기존 9x9 데이터의 결측도 채운다.
# 특히 2026-08 이 33% 비어 있는데, 채점 시기와 가장 가까운 구간이라 중요하다.
cd "$(dirname "$0")/.."
while pgrep -f "sme.collect.refill data/sat_ir105_ir123_sw038_h12" > /dev/null; do sleep 20; done
echo "=== 광역 복구 완료. 9x9 복구 시작 $(date +%H:%M) ==="
./.venv/bin/python -u -m sme.collect.refill data/sat_final 4 2>&1 | grep -vE "NotOpenSSL|warnings.warn"
echo "=== 전체 복구 완료 $(date +%H:%M) ==="
