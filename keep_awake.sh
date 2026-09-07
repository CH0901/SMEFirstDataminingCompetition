#!/bin/zsh
# 뚜껑을 닫아도 수집이 계속 돌게 한다. 반드시 sudo 로 실행할 것.
#
#   sudo ~/Desktop/sme/keep_awake.sh
#
# caffeinate 는 idle sleep 만 막고 뚜껑 닫힘(lid-close sleep)은 못 막는다.
# 그건 pmset disablesleep 이라야 하는데 root 권한이 필요하다.
#
# 수집이 끝나면(또는 Ctrl-C 로 끊으면) 절전 설정을 원래대로 되돌린다.
# 안 그러면 맥이 영영 안 자게 되어 배터리가 녹는다.

if [[ $EUID -ne 0 ]]; then
  echo "sudo 로 실행하세요:  sudo $0"
  exit 1
fi

restore() {
  pmset -a disablesleep 0
  echo "\n[keep_awake] 절전 설정 원복 완료. 이제 뚜껑 닫으면 정상적으로 잠듭니다."
}
trap restore EXIT INT TERM

pmset -a disablesleep 1
echo "[keep_awake] 절전 비활성화. 뚜껑 닫아도 수집이 계속 돕니다."
echo "[keep_awake] 전원 연결 확인하세요. 끝나면 자동 원복됩니다. (중단: Ctrl-C)"

while true; do
  if ! pgrep -f "run_collect.sh" > /dev/null \
     && ! pgrep -f "collect.py" > /dev/null; then
    break
  fi
  BATT=$(pmset -g batt | grep -o '[0-9]*%' | head -1)
  printf "\r[keep_awake] 수집 진행 중... 배터리 %s  " "$BATT"
  sleep 30
done

echo "\n[keep_awake] 수집이 끝났습니다."
