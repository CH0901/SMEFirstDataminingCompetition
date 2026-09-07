"""저장소 안의 표준 경로.

스크립트가 어느 패키지 깊이에 있든 같은 곳을 가리키도록 저장소 루트를
`__file__` 에서 거슬러 올라가 계산한다. `data/` 아래를 가리키는 상대경로
문자열은 저장소 루트에서 실행하는 것을 전제로 한다.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data"                 # 수집한 위성·ASOS 자료
REFERENCE = DATA / "reference"       # 지점 목록 등 손으로 관리하는 표
OUTPUTS = ROOT / "outputs"           # 제출 파일 등 산출물
API_KEY_FILE = ROOT / "apikey.txt"   # 커밋 대상 아님. README 의 API 키 설정 참고
MODEL_FILE = ROOT / "model.pkl"      # 학습 결과. 크기 때문에 커밋하지 않는다
