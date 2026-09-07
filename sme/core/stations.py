"""ASOS 지점 좌표 관리.

위성 화소를 찍으려면 지점별 위경도가 필요한데, sfctm2 응답에는 좌표가 없다.
좌표는 stn_inf.php(= '지상관측 지점정보' 활용신청 필요)에서 온다.

승인 전에도 개발이 막히지 않도록 stations.csv 를 캐시로 쓴다.
승인되면 `python stations.py --update` 한 번이면 전체가 채워진다.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import pandas as pd
from sme.core import paths

CSV = paths.REFERENCE / "stations.csv"

# 승인 전 스모크테스트용 씨앗. 좌표 검증이 끝난 지점만 넣어둔다.
# 전체 96개는 --update 로 받아야 한다. (임의로 채워넣으면 2km 격자에서
# 0.02도 오차만 나도 화소가 통째로 어긋난다.)
_SEED = pd.DataFrame(
    [
        (90,  128.5647, 38.2508, 18.06,  "속초"),
        (105, 128.8910, 37.7515, 26.04,  "강릉"),
        (108, 126.9658, 37.5714, 85.50,  "서울"),
        (112, 126.6249, 37.4776, 68.99,  "인천"),
        (133, 127.3721, 36.3720, 68.94,  "대전"),
        (143, 128.6186, 35.8251, 53.50,  "대구"),
        (156, 126.8916, 35.1729, 72.38,  "광주"),
        (159, 129.0320, 35.1047, 69.56,  "부산"),
        (184, 126.5297, 33.5141, 20.45,  "제주"),
        (100, 128.7183, 37.6771, 772.57, "대관령"),
    ],
    columns=["STN", "LON", "LAT", "HT", "NAME"],
)


def load(*, require_full: bool = False, n_expected: int = 96) -> pd.DataFrame:
    """지점 좌표표를 돌려준다. 없으면 씨앗으로 만들어 둔다."""
    if not CSV.exists():
        _SEED.to_csv(CSV, index=False)
        print(f"[stations] 씨앗 {len(_SEED)}개로 {CSV.name} 생성 "
              f"(전체 {n_expected}개는 --update 필요)")
    df = pd.read_csv(CSV)
    if require_full and len(df) < n_expected:
        raise SystemExit(
            f"[stations] 좌표가 {len(df)}/{n_expected}개뿐입니다.\n"
            "  → API 허브에서 '지상관측 → 지상관측 지점정보' 활용신청 후\n"
            "     python stations.py --update 를 실행하세요."
        )
    return df


def update() -> pd.DataFrame:
    """stn_inf 에서 전체 지점 좌표를 받아 stations.csv 를 갱신한다."""
    from sme.core.kma_client import fetch_station_info

    info = fetch_station_info()
    cols = [c for c in ("STN", "LON", "LAT", "HT", "NAME") if c in info.columns]
    out = info[cols].copy()
    out = out.dropna(subset=["STN", "LON", "LAT"]).drop_duplicates("STN")
    out["STN"] = out["STN"].astype(int)
    out = out.sort_values("STN").reset_index(drop=True)
    out.to_csv(CSV, index=False)
    print(f"[stations] {len(out)}개 지점 갱신 -> {CSV}")
    return out


def from_portal_csv(path: str | Path) -> pd.DataFrame:
    """기상자료개방포털(data.kma.go.kr) 에서 받은 지점정보 CSV 를 읽어들인다.

    활용신청 없이도 쓸 수 있는 우회로. 포털 CSV 는 보통 EUC-KR 이고
    컬럼명이 한글이라 이름으로 찾아 매핑한다.
    """
    path = Path(path)
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "euc-kr", "cp949", "utf-8"):
        try:
            df = pd.read_csv(io.BytesIO(raw), encoding=enc)
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
    else:
        raise SystemExit(f"CSV 인코딩을 못 읽었습니다: {path}")

    def pick(*keys: str) -> str:
        for k in keys:
            for c in df.columns:
                if k in str(c).replace(" ", ""):
                    return c
        raise SystemExit(
            f"'{keys[0]}' 컬럼을 못 찾았습니다. 실제 컬럼: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "STN": pd.to_numeric(df[pick("지점번호", "지점")], errors="coerce"),
        "LON": pd.to_numeric(df[pick("경도")], errors="coerce"),
        "LAT": pd.to_numeric(df[pick("위도")], errors="coerce"),
        "HT":  pd.to_numeric(df[pick("해발고도", "노장해발고도", "고도")],
                             errors="coerce"),
    })
    try:
        out["NAME"] = df[pick("지점명")].astype(str)
    except SystemExit:
        pass

    out = out.dropna(subset=["STN", "LON", "LAT"])
    out["STN"] = out["STN"].astype(int)
    # 이력 자료라 한 지점이 여러 행일 수 있다. 최근 기록을 남긴다.
    out = out.drop_duplicates("STN", keep="last").sort_values("STN")
    out = out.reset_index(drop=True)
    out.to_csv(CSV, index=False)
    print(f"[stations] 포털 CSV 에서 {len(out)}개 지점 -> {CSV}")
    return out


if __name__ == "__main__":
    if "--update" in sys.argv:
        try:
            update()
        except PermissionError as e:
            raise SystemExit(
                f"{e}\n  → '지상관측 지점정보' 활용신청이 아직 안 됐습니다."
            )
    elif "--from-csv" in sys.argv:
        from_portal_csv(sys.argv[sys.argv.index("--from-csv") + 1])
    else:
        df = load()
        print(df.to_string(index=False))
