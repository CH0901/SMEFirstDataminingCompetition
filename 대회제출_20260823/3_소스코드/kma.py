"""기상청 API 허브 클라이언트.

ASOS 지상관측 + GK2A 위성 자료를 받아온다.
API 키는 환경변수 KMA_API_KEY 또는 같은 폴더의 apikey.txt 에서 읽는다.
"""
from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HUB = "https://apihub.kma.go.kr"

# 서버가 초당 10회로 제한한다 (X-RateLimit-Replenish-Rate: 10).
# 여유를 두고 8회/초로 잡는다. 여러 스레드가 공유하므로 잠금이 필요하다.
_MIN_INTERVAL = 1.0 / 8
_last_call = 0.0
_throttle_lock = threading.Lock()


def api_keys() -> list[str]:
    """사용 가능한 API 키들. apikey.txt 에 한 줄에 하나씩 적는다.

    키마다 활용신청된 자료가 다르므로(같은 URL도 키에 따라 200/403 이 갈린다)
    여러 개를 등록해두고 403 이 나면 다음 키로 넘어간다.
    """
    env = os.environ.get("KMA_API_KEY", "")
    keys = [k.strip() for k in env.split(",") if k.strip()]
    f = Path(__file__).with_name("apikey.txt")
    if f.exists():
        for line in f.read_text().splitlines():
            line = line.split("#")[0].strip()
            if line and line not in keys:
                keys.append(line)
    if not keys:
        raise RuntimeError(
            "API 키가 없습니다. apikey.txt 에 키를 한 줄에 하나씩 적으세요."
        )
    return keys


def _throttle() -> None:
    """전역 요청 간격을 지킨다. 잠금 안에서 다음 슬롯을 예약하고 밖에서 잔다."""
    global _last_call
    with _throttle_lock:
        now = time.monotonic()
        slot = max(now, _last_call + _MIN_INTERVAL)
        _last_call = slot
        wait = slot - now
    if wait > 0:
        time.sleep(wait)


def _is_denied(body: bytes) -> bool:
    """권한 없음은 HTTP 200 안에 JSON 으로 실려 온다."""
    return len(body) < 500 and b'"status" : 403' in body


def _get(url: str, params: dict, *, tries: int = 10, timeout: int = 8) -> bytes:
    """재시도 + 키 폴백 GET. 모든 키가 403 이면 PermissionError.

    이 API 는 실패할 때 HTTP 오류를 주지 않고 연결이 그대로 멈춘다.
    성공하면 0.3초면 오지만, 실패하면 응답이 영영 오지 않는다. 그래서 오래
    기다리는 것은 순수한 낭비이고, 짧게 끊고 새 연결로 다시 붙는 편이
    빠르면서 성공률도 높다.
      실측(10장): 5초x8회 -> 10/10 성공 · 15초x4회 -> 8/10 성공
    """
    keys = api_keys()
    last, denied = None, 0
    for key in keys:
        p = {**params, "authKey": key}
        key_denied = False
        for _ in range(tries):
            _throttle()
            try:
                r = requests.get(url, params=p, timeout=timeout)
                if r.status_code == 200:
                    if _is_denied(r.content):
                        key_denied = True   # 권한 없음·한도 초과
                        break               # 재시도해도 소용없다
                    return r.content
                last = f"HTTP {r.status_code}"
            except Exception as e:          # 무응답 포함
                last = repr(e)
            time.sleep(0.3)
        if key_denied:
            denied += 1
    if denied == len(keys):
        raise PermissionError(
            f"등록된 키 {len(keys)}개 모두 권한이 없습니다: {url}\n"
            "  → API 허브에서 해당 자료의 'API 활용신청' 을 하세요."
        )
    raise RuntimeError(f"요청 실패: {url} :: {last}")


# ---------------------------------------------------------------- ASOS

# kma_sfctm2 응답의 46개 컬럼 (헤더 주석 순서 그대로)
ASOS_COLS = [
    "TM", "STN", "WD", "WS", "GST_WD", "GST_WS", "GST_TM", "PA", "PS", "PT",
    "PR", "TA", "TD", "HM", "PV", "RN", "RN_DAY", "RN_JUN", "RN_INT",
    "SD_HR3", "SD_DAY", "SD_TOT", "WC", "WP", "WW", "CA_TOT", "CA_MID",
    "CH_MIN", "CT", "CT_TOP", "CT_MID", "CT_LOW", "VS", "SS", "SI", "ST_GD",
    "TS", "TE_005", "TE_01", "TE_02", "TE_03", "ST_SEA", "WH", "BF", "IR",
    "IX",
]

# 결측 코드. 컬럼마다 다르게 쓰이므로 넉넉히 잡는다.
_MISSING = {-9.0, -99.0, -999.0, -9999.0}

# 문자열로 남겨야 하는 컬럼 (운형, 일기코드 등)
_STR_COLS = {"WW", "CT"}


def fetch_asos_hour(tm: str, stn: int | str = 0) -> pd.DataFrame:
    """ASOS 시간자료 한 시각치. tm 은 'YYYYMMDDHHMM' (KST).

    stn=0 이면 전 지점(96개)이 한 번에 온다.
    """
    raw = _get(
        f"{HUB}/api/typ01/url/kma_sfctm2.php",
        {"tm": tm, "stn": stn, "help": 0},
    )
    return _parse_asos(raw)


def _parse_asos(raw: bytes) -> pd.DataFrame:
    text = raw.decode("euc-kr", errors="replace")
    rows = [
        ln.split()
        for ln in text.splitlines()
        if ln and not ln.startswith("#") and not ln.startswith("7777")
    ]
    rows = [r for r in rows if len(r) == len(ASOS_COLS)]
    if not rows:
        return pd.DataFrame(columns=ASOS_COLS)

    df = pd.DataFrame(rows, columns=ASOS_COLS)
    for c in df.columns:
        if c in _STR_COLS:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 결측 코드를 NaN 으로. -99 를 기온으로 학습시키면 모델이 망가진다.
    num = [c for c in df.columns if c not in _STR_COLS]
    df[num] = df[num].mask(df[num].isin(_MISSING))

    df["TM"] = pd.to_datetime(df["TM"].astype("Int64").astype(str), format="%Y%m%d%H%M")
    df["STN"] = df["STN"].astype("Int64")
    return df


def fetch_asos_range(start: str, end: str, *, freq: str = "h",
                     workers: int = 8, progress: bool = True) -> pd.DataFrame:
    """[start, end] 구간의 ASOS 를 시각별로 긁어 붙인다. 날짜는 'YYYYMMDD'.

    sfctm2 는 한 요청에 한 시각뿐이라 시각 수만큼 호출해야 한다. 순차로 하면
    왕복 지연에 묶여 초당 0.3회밖에 안 나오므로(제한은 10회) 스레드로 겹친다.
    실제 발사 간격은 _throttle 이 전역으로 지켜준다.
    """
    stamps = pd.date_range(f"{start} 00:00", f"{end} 23:00", freq=freq)
    out, failed, done = [], [], 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_asos_hour, ts.strftime("%Y%m%d%H%M")): ts
                for ts in stamps}
        for fut in as_completed(futs):
            done += 1
            try:
                out.append(fut.result())
            except Exception as e:
                failed.append((futs[fut].strftime("%Y%m%d%H%M"), repr(e)))
            if progress and (done % 200 == 0 or done == len(stamps)):
                got = sum(len(d) for d in out)
                print(f"  ASOS {done}/{len(stamps)} 시각  누적 {got:,}행  "
                      f"실패 {len(failed)}", flush=True)

    if failed:
        print(f"  [주의] {len(failed)}개 시각 실패. 예: {failed[:3]}")
    if not out:
        return pd.DataFrame(columns=ASOS_COLS)
    df = pd.concat(out, ignore_index=True)
    return df.sort_values(["TM", "STN"]).reset_index(drop=True)


# ---------------------------------------------------------------- 지점정보

def fetch_station_info(inf: str = "SFC", tm: str | None = None) -> pd.DataFrame:
    """관측 지점정보(위경도·고도). inf='SFC'(남한 지상) / 'NKO'(북한) 등.

    tm 은 '어느 시점의 지점 구성인지' 를 고르는 스냅샷 날짜다. 관측소가
    신설·폐지되므로 시점에 따라 지점 수가 달라진다 (2022년 96개, 2025년 97개).
    기본값은 오늘 — 최신 구성을 받아둔다.

    응답 뒤쪽에 주소가 공백을 포함한 채 붙어 있어서 단순 split 으로는
    컬럼 수가 들쭉날쭉하다. 앞쪽 고정 필드만 위치로 읽는다.
      0:STN  1:LON  2:LAT  3:STN_SP  4:HT  5:HT_PA  6:HT_TA  7:HT_WD  8:HT_RN
    """
    if tm is None:
        tm = pd.Timestamp.now().strftime("%Y%m%d") + "0900"
    raw = _get(
        f"{HUB}/api/typ01/url/stn_inf.php",
        {"inf": inf, "stn": "", "tm": tm, "help": 0},
    )
    text = raw.decode("euc-kr", errors="replace")
    recs = []
    for ln in text.splitlines():
        if not ln or ln.startswith("#") or ln.startswith("7777"):
            continue
        t = ln.split()
        if len(t) < 9 or not t[0].lstrip("-").isdigit():
            continue
        # 지점명(한글)은 숫자 필드가 끝난 뒤 처음 나오는 비수치 토큰
        name = next(
            (x for x in t[9:] if not x.replace(".", "").replace("-", "").isdigit()),
            None,
        )
        recs.append((t[0], t[1], t[2], t[4], name))

    if not recs:
        raise RuntimeError("지점정보 응답을 파싱하지 못했습니다.")
    df = pd.DataFrame(recs, columns=["STN", "LON", "LAT", "HT", "NAME"])
    for c in ("STN", "LON", "LAT", "HT"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["STN", "LON", "LAT"]).reset_index(drop=True)


# ---------------------------------------------------------------- GK2A 위성

# LE1B 기본관측 16채널과 KO 영역 해상도.
# 채널마다 격자 크기가 다르므로 픽셀 좌표를 따로 계산해야 한다.
GK2A_CHANNELS = {
    "VI004": 1.0, "VI005": 1.0, "VI006": 0.5, "VI008": 1.0,
    "NR013": 2.0, "NR016": 2.0, "SW038": 2.0,
    "WV063": 2.0, "WV069": 2.0, "WV073": 2.0,
    "IR087": 2.0, "IR096": 2.0, "IR105": 2.0,
    "IR112": 2.0, "IR123": 2.0, "IR133": 2.0,
}

# 채널 3개. KMA 공식 GK2A 지표면온도 알고리즘(ATBD)이 쓰는 분리대기창
# 쌍에 단파적외를 더한 구성이다.
#   IR105 10.5um  분리대기창 T13 - 지표 온도 주신호
#   IR123 12.3um  분리대기창 T15 - IR105 와의 차이가 수증기 흡수를 상쇄
#   SW038  3.8um  단파적외 - 주야 구름·안개 판별
#
# 두 창채널의 파장이 가까워 지표 신호는 거의 같지만 수증기 흡수량이 달라,
# 차이를 빼면 대기 효과가 지워진다. 트리 단독일 때는 이 선형 조합을 표현하지
# 못해 가치가 안 보였고, 그래서 이전 선택(IR087/NR016/SW038)에서 빠졌다.
# 선형 성분을 앞에 둔 뒤 다시 재니 같은 다운로드 비용에서 10% 더 좋았다
# (계절 안 2.187 -> 1.961).
DEFAULT_CHANNELS = ["IR105", "IR123", "SW038"]

# 2km 해상도 전체(채널 탐색용)
ALL_2KM_CHANNELS = [c for c, r in GK2A_CHANNELS.items() if r == 2.0]


def fetch_gk2a(channel: str, dt: str, area: str = "KO") -> bytes:
    """GK2A LE1B 한 채널 한 시각. dt 는 'YYYYMMDDHHMM' (UTC).

    area: KO(한반도) / EA(동아시아) / FD(전구).
    KO 는 FD 의 1/30 용량이고 96개 지점을 모두 포함한다.
    """
    return _get(
        f"{HUB}/api/typ05/api/GK2A/LE1B/{channel}/{area}/data",
        {"date": dt},
    )   # 타임아웃·재시도는 _get 의 기본값(8초 x 10회)을 쓴다


def read_gk2a(raw: bytes) -> tuple[np.ndarray, dict]:
    """NetCDF 바이트를 (화소배열, 투영정보) 로 연다. 디스크에 쓰지 않는다."""
    import netCDF4  # 지연 임포트 (Kaggle 환경에서 없을 수도 있어서)

    ds = netCDF4.Dataset("inmem", mode="r", memory=raw)
    try:
        img = np.asarray(ds["image_pixel_values"][:])
        meta = {a: getattr(ds, a) for a in ds.ncattrs()}
    finally:
        ds.close()
    return img, meta
