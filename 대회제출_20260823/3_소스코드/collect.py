"""학습 데이터 수집: GK2A 위성 피처 + ASOS 레이블.

핵심 설계 — 위성 파일을 디스크에 쌓지 않는다.
  다운로드(메모리) -> 96개 지점 창만 추출 -> 원본 폐기
900x900 화소 중 실제로 쓰는 건 0.1% 도 안 되므로, 이렇게 하면
수 GB 를 받아도 최종 저장은 수십 MB 로 끝난다.

시간대 주의:
  ASOS TM   = KST
  GK2A date = UTC
둘을 붙일 때 반드시 변환한다 (KST = UTC + 9h).

사용:
  python collect.py sat  --start 20250601 --end 20250831
  python collect.py asos --start 20250601 --end 20250831
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import kma
import stations
from geo import (extract_windows, lonlat_to_rowcol, satellite_zenith,
                 solar_position, window_features)

OUT = Path(__file__).with_name("data")
OUT.mkdir(exist_ok=True)

# 받을 시각(UTC). 규정상 대상 시각(14:00 KST = 05:00 UTC)'까지' 관측된
# 영상만 입력으로 쓸 수 있다. 그래서 05 UTC 이하를 대상 시각 쪽으로
# 촘촘히 잡고, 18/21 UTC 는 '전날 늦은 시각'으로 쓴다 (오늘 05 UTC 보다
# 이르므로 합법이며, 하루 사이 변화를 담는다).
DEFAULT_HOURS_UTC = [0, 2, 4, 5, 18, 21]


def collect_satellite(start: str, end: str, channels: list[str],
                      hours: list[int], half: int, workers: int = 8) -> Path:
    stn = stations.load()
    lon, lat = stn["LON"].to_numpy(), stn["LAT"].to_numpy()
    days = pd.date_range(start, end, freq="D")
    print(f"[sat] 지점 {len(stn)}개 · 채널 {len(channels)}개 · "
          f"하루 {len(hours)}시각 · 창 {2*half+1}x{2*half+1} · {len(days)}일")

    # GK2A 는 정지위성이라 지점별 관측 기하가 시간과 무관한 상수다.
    sat_zen = satellite_zenith(lon, lat)

    rowcol_cache: dict[tuple, tuple] = {}
    n_fail = 0
    # 수 시간짜리 작업이라 중간에 끊길 수 있다. 하루씩 저장해두면
    # 다시 실행할 때 이미 받은 날은 건너뛰고 이어받는다.
    # 채널 구성이 다르면 폴더를 나눈다 (기존 수집물을 덮지 않도록)
    # 기본 채널·기본 창이면 학습이 읽는 폴더에 바로 쓴다.
    # (예전에는 sat_days 에 썼는데, 채널 구성이 바뀐 뒤로 그 폴더에는
    #  옛 채널 파일이 남아 있어 섞이면 스키마가 어긋난다.)
    tag = "sat_final" if set(channels) == set(kma.DEFAULT_CHANNELS) and half == 4 \
        else "sat_" + "_".join(sorted(channels)).lower() + (f"_h{half}" if half != 4 else "")
    day_dir = OUT / tag
    day_dir.mkdir(exist_ok=True)

    for di, day in enumerate(days, 1):
        p_day = day_dir / f"{day:%Y%m%d}.parquet"
        if p_day.exists():
            continue
        # 하루치 (시각 x 채널) 를 한꺼번에 병렬로 받는다. 순차로 하면
        # 왕복 지연에 묶여 초당 1회밖에 안 나온다 (제한은 10회).
        jobs = [(h, ch) for h in hours for ch in channels]
        blobs: dict[tuple, bytes] = {}
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(kma.fetch_gk2a, ch,
                          (day + pd.Timedelta(hours=h)).strftime("%Y%m%d%H%M")):
                    (h, ch)
                for h, ch in jobs
            }
            for fut in as_completed(futs):
                try:
                    blobs[futs[fut]] = fut.result()
                except Exception:
                    n_fail += 1  # 이 채널은 NaN 으로 남는다

        recs = []
        for h in hours:
            ts = day + pd.Timedelta(hours=h)
            sol_zen, sol_az = solar_position(ts, lon, lat)
            rec = {
                "time_utc": ts,
                "STN": stn["STN"].to_numpy(),
                "SAT_ZEN": sat_zen,
                "SOL_ZEN": sol_zen,
                "SOL_AZ": sol_az,
                # 태양고도 코사인. 음수면 야간, 주간엔 지표 가열량에 비례한다.
                "SOL_COS": np.cos(np.radians(sol_zen)),
            }
            for ch in channels:
                raw = blobs.get((h, ch))
                if raw is None:
                    continue
                img, meta = kma.read_gk2a(raw)

                # 채널 해상도별로 화소 좌표가 다르므로 격자 형태를 키로 캐시
                key = (img.shape, float(meta["pixel_size"]))
                if key not in rowcol_cache:
                    rowcol_cache[key] = lonlat_to_rowcol(lon, lat, meta)
                r, c = rowcol_cache[key]

                win = extract_windows(img, r, c, half=half)
                rec.update(window_features(win, ch))
            recs.append(pd.DataFrame(rec))
        del blobs  # 원본은 바로 버린다 (하루치 ~80MB)

        pd.concat(recs, ignore_index=True).to_parquet(p_day, index=False)
        if di % 5 == 0 or di == len(days):
            print(f"  {di}/{len(days)}일  ({day:%Y-%m-%d})  누적실패 {n_fail}",
                  flush=True)

    want = {f"{d:%Y%m%d}.parquet" for d in days}
    df = pd.concat(
        [pd.read_parquet(p) for p in sorted(day_dir.glob("*.parquet"))
         if p.name in want],
        ignore_index=True,
    )
    p = OUT / f"sat_{start}_{end}.parquet"
    df.to_parquet(p, index=False)
    print(f"[sat] {len(df):,}행 -> {p}  ({p.stat().st_size/1e6:.1f} MB)")
    return p


def collect_asos(start: str, end: str) -> Path:
    df = kma.fetch_asos_range(start, end)
    p = OUT / f"asos_{start}_{end}.parquet"
    df.to_parquet(p, index=False)
    print(f"[asos] {len(df):,}행 · 지점 {df['STN'].nunique()}개 -> {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["sat", "asos"])
    ap.add_argument("--start", required=True, help="YYYYMMDD")
    ap.add_argument("--end", required=True, help="YYYYMMDD")
    ap.add_argument("--channels", default=",".join(kma.DEFAULT_CHANNELS))
    ap.add_argument("--hours", default=",".join(map(str, DEFAULT_HOURS_UTC)))
    ap.add_argument("--half", type=int, default=4, help="창 반경(화소)")
    ap.add_argument("--workers", type=int, default=8, help="동시 요청 수")
    a = ap.parse_args()

    if a.what == "sat":
        collect_satellite(a.start, a.end, a.channels.split(","),
                          [int(h) for h in a.hours.split(",")], a.half,
                          a.workers)
    else:
        collect_asos(a.start, a.end)


if __name__ == "__main__":
    main()
