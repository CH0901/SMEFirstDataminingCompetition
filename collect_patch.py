"""원본 패치를 그대로 저장하는 수집기.

기존 collect.py 는 다운로드 직후 9x9 통계만 남기고 원본을 버린다. 디스크는
아끼지만, 창 크기나 통계 정의를 바꿀 때마다 전부 다시 받아야 한다. 오늘
넓은 창 실험에 그 대가를 치렀다.

여기서는 지점 주변 41x41(=82km) 패치를 uint16 그대로 저장한다.
  96지점 x 41x41 x 3채널 x 6시각 x 2바이트 = 약 5.8MB/일
  227일 = 1.3GB  — 디스크 부담은 사실상 없다.
이렇게 두면 이후 어떤 피처를 만들든 재다운로드가 필요 없다.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import kma
import stations
from geo import lonlat_to_rowcol

OUT = Path(__file__).with_name("data") / "patches"
OUT.mkdir(parents=True, exist_ok=True)
HALF = 20                       # 41x41 = 82km. 어떤 창 크기든 여기서 잘라 쓴다
HOURS = [0, 2, 4, 5, 18, 21]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--channels", default=",".join(kma.DEFAULT_CHANNELS))
    a = ap.parse_args()
    channels = a.channels.split(",")

    stn = stations.load()
    lon, lat = stn.LON.to_numpy(), stn.LAT.to_numpy()
    days = pd.date_range(a.start, a.end, freq="D")
    w = 2 * HALF + 1
    cache: dict = {}
    print(f"[patch] {len(days)}일 · 채널 {channels} · 패치 {w}x{w}")

    for di, day in enumerate(days, 1):
        p = OUT / f"{day:%Y%m%d}.npz"
        if p.exists():
            continue
        jobs = [(h, ch) for h in HOURS for ch in channels]
        blobs = {}
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(kma.fetch_gk2a, ch,
                              (day + pd.Timedelta(hours=h)).strftime("%Y%m%d%H%M")):
                    (h, ch) for h, ch in jobs}
            for f in as_completed(futs):
                try:
                    blobs[futs[f]] = f.result()
                except Exception:
                    pass

        # (시각, 채널) -> (지점, w, w). 결측은 0 으로 두고 mask 로 표시한다.
        arr = np.zeros((len(HOURS), len(channels), len(stn), w, w), np.uint16)
        mask = np.zeros((len(HOURS), len(channels)), bool)
        for (h, ch), raw in blobs.items():
            img, meta = kma.read_gk2a(raw)
            key = (img.shape, float(meta["pixel_size"]))
            if key not in cache:
                cache[key] = lonlat_to_rowcol(lon, lat, meta)
            r, c = (np.rint(v).astype(int) for v in cache[key])
            hi, ci = HOURS.index(h), channels.index(ch)
            for i in range(len(stn)):
                r0, c0 = r[i] - HALF, c[i] - HALF
                if r0 >= 0 and c0 >= 0 and r0 + w <= img.shape[0] and c0 + w <= img.shape[1]:
                    arr[hi, ci, i] = img[r0:r0 + w, c0:c0 + w]
            mask[hi, ci] = True

        np.savez_compressed(p, patches=arr, mask=mask,
                            hours=np.array(HOURS), channels=np.array(channels),
                            stn=stn.STN.to_numpy())
        if di % 10 == 0 or di == len(days):
            print(f"  {di}/{len(days)}일  ({day:%Y-%m-%d})  "
                  f"수신 {int(mask.sum())}/{len(jobs)}칸", flush=True)


if __name__ == "__main__":
    main()
